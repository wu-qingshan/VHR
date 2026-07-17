import copy
import math
import torch
import transformers
from transformers.generation.utils import *
try:
    from transformers.generation.utils import _split_model_inputs  # transformers <= 4.50
except ImportError:  # 新版 transformers (>=4.51) 已移除该私有符号
    def _split_model_inputs(*args, **kwargs):  # type: ignore[no-redef]
        raise NotImplementedError(
            "_split_model_inputs is unavailable in current transformers version; "
            "this branch is only used by low_memory beam_search and is not needed for VHR."
        )
from transformers.generation.configuration_utils import *

from vhr import replace_attn as replace_attn_vhr, set_directional_active, set_vhr_phase
from vcrp import collect_vcrp_step_stats, replace_vcrp_decoder_layers

__version__ = "4.45.0"

GenerationMode.VHR_GENERATION = "vhr_generation"
GenerationMode.VCD_GENERATION = "vcd_generation"
GenerationMode.OPERA_GENERATION = "opera_generation"
GenerationMode.CODE_GENERATION = "code_generation"


def compute_contrasted_logits(primary_logits, contrast_logits, alpha, positive_delta=False):
    """Apply standard two-sided or positive-delta-only logit extrapolation."""
    delta = primary_logits - contrast_logits
    if positive_delta:
        delta = torch.clamp_min(delta, 0.0)
    return primary_logits + alpha * delta


def apply_disagreement_gate(primary_logits, contrast_logits, contrasted_logits):
    """Use contrasted logits only for batch items whose branch argmax differs."""
    disagreement = primary_logits.argmax(dim=-1, keepdim=True) != contrast_logits.argmax(
        dim=-1, keepdim=True
    )
    return torch.where(disagreement, contrasted_logits, primary_logits), disagreement


def apply_visual_evidence_gate(primary_logits, base_logits, text_logits, contrasted_logits):
    """Allow a proposed SC switch only when its relative visual support is non-negative."""
    primary_top = primary_logits.argmax(dim=-1, keepdim=True)
    proposed_top = contrasted_logits.argmax(dim=-1, keepdim=True)
    proposed_switch = proposed_top != primary_top
    visual_support = base_logits - text_logits
    proposed_support = visual_support.gather(-1, proposed_top)
    primary_support = visual_support.gather(-1, primary_top)
    support_margin = proposed_support - primary_support
    allowed_switch = proposed_switch & (support_margin >= 0)
    routed_logits = torch.where(allowed_switch, contrasted_logits, primary_logits)
    return routed_logits, proposed_switch, allowed_switch, support_margin


def _vhr_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **dola decoding** and can be
    used for decoder-only text models.
    The method is based on the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
    Models" (https://arxiv.org/abs/2309.03883) in ICLR 2024.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
        or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values

    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    image_start = generation_config.vhr_image_start
    image_end = generation_config.vhr_image_end
    contrast_mode = generation_config.vhr_contrast_mode
    score_mode = generation_config.vhr_score_mode
    directional_delay = generation_config.vhr_directional_delay
    images_noisy = generation_config.vhr_images_noisy
    images_cd = generation_config.images_cd
    use_vcd = images_cd is not None
    use_base_contrast = generation_config.vhr_base_contrast
    use_disagreement_gate = generation_config.vhr_disagreement_gate
    use_switch_diagnostics = generation_config.vhr_switch_diagnostics
    use_visual_evidence_gate = generation_config.vhr_visual_evidence_gate
    use_vcrp_diagnostic = generation_config.vhr_vcrp_diagnostic
    use_vcrp_intervention = generation_config.vhr_vcrp_intervention
    use_vcrp = use_vcrp_diagnostic or use_vcrp_intervention
    use_aux_contrast = use_vcd or use_base_contrast
    if use_vcd and use_base_contrast:
        raise ValueError("Choose either noisy-image VCD or VHR self-contrast, not both")
    if use_vcd and score_mode == "consensus":
        raise ValueError("Full VCD is not supported with the abandoned CC-VHR consensus score")
    if use_disagreement_gate and not use_base_contrast:
        raise ValueError("Disagreement gating requires VHR self-contrast")
    if use_switch_diagnostics and not use_base_contrast:
        raise ValueError("Switch diagnostics require VHR self-contrast")
    if use_visual_evidence_gate and not use_base_contrast:
        raise ValueError("Visual-evidence gating requires VHR self-contrast")
    if use_visual_evidence_gate and (use_disagreement_gate or generation_config.vhr_positive_delta_contrast):
        raise ValueError("Visual-evidence gating cannot be combined with DG or positive-delta contrast")

    disagreement_total_steps = 0
    disagreement_contrast_steps = 0
    switch_total_steps = 0
    switch_branch_disagreement_steps = 0
    switch_selected_steps = 0
    switch_same_argmax_steps = 0
    switch_disagreement_steps = 0
    switch_disagreement_no_switch_steps = 0
    switch_same_selected_delta_sum = 0.0
    switch_same_vhr_delta_sum = 0.0
    switch_same_margin_sum = 0.0
    visual_total_steps = 0
    visual_proposed_switch_steps = 0
    visual_allowed_switch_steps = 0
    visual_blocked_switch_steps = 0
    visual_allowed_margin_sum = 0.0
    visual_blocked_margin_sum = 0.0
    vcrp_token_stats = []

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    prompt_length = cur_len
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    model_kwargs_tc = None
    model_kwargs_cd = None
    input_ids_tc = None
    outputs_tc = None

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        vcrp_current_stats = None
        if score_mode == "delayed_directional_linear":
            set_directional_active(self, cur_len - prompt_length >= directional_delay)
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if model_kwargs_tc is None:
            input_ids_tc = input_ids.clone()
            model_kwargs_tc = copy.deepcopy(model_kwargs)

            if image_start > 0:
                if contrast_mode == "zero":
                    model_kwargs_tc['pixel_values'] = torch.zeros_like(model_kwargs_tc['pixel_values'])
                else:
                    model_kwargs_tc['pixel_values'] = None
                    model_kwargs_tc['image_sizes'] = None
                    input_ids_tc = torch.cat((input_ids_tc[:, :image_start], input_ids_tc[:, image_end:]), dim=1)
                    attention_mask = model_kwargs_tc['attention_mask']
                    model_kwargs_tc['attention_mask'] = torch.cat((attention_mask[:, :image_start], attention_mask[:, image_end:]), dim=1)
            else:
                # instructBLIP generation
                if contrast_mode == "zero":
                    model_kwargs_tc['inputs_embeds'] = model_kwargs_tc['inputs_embeds'].clone()
                    model_kwargs_tc['inputs_embeds'][:, :32] = 0
                else:
                    model_kwargs_tc['inputs_embeds'] = model_kwargs_tc['inputs_embeds'][:, 32:]
                    model_kwargs_tc['attention_mask'] = model_kwargs_tc['attention_mask'][:, 32:]
            
            model_kwargs_tc = self._get_initial_cache_position(input_ids_tc, model_kwargs_tc)

            model_inputs_tc = self.prepare_inputs_for_generation(input_ids_tc, **model_kwargs_tc)

            # forward pass to store the attn_outputs of the text contrast inputs
            if score_mode == "consensus":
                set_vhr_phase(self, "text")
            outputs_tc = self(**model_inputs_tc, return_dict=True)

            if score_mode == "consensus":
                if images_noisy is None:
                    raise ValueError("CC-VHR requires `vhr_images_noisy`")
                model_kwargs_nc = copy.deepcopy(model_kwargs)
                if "pixel_values" in model_kwargs_nc:
                    model_kwargs_nc["pixel_values"] = images_noisy
                else:
                    num_query_tokens = images_noisy.shape[1]
                    model_kwargs_nc["inputs_embeds"] = torch.cat(
                        [images_noisy, model_kwargs_nc["inputs_embeds"][:, num_query_tokens:]], dim=1
                    )
                model_kwargs_nc = self._get_initial_cache_position(input_ids, model_kwargs_nc)
                model_inputs_nc = self.prepare_inputs_for_generation(input_ids, **model_kwargs_nc)
                set_vhr_phase(self, "noisy")
                outputs_nc = self(**model_inputs_nc, return_dict=True)
                del outputs_nc
            if not use_visual_evidence_gate:
                del outputs_tc

            if use_vcd:
                model_kwargs_cd = copy.deepcopy(model_kwargs)
                if "pixel_values" in model_kwargs_cd:
                    model_kwargs_cd["pixel_values"] = images_cd
                else:
                    num_query_tokens = images_cd.shape[1]
                    model_kwargs_cd["inputs_embeds"] = torch.cat(
                        [images_cd, model_kwargs_cd["inputs_embeds"][:, num_query_tokens:]], dim=1
                    )
                model_kwargs_cd = self._get_initial_cache_position(input_ids, model_kwargs_cd)
            elif use_base_contrast:
                model_kwargs_cd = copy.deepcopy(model_kwargs)
                model_kwargs_cd = self._get_initial_cache_position(input_ids, model_kwargs_cd)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        if score_mode == "consensus":
            set_vhr_phase(self, "clean" if cur_len == prompt_length else "decode")
        elif use_base_contrast:
            set_vhr_phase(self, "clean" if cur_len == prompt_length else "decode")
        outputs = self(**model_inputs, return_dict=True)
        if use_vcrp:
            vcrp_current_stats = collect_vcrp_step_stats(self)
            vcrp_current_stats["step_index"] = cur_len - prompt_length

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits.clone()[:, -1, :].float()
        diagnostic_vhr_argmax = None
        diagnostic_base_argmax = None
        diagnostic_delta = None

        if use_aux_contrast:
            model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
            model_inputs_cd.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs_cd.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            set_vhr_phase(self, "reuse" if use_vcd else "vanilla")
            outputs_cd = self(**model_inputs_cd, return_dict=True)
            next_token_logits_cd = outputs_cd.logits.clone()[:, -1, :].float()

            next_token_logits_tc = None
            if use_visual_evidence_gate:
                if cur_len != prompt_length:
                    model_inputs_tc = self.prepare_inputs_for_generation(input_ids_tc, **model_kwargs_tc)
                    model_inputs_tc.update(
                        {"output_attentions": output_attentions} if output_attentions else {}
                    )
                    model_inputs_tc.update(
                        {"output_hidden_states": output_hidden_states} if output_hidden_states else {}
                    )
                    set_vhr_phase(self, "vanilla")
                    outputs_tc = self(**model_inputs_tc, return_dict=True)
                next_token_logits_tc = outputs_tc.logits.clone()[:, -1, :].float()

            if use_switch_diagnostics:
                diagnostic_vhr_argmax = next_token_logits.argmax(dim=-1, keepdim=True)
                diagnostic_base_argmax = next_token_logits_cd.argmax(dim=-1, keepdim=True)
                diagnostic_delta = next_token_logits - next_token_logits_cd

            contrast_alpha = generation_config.cd_alpha if use_vcd else generation_config.vhr_logit_alpha
            contrast_beta = generation_config.cd_beta if use_vcd else generation_config.vhr_logit_beta
            cutoff = math.log(contrast_beta) + next_token_logits.max(dim=-1, keepdim=True).values
            contrasted_logits = compute_contrasted_logits(
                next_token_logits,
                next_token_logits_cd,
                contrast_alpha,
                positive_delta=(
                    use_base_contrast and generation_config.vhr_positive_delta_contrast
                ),
            )
            contrasted_logits = contrasted_logits.masked_fill(next_token_logits < cutoff, -float("inf"))
            if use_disagreement_gate:
                contrasted_logits, disagreement = apply_disagreement_gate(
                    next_token_logits,
                    next_token_logits_cd,
                    contrasted_logits,
                )
                disagreement_total_steps += disagreement.numel()
                disagreement_contrast_steps += int(disagreement.sum().item())
            if use_visual_evidence_gate:
                contrasted_logits, proposed_switch, allowed_switch, support_margin = (
                    apply_visual_evidence_gate(
                        next_token_logits,
                        next_token_logits_cd,
                        next_token_logits_tc,
                        contrasted_logits,
                    )
                )
                blocked_switch = proposed_switch & (~allowed_switch)
                visual_total_steps += proposed_switch.numel()
                visual_proposed_switch_steps += int(proposed_switch.sum().item())
                visual_allowed_switch_steps += int(allowed_switch.sum().item())
                visual_blocked_switch_steps += int(blocked_switch.sum().item())
                visual_allowed_margin_sum += float(
                    (support_margin * allowed_switch.float()).sum().item()
                )
                visual_blocked_margin_sum += float(
                    (support_margin * blocked_switch.float()).sum().item()
                )
            next_token_scores = logits_processor(input_ids, contrasted_logits)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        
        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:  # argmax
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if use_vcrp:
            if next_tokens.numel() != 1:
                raise ValueError("VCRP diagnostics currently require batch size 1")
            vcrp_current_stats["selected_token_id"] = int(next_tokens[0].item())
            vcrp_token_stats.append(vcrp_current_stats)

        if use_switch_diagnostics:
            selected = next_tokens.unsqueeze(-1)
            branch_disagreement = diagnostic_vhr_argmax != diagnostic_base_argmax
            selected_switch = selected != diagnostic_vhr_argmax
            same_argmax_switch = (~branch_disagreement) & selected_switch
            disagreement_switch = branch_disagreement & selected_switch
            disagreement_no_switch = branch_disagreement & (~selected_switch)

            switch_total_steps += branch_disagreement.numel()
            switch_branch_disagreement_steps += int(branch_disagreement.sum().item())
            switch_selected_steps += int(selected_switch.sum().item())
            switch_same_argmax_steps += int(same_argmax_switch.sum().item())
            switch_disagreement_steps += int(disagreement_switch.sum().item())
            switch_disagreement_no_switch_steps += int(disagreement_no_switch.sum().item())

            if same_argmax_switch.any():
                chosen_delta = diagnostic_delta.gather(-1, selected)
                vhr_delta = diagnostic_delta.gather(-1, diagnostic_vhr_argmax)
                chosen_score = next_token_scores.gather(-1, selected)
                vhr_score = next_token_scores.gather(-1, diagnostic_vhr_argmax)
                mask = same_argmax_switch.float()
                switch_same_selected_delta_sum += float((chosen_delta * mask).sum().item())
                switch_same_vhr_delta_sum += float((vhr_delta * mask).sum().item())
                switch_same_margin_sum += float(((chosen_score - vhr_score) * mask).sum().item())

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if use_aux_contrast:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
        if use_visual_evidence_gate:
            input_ids_tc = torch.cat([input_ids_tc, next_tokens[:, None]], dim=-1)
            model_kwargs_tc = self._update_model_kwargs_for_generation(
                outputs_tc,
                model_kwargs_tc,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
        # stop when each sentence is finished
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        if use_aux_contrast:
            del outputs_cd
        if use_visual_evidence_gate:
            del outputs_tc

    if streamer is not None:
        streamer.end()

    if use_disagreement_gate:
        self._last_vhr_routing_stats = {
            "contrast_steps": disagreement_contrast_steps,
            "total_steps": disagreement_total_steps,
            "contrast_rate": disagreement_contrast_steps / max(1, disagreement_total_steps),
        }
    if use_switch_diagnostics:
        self._last_vhr_switch_stats = {
            "total_steps": switch_total_steps,
            "branch_disagreement_steps": switch_branch_disagreement_steps,
            "selected_switch_steps": switch_selected_steps,
            "same_argmax_switch_steps": switch_same_argmax_steps,
            "disagreement_switch_steps": switch_disagreement_steps,
            "disagreement_no_switch_steps": switch_disagreement_no_switch_steps,
            "branch_disagreement_rate": switch_branch_disagreement_steps / max(1, switch_total_steps),
            "selected_switch_rate": switch_selected_steps / max(1, switch_total_steps),
            "same_argmax_switch_rate": switch_same_argmax_steps / max(
                1, switch_total_steps - switch_branch_disagreement_steps
            ),
            "same_switch_selected_delta_mean": switch_same_selected_delta_sum
            / max(1, switch_same_argmax_steps),
            "same_switch_vhr_delta_mean": switch_same_vhr_delta_sum
            / max(1, switch_same_argmax_steps),
            "same_switch_score_margin_mean": switch_same_margin_sum
            / max(1, switch_same_argmax_steps),
        }
    if use_visual_evidence_gate:
        self._last_vhr_visual_evidence_stats = {
            "total_steps": visual_total_steps,
            "proposed_switch_steps": visual_proposed_switch_steps,
            "allowed_switch_steps": visual_allowed_switch_steps,
            "blocked_switch_steps": visual_blocked_switch_steps,
            "proposed_switch_rate": visual_proposed_switch_steps / max(1, visual_total_steps),
            "allowed_switch_rate": visual_allowed_switch_steps / max(1, visual_total_steps),
            "allow_fraction_of_proposals": visual_allowed_switch_steps
            / max(1, visual_proposed_switch_steps),
            "allowed_support_margin_mean": visual_allowed_margin_sum
            / max(1, visual_allowed_switch_steps),
            "blocked_support_margin_mean": visual_blocked_margin_sum
            / max(1, visual_blocked_switch_steps),
        }
    if use_vcrp:
        self._last_vcrp_stats = {
            "token_count": len(vcrp_token_stats),
            "tokens": vcrp_token_stats,
        }

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def _evolve_vcd_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    images_cd = generation_config.images_cd
    
    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    model_kwargs_cd = None

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if model_kwargs_cd is None:
            model_kwargs_cd = copy.deepcopy(model_kwargs)
            if 'pixel_values' in model_kwargs_cd:
                model_kwargs_cd['pixel_values'] = images_cd
            else:
                model_kwargs_cd['inputs_embeds'] = torch.cat([images_cd, model_kwargs_cd['inputs_embeds'][:, 32:]], 1)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        outputs = self(**model_inputs, return_dict=True)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits.clone()[:, -1, :].float()
        
        # VCD decoding
        model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs_cd.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs_cd.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # print("model_inputs_cd", model_inputs_cd)
        # print("self", self)
        outputs_cd = self(**model_inputs_cd, return_dict=True)
        
        next_token_logits_cd = outputs_cd.logits.clone()[:, -1, :].float()
        
        ## cd_comments: pre-process logits from contrastive inputs
        cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
        cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
        
        # version 1  set cutoff for Adaptive Plausibility Constraints
        # probs = nn.functional.softmax(next_token_logits, dim=-1)
        # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

        # version 2 set cutoff for Adaptive Plausibility Constraints
        cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
        
        diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
        cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

        ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
        cd_logits = logits_processor(input_ids, cd_logits)

        next_token_scores = cd_logits
        if do_sample:
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(cd_logits, dim=-1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        
         # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        model_kwargs_cd = self._update_model_kwargs_for_generation(
            outputs_cd, 
            model_kwargs_cd, 
            is_encoder_decoder=self.config.is_encoder_decoder
        )
        # stop when each sentence is finished
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs, outputs_cd

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def _code_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    instead_images = generation_config.code_instead_images
    image_start = generation_config.code_image_start
    image_end = generation_config.code_image_end
    bd_k = generation_config.code_k
   
    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    input_ids_cd = None

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if input_ids_cd is None:
            model_kwargs_cd = copy.deepcopy(model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        outputs = self(**model_inputs, return_dict=True)

        if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits.clone()[:, -1, :].float()
        
        # CODE decoding
        if input_ids_cd is None:
            attention_mask = model_kwargs_cd['attention_mask']
            model_kwargs_cd['pixel_values'] = None  # TODO: change to instead_images
            model_kwargs_cd['image_sizes'] = None  # TODO: change to instead_images
            if image_start > 0:
                input_ids_before, input_ids_after = input_ids[:, :image_start].clone(), input_ids[:, image_end:].clone()
                attention_mask_before, attention_mask_after = attention_mask[:, :image_start], attention_mask[:, image_end:]
                input_ids_cd = torch.cat((input_ids_before, instead_images, input_ids_after), dim=1)
                model_kwargs_cd['attention_mask'] = torch.cat((attention_mask_before, torch.ones_like(instead_images), attention_mask_after), dim=1)
            else:
                # instructBLIP generation
                assert input_ids.shape[1] == 0
                input_ids_cd = input_ids.clone()
                instead_images_embeds = self.model.embed_tokens(instead_images)
                model_kwargs_cd["inputs_embeds"] = torch.cat([instead_images_embeds, model_kwargs_cd["inputs_embeds"][:, 32:]], 1)
                model_kwargs_cd["attention_mask"] = torch.cat(
                    [torch.ones_like(instead_images).to(input_ids.device), model_kwargs_cd["attention_mask"][:, 32:]], 1)
            model_kwargs_cd = self._get_initial_cache_position(input_ids_cd, model_kwargs_cd)
        
        model_inputs_cd = self.prepare_inputs_for_generation(input_ids_cd, **model_kwargs_cd)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs_cd.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs_cd.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # print("model_inputs_cd", model_inputs_cd)
        # print("self", self)
        outputs_cd = self(**model_inputs_cd, return_dict=True)
        
        next_token_logits_cd = outputs_cd.logits.clone()[:, -1, :].float()
        
        ## cd_comments: pre-process logits from contrastive inputs
       
        p_v = nn.functional.softmax(next_token_logits, dim=-1)
        p_d = nn.functional.softmax(next_token_logits_cd, dim=-1)
        
        kl_d = 0.5 * ((torch.log2(torch.abs(p_v - p_d) ** bd_k + 1)) * (p_v + p_d)).sum(dim=-1).unsqueeze(-1)

        kld_alpha = 1 - kl_d 
        
        cutoff = kl_d * p_v.max(dim=-1, keepdim=True).values

        ##############################
        diffs = (1 + kld_alpha) * next_token_logits - kld_alpha * next_token_logits_cd
        cd_logits = diffs.masked_fill(p_v < cutoff, -float("inf"))

        ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
        cd_logits = logits_processor(input_ids, cd_logits)

        next_token_scores = cd_logits
        if do_sample:
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(cd_logits, dim=-1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        
         # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        model_kwargs_cd = self._update_model_kwargs_for_generation(
            outputs_cd, 
            model_kwargs_cd, 
            is_encoder_decoder=self.config.is_encoder_decoder
        )
        # stop when each sentence is finished
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def _opera_beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    sequential = generation_config.low_memory
    key_position = generation_config.opera_key_position
    scale_factor = generation_config.opera_scale_factor
    threshold = generation_config.opera_threshold
    num_attn_candidates = generation_config.opera_num_attn_candidates
    window_size = generation_config.opera_window_size
    penalty_weights = generation_config.opera_penalty_weights

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    # initialise the history variables
    history_states = []
    history_rollback_locs = None
    beam_next_tokens = None
    beam_idx = None
    rollback_pos = 0
    max_rollback_time = torch.zeros(window_size)
    history_length = window_size
    reject_token_pos_gather = [[] for _ in range(window_size)]
    model_kwargs_ori = copy.deepcopy(model_kwargs)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # Define current states
        current_state = {}
        current_state["input_ids"] = input_ids.clone()
        current_state["beam_scorer"] = copy.deepcopy(beam_scorer)
        current_state["beam_indices"] = copy.deepcopy(beam_indices) if beam_indices is not None else None
        current_state["cur_len"] = cur_len
        
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # if sequential is True, split the input to batches of batch_size and run sequentially
        if sequential:
            if any(
                model_name in self.__class__.__name__.lower()
                for model_name in [
                    "fsmt",
                    "reformer",
                    "ctrl",
                    "gpt_bigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            inputs_per_sub_batches = _split_model_inputs(
                model_inputs, split_size=batch_size, full_batch_size=batch_beam_size, config=self.config.get_text_config()
            )
            outputs_per_sub_batch = [
                self(**inputs_per_sub_batch, return_dict=True) for inputs_per_sub_batch in inputs_per_sub_batches
            ]

            outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())

        else:  # Unchanged original behavior
            try:
                outputs = self(**model_inputs, return_dict=True)
            except IndexError:
                print(input_ids)

        # print("outputs.attentions[-1]", outputs.attentions[-1].shape)
        # input()

        # Load the previous self-attention weights
        if beam_idx is None:
            attn_previous = outputs.attentions[-1].clone() # [batch_size * num_beams, num_head, q, kv]
        else:
            attn_previous = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1)
            attn_previous = torch.cat([attn_previous[beam_idx], outputs.attentions[-1].clone().max(1, keepdim=True).values.data], -2) # [batch_size * num_beams, num_head, q, kv]
        attn_previous = attn_previous.max(1, keepdim=True).values.data # [batch_size * num_beams, 1, q, kv]
        current_state["attn_previous"] = attn_previous.data.cpu()

        # print("attn_previous", attn_previous.shape)
        # input()

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # .float() is needed to retain precision for later logits manipulations
        next_token_logits = outputs.logits[:, -1, :].clone().float()

        # Select candidates
        if num_attn_candidates < 1:
            raise ValueError(
                f"Num of candidates must be larger than 0, but it is currently {num_attn_candidates}."
            )
        candidate_token_scores, candidate_tokens = torch.topk(
            next_token_logits, num_attn_candidates, dim=-1, largest=True, sorted=True
        ) # [batch_size * num_beams, num_attn_candidates]
        current_state["candidate_tokens"] = candidate_tokens.clone()

        current_state["beam_scores"] = beam_scores.clone()
        current_state["beam_next_tokens"] = beam_next_tokens.clone() if beam_next_tokens is not None else None
        current_state["beam_idx"] = beam_idx.clone() if beam_idx is not None else None

        # Walk through all candidates to get their self-attention weights
        attn_last = []
        for candidate_id in range(num_attn_candidates):
            # update temporary generated ids, model inputs, and length for next step
            input_ids_tmp = torch.cat([input_ids, candidate_tokens[:, candidate_id].unsqueeze(-1)], dim=-1)

            model_kwargs_tmp = copy.deepcopy(model_kwargs)  # deepcopy to ensure forward using model_kwargs_tmp won't affect content in model_kwargs
            model_kwargs_tmp = self._update_model_kwargs_for_generation(
                copy.deepcopy(outputs), model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # prepare model inputs
            model_inputs_tmp = self.prepare_inputs_for_generation(input_ids_tmp, **model_kwargs_tmp)

            # forward pass to get the self-attention maps of next token prediction
            outputs_tmp = self(
                **model_inputs_tmp,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            
            # attn_square = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1)
            # attn_square = torch.cat([attn_square, outputs_tmp.attentions[-1].clone()], -2) # [batch_size * num_beams, num_head, q+1, kv+1]
            # attn_last.append(attn_square.max(1, keepdim=True).values.data) # [batch_size * num_beams, 1, q+1, kv+1]
            attn_output = outputs_tmp.attentions[-1].clone()
            attn_output = attn_output.max(1, keepdim=True).values.data # [batch_size * num_beams, 1, 1, kv+1]
            attn_square = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1) # [batch_size * num_beams, 1, q, kv+1]
            attn_square = torch.cat([attn_square, attn_output], -2) # [batch_size * num_beams, 1, q+1, kv+1]
            attn_last.append(attn_square) # [batch_size * num_beams, 1, q+1, kv+1]

        del input_ids_tmp, model_kwargs_tmp, model_inputs_tmp, outputs_tmp

        # Gather the attentions of all candidates
        attn_last = torch.cat(attn_last, 1) # [batch_size * num_beams, num_attn_candidates, q+1, kv+1]
        attn_last = attn_last / attn_last.sum(-1, keepdim=True)

        # Catch the self-attention weights with the size of local window
        # [batch_size * num_beams, num_attn_candidates, window_size, window_size]
        attn_pos = key_position
        attn_local = attn_last[:, :, attn_pos["response_start"]:, attn_pos["response_start"]:]

        # Scale up the self-attention weights and calculate the scores
        attn_local = scale_factor * attn_local
        attn_local_scores = torch.zeros((
            attn_local.shape[0], attn_local.shape[1], attn_local.shape[-1]), dtype=torch.float16).to(candidate_token_scores.device)
        for j in range(attn_local.shape[-1]):
            local_score = 1e-7 * attn_local[..., j:, j].prod(-1).data
            attn_local_scores[..., j] = local_score.to(torch.float32) # [batch_size * num_beams, num_attn_candidates, window_size]

        # We use the attention scores to penalize the first 10 tokens
        cur_response_lens = attn_local.shape[-1]
        attn_i = attn_last[:, :, -1, attn_pos["image_start"]:attn_pos["image_end"]+1].sum(-1)
        attn_scores = attn_i # [batch_size * num_beams, num_attn_candidates]

        # We use the rollback scores to penalize the subsequent tokens
        rollback_scores, rollback_locs = attn_local_scores.max(-1) # [batch_size * num_beams, num_attn_candidates]
        rollback_loc = rollback_locs.mode().values.data # [batch_size * num_beams]
        rollback_loc = rollback_loc.mode().values.data # [1]

        penalty_scores = - attn_scores if cur_response_lens <= 10 else rollback_scores # [batch_size * num_beams, num_attn_candidates]

        # incorporate with the history locations of the maximum of penalty scores
        if history_rollback_locs is None:
            history_rollback_locs = [rollback_locs.mode().values.data[:, None]]
        else:
            history_rollback_locs.append(rollback_locs.mode().values.data[:, None])
        rollback_loc_gathers = torch.cat(history_rollback_locs, -1)# [batch_size * num_beams, window_size]

        candidate_token_scores -= penalty_weights * penalty_scores
        current_state["candidate_token_scores"] = candidate_token_scores.clone()

        # history check
        if len(history_states) >= history_length:
            history_states.pop(0)
        history_states.append(current_state)

        # check if we need rollback
        try:
            if all((rollback_loc_gather == rollback_loc).long().sum() > int(threshold) for _, rollback_loc_gather in enumerate(rollback_loc_gathers)):
                if rollback_loc < 10: # or rollback_loc + 1 < rollback_pos:
                    assert False
                # locate the rollback position
                rollback_pos = rollback_loc + 1
                if max_rollback_time[rollback_pos] >= num_attn_candidates:
                    # print(f"Already reach the maximum rollback times at position {rollback_pos}, so shift the rollback position to {rollback_pos-1}")
                    rollback_pos = rollback_pos - 1
                    if max_rollback_time[rollback_pos] >= num_attn_candidates:
                        assert False
                    else:
                        max_rollback_time[rollback_pos] += 1
                else:
                    max_rollback_time[rollback_pos] += 1
                if cur_response_lens - rollback_pos > history_length + 1:
                    rollback_pos = max(1, cur_response_lens - history_length - 1)
                # print(f"rollback from pos {cur_response_lens-1} to pos {rollback_pos} for the time {int(max_rollback_time[rollback_pos])}")

                # discard the rollbacked states in history
                for j in range(cur_response_lens-rollback_pos-2):
                    history_states.pop(-1)
                    history_rollback_locs.pop(-1)
                    reject_token_pos_gather[-(j+1)] = []

                # Revive all of variables in the state of the rollback position
                input_ids = history_states[-2]["input_ids"]
                beam_scorer = history_states[-2]["beam_scorer"]
                beam_indices = history_states[-2]["beam_indices"]
                cur_len = history_states[-2]["cur_len"]

                attn_previous = history_states[-2]["attn_previous"].to(input_ids.device)
                candidate_token_scores = history_states[-2]["candidate_token_scores"]
                candidate_tokens = history_states[-2]["candidate_tokens"]

                beam_scores = history_states[-2]["beam_scores"]
                beam_next_tokens = history_states[-1]["beam_next_tokens"]
                beam_idx = history_states[-1]["beam_idx"]

                # first inference to get model kwargs
                if "pixel_values" in model_kwargs_ori.keys():
                    model_kwargs = copy.deepcopy(model_kwargs_ori)
                    model_kwargs["attention_mask"] = torch.cat([
                        model_kwargs["attention_mask"], torch.ones((
                            input_ids.shape[0], input_ids[:,:-1].shape[1] - model_kwargs["attention_mask"].shape[1]
                        )).to(device=input_ids.device, dtype=model_kwargs["attention_mask"].dtype)], 1)
                    model_kwargs = self._get_initial_cache_position(input_ids[:, :-1], model_kwargs)
                    model_inputs_tmp = self.prepare_inputs_for_generation(input_ids[:,:-1], **model_kwargs)
                else:
                    answer_embeds = self.model.embed_tokens(input_ids[:,:-1])
                    model_kwargs = copy.deepcopy(model_kwargs_ori)
                    model_kwargs["inputs_embeds"] = torch.cat([model_kwargs["inputs_embeds"], answer_embeds], 1)
                    model_kwargs["attention_mask"] = torch.cat(
                        [model_kwargs["attention_mask"], torch.ones_like(input_ids[:,:-1]).to(input_ids.device)], 1)
                    model_kwargs = self._get_initial_cache_position(input_ids[:, :0], model_kwargs)
                    model_inputs_tmp = self.prepare_inputs_for_generation(input_ids[:,:0], **model_kwargs)  # empty input_ids

                outputs_tmp = self(
                    **model_inputs_tmp,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs_tmp, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # another inference to get outputs and logits
                model_inputs_tmp = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                outputs = self(
                    **model_inputs_tmp,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits = outputs.logits[:, -1, :].clone().float()
                del outputs_tmp, model_inputs_tmp

                # discard the last rollbacked state in history
                history_states.pop(-1)
                history_rollback_locs.pop(-1)
                reject_token_pos_gather[rollback_pos+1] = []

                # set penalty on the corresponding candidates
                next_token_logits -= 999. + next_token_logits.min(-1, keepdim=True).values.data
                next_token_logits = next_token_logits.view(batch_size, num_beams * vocab_size)
                beam_idx = beam_idx.view(batch_size, num_beams)
                beam_next_tokens = beam_next_tokens.view(batch_size, num_beams)
                reject_token_pos = beam_idx * vocab_size + beam_next_tokens
                if len(reject_token_pos_gather[rollback_pos]) > 0:
                    reject_token_pos = torch.cat([reject_token_pos_gather[rollback_pos], reject_token_pos], -1)
                reject_token_pos_gather[rollback_pos] = reject_token_pos
                next_token_logits = next_token_logits.scatter_(-1, reject_token_pos, -999.)
                next_token_logits = next_token_logits.view(batch_size * num_beams, vocab_size)
            else:
                assert False
        except AssertionError:
            next_token_logits.fill_(-999.)
            next_token_logits = next_token_logits.scatter_(-1, candidate_tokens, candidate_token_scores)

        del attn_last, attn_local, attn_local_scores
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)

        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
        # non eos token per beam.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        cur_response_lens = input_ids.shape[-1]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
        # (that way the memory peak does not include outputs.logits)
        del outputs

        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]


@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        model_kwargs["num_logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
    user_defined_cache = model_kwargs.get(cache_name)
    max_cache_length = generation_config.max_length
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
            raise ValueError("assisted generate is not supported with Static cache classes`")
        if self._is_stateful:
            # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
            # which is not possible with stateful models (they can't reset to a previous subset of generated text)
            raise ValueError(
                f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
            )

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result = self._assisted_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.VHR_GENERATION:
        replace_attn_vhr(self,
                         target_layers=generation_config.vhr_layers,
                         aug_ratio=generation_config.vhr_aug_ratio,
                         filter=generation_config.vhr_filter,
                         score_mode=generation_config.vhr_score_mode,
                         head_fraction=generation_config.vhr_head_fraction,
                         positive_only=generation_config.vhr_positive_only,
                         use_avhr=generation_config.vhr_use_avhr,
                         avhr_tau=generation_config.vhr_tau,
                         avhr_alpha=generation_config.vhr_alpha,
                         avhr_rho_max=generation_config.vhr_rho_max,
                         vcrp_capture=(
                             generation_config.vhr_vcrp_diagnostic
                             or generation_config.vhr_vcrp_intervention
                         ))
        replace_vcrp_decoder_layers(
            self,
            target_layers=generation_config.vhr_layers,
            enabled=(
                generation_config.vhr_vcrp_diagnostic
                or generation_config.vhr_vcrp_intervention
            ),
            intervention=generation_config.vhr_vcrp_intervention,
        )
        try:
            result = self._vhr_decoding(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        finally:
            replace_attn_vhr(self)
            replace_vcrp_decoder_layers(self)
    elif generation_mode == GenerationMode.DOLA_GENERATION:
        if self._is_stateful:
            # DoLa decoding was not designed for stateful models, and would require some changes
            raise ValueError(
                f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )
        result = self._dola_decoding(
            input_ids,
            dola_layers=generation_config.dola_layers,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.VCD_GENERATION:
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        result = self._evolve_vcd_decoding(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CODE_GENERATION:
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        result = self._code_decoding(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.OPERA_GENERATION:
        assert generation_config.output_attentions, "OPERA decoding requires output_attentions=True!"
        if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if prepared_stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        result = self._opera_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")
        if self._is_stateful:
            # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
            raise ValueError(
                f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._contrastive_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and hasattr(result.past_key_values, "to_legacy_cache")
        and result.past_key_values.to_legacy_cache is not None
    ):
        # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
        should_convert_cache = generation_config.return_legacy_cache
        is_user_defined_cache = user_defined_cache is not None
        is_default_cache_type = (
            type(result.past_key_values) == DynamicCache  # noqa E721
            or (
                isinstance(result.past_key_values, EncoderDecoderCache)
                and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
                and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
            )
        )
        if not is_user_defined_cache and is_default_cache_type:
            logger.warning_once(
                "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                "keep returning the legacy format, please set `return_legacy_cache=True`."
            )
            should_convert_cache = True
        if should_convert_cache:
            result.past_key_values = result.past_key_values.to_legacy_cache()
    return result


def generation_config_init(self, **kwargs):
    # Parameters that control the length of the output
    self.max_length = kwargs.pop("max_length", 20)
    self.max_new_tokens = kwargs.pop("max_new_tokens", None)
    self.min_length = kwargs.pop("min_length", 0)
    self.min_new_tokens = kwargs.pop("min_new_tokens", None)
    self.early_stopping = kwargs.pop("early_stopping", False)
    self.max_time = kwargs.pop("max_time", None)
    self.stop_strings = kwargs.pop("stop_strings", None)

    # Parameters that control the generation strategy used
    self.do_sample = kwargs.pop("do_sample", False)
    self.num_beams = kwargs.pop("num_beams", 1)
    self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
    self.penalty_alpha = kwargs.pop("penalty_alpha", None)
    self.use_cache = kwargs.pop("use_cache", True)

    # Parameters for manipulation of the model output logits
    self.temperature = kwargs.pop("temperature", 1.0)
    self.top_k = kwargs.pop("top_k", 50)
    self.top_p = kwargs.pop("top_p", 1.0)
    self.min_p = kwargs.pop("min_p", None)
    self.typical_p = kwargs.pop("typical_p", 1.0)
    self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
    self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
    self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
    self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
    self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
    self.length_penalty = kwargs.pop("length_penalty", 1.0)
    self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
    self.bad_words_ids = kwargs.pop("bad_words_ids", None)
    self.force_words_ids = kwargs.pop("force_words_ids", None)
    self.renormalize_logits = kwargs.pop("renormalize_logits", False)
    self.constraints = kwargs.pop("constraints", None)
    self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
    self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
    self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
    self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
    self.suppress_tokens = kwargs.pop("suppress_tokens", None)
    self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
    self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
    self.sequence_bias = kwargs.pop("sequence_bias", None)
    self.token_healing = kwargs.pop("token_healing", False)
    self.guidance_scale = kwargs.pop("guidance_scale", None)
    self.low_memory = kwargs.pop("low_memory", None)
    watermarking_config = kwargs.pop("watermarking_config", None)
    if watermarking_config is None:
        self.watermarking_config = None
    elif isinstance(watermarking_config, WatermarkingConfig):
        self.watermarking_config = watermarking_config
    else:
        self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

    # Parameters that define the output variables of `generate`
    self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
    self.output_attentions = kwargs.pop("output_attentions", False)
    self.output_hidden_states = kwargs.pop("output_hidden_states", False)
    self.output_scores = kwargs.pop("output_scores", False)
    self.output_logits = kwargs.pop("output_logits", None)
    self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

    # Special tokens that can be used at generation time
    self.pad_token_id = kwargs.pop("pad_token_id", None)
    self.bos_token_id = kwargs.pop("bos_token_id", None)
    self.eos_token_id = kwargs.pop("eos_token_id", None)

    # Generation parameters exclusive to encoder-decoder models
    self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
    self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

    # Assistant generation
    self.is_assistant = False
    self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
    self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "constant")
    self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", 0.4)

    # DoLa generation
    self.dola_layers = kwargs.pop("dola_layers", None)

    # VHR generation
    self.vhr_layers = kwargs.pop("vhr_layers", None)
    self.vhr_aug_ratio = kwargs.pop("vhr_aug_ratio", 1.)
    self.vhr_filter = kwargs.pop("vhr_filter", False)
    self.vhr_score_mode = kwargs.pop("vhr_score_mode", "magnitude")
    self.vhr_head_fraction = kwargs.pop("vhr_head_fraction", 0.5)
    self.vhr_positive_only = kwargs.pop("vhr_positive_only", False)
    self.vhr_contrast_mode = kwargs.pop("vhr_contrast_mode", "remove")
    self.vhr_directional_delay = kwargs.pop("vhr_directional_delay", 64)
    self.vhr_image_start = kwargs.pop("vhr_image_start", None)
    self.vhr_image_end = kwargs.pop("vhr_image_end", None)
    self.vhr_images_noisy = kwargs.pop("vhr_images_noisy", None)
    self.vhr_base_contrast = kwargs.pop("vhr_base_contrast", False)
    self.vhr_positive_delta_contrast = kwargs.pop("vhr_positive_delta_contrast", False)
    self.vhr_disagreement_gate = kwargs.pop("vhr_disagreement_gate", False)
    self.vhr_switch_diagnostics = kwargs.pop("vhr_switch_diagnostics", False)
    self.vhr_visual_evidence_gate = kwargs.pop("vhr_visual_evidence_gate", False)
    self.vhr_vcrp_diagnostic = kwargs.pop("vhr_vcrp_diagnostic", False)
    self.vhr_vcrp_intervention = kwargs.pop("vhr_vcrp_intervention", False)
    self.vhr_logit_alpha = kwargs.pop("vhr_logit_alpha", 0.5)
    self.vhr_logit_beta = kwargs.pop("vhr_logit_beta", 0.1)
    # A-VHR (Adaptive-ratio VHR) hyper-parameters
    self.vhr_use_avhr = kwargs.pop("vhr_use_avhr", False)
    self.vhr_tau = kwargs.pop("vhr_tau", 1.0)
    self.vhr_alpha = kwargs.pop("vhr_alpha", 0.5)
    self.vhr_rho_max = kwargs.pop("vhr_rho_max", 2.0)

    # VCD generation
    self.images_cd = kwargs.pop("images_cd", None)
    self.cd_alpha = kwargs.pop("cd_alpha", 1)
    self.cd_beta = kwargs.pop("cd_beta", 0.1)

    # Opera generation
    self.opera_key_position = kwargs.pop("opera_key_position", None)
    self.opera_scale_factor = kwargs.pop("opera_scale_factor", 50.0)
    self.opera_threshold = kwargs.pop("opera_threshold", 15)
    self.opera_num_attn_candidates = kwargs.pop("opera_num_attn_candidates", 5)
    self.opera_window_size = kwargs.pop("opera_window_size", 512)
    self.opera_penalty_weights = kwargs.pop("opera_penalty_weights", 1.0)

    # CODE generation
    self.code_k = kwargs.pop("coda_k", 0.3)
    self.code_instead_images = kwargs.pop("code_instead_images", None)
    self.code_image_start = kwargs.pop("code_image_start", 0)
    self.code_image_end = kwargs.pop("code_image_end", 0)

    # Cache implementation
    self.cache_implementation = kwargs.pop("cache_implementation", None)
    self.cache_config = kwargs.pop("cache_config", None)
    if self.cache_implementation is not None and self.cache_implementation in NEEDS_CACHE_CONFIG:
        cache_config_class = NEEDS_CACHE_CONFIG[self.cache_implementation]
        if self.cache_config is None:
            self.cache_config = cache_config_class()
        elif isinstance(self.cache_config, dict):
            self.cache_config = cache_config_class.from_dict(self.cache_config)
    self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)

    # Prompt lookup decoding
    self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
    self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)

    # Wild card
    self.generation_kwargs = kwargs.pop("generation_kwargs", {})

    # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
    # interface.
    self._from_model_config = kwargs.pop("_from_model_config", False)
    self._commit_hash = kwargs.pop("_commit_hash", None)
    self.transformers_version = kwargs.pop("transformers_version", __version__)

    # Additional attributes without default values
    if not self._from_model_config:
        # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
        # model's default configuration file
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    # Validate the values of the attributes
    self.validate(is_init=True)


def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
    """
    Returns the generation mode triggered by the [`GenerationConfig`] instance.

    Arg:
        assistant_model (`PreTrainedModel`, *optional*):
            The assistant model to be used for assisted generation. If set, the generation mode will be
            assisted generation.

    Returns:
        `GenerationMode`: The generation mode triggered by the instance.
    """
    # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
    # property and part of the `__repr__`
    if self.constraints is not None or self.force_words_ids is not None:
        generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
    elif self.num_beams == 1:
        if self.do_sample is False:
            if (
                self.top_k is not None
                and self.top_k > 1
                and self.penalty_alpha is not None
                and self.penalty_alpha > 0
            ):
                generation_mode = GenerationMode.CONTRASTIVE_SEARCH
            else:
                generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = GenerationMode.SAMPLE
    else:
        if self.num_beam_groups > 1:
            generation_mode = GenerationMode.GROUP_BEAM_SEARCH
        elif self.do_sample is True:
            generation_mode = GenerationMode.BEAM_SAMPLE
        else:
            generation_mode = GenerationMode.BEAM_SEARCH

    # Assisted generation may extend some generation modes
    if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.ASSISTED_GENERATION
        else:
            raise ValueError(
                "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                "is only supported with Greedy Search and Sample."
            )

    # DoLa generation may extend some generation modes
    if self.dola_layers is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.DOLA_GENERATION
        else:
            raise ValueError(
                "You've set `dola_layers`, which triggers DoLa generate. Currently, DoLa generate "
                "is only supported with Greedy Search and Sample."
            )
    
    # VHR generation
    if self.vhr_layers is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.VHR_GENERATION
        else:
            raise ValueError(
                "You've set `vhr_layers`, which triggers VHR generate. Currently, VHR generate "
                "is only supported with Sampling and Search."
            )

    # VCD generation
    if self.images_cd is not None:
        if generation_mode == GenerationMode.VHR_GENERATION:
            # Symmetric VHR-VCD is executed by the VHR loop so that both
            # clean and noisy branches reuse exactly the same selected heads.
            pass
        elif generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.VCD_GENERATION
        
        else:
            raise ValueError(
                "You've set `images_cd`, which triggers VCD generate. Currently, VCD generate "
                "is only supported with Sampling."
            )
    
    # OPERA generation
    if self.opera_key_position is not None:
        if generation_mode != "beam_search":
            raise ValueError(
                "You've set `opera_key_position`, which triggers Opera generate. Currently, Opera generate "
                "is only supported with Beam Search."
            )
        generation_mode = GenerationMode.OPERA_GENERATION

    if self.code_instead_images is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.CODE_GENERATION
        else:
            raise ValueError(
                "You've set `code_instead_images`, which triggers CODE generate. Currently, CODE generate "
                "is only supported with Sampling."
            )

    return generation_mode


def replace_generation():
    transformers.generation.utils.GenerationMixin.generate = generate
    transformers.generation.utils.GenerationMixin._vhr_decoding = _vhr_decoding
    transformers.generation.utils.GenerationMixin._evolve_vcd_decoding = _evolve_vcd_decoding
    transformers.generation.utils.GenerationMixin._code_decoding = _code_decoding
    transformers.generation.utils.GenerationMixin._opera_beam_search = _opera_beam_search
    transformers.generation.configuration_utils.GenerationConfig.get_generation_mode = get_generation_mode
    transformers.generation.configuration_utils.GenerationConfig.__init__ = generation_config_init
