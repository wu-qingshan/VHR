from typing import Optional, Tuple, Union, List

import torch

from transformers.models.instructblip.modeling_instructblip import *
from transformers.models.instructblip.modeling_instructblip import (
    InstructBlipForConditionalGenerationModelOutput,
)
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration


def get_model(model_path, device, attn_implementation="eager"):
    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if "instructblip" in model_path:
        model_kwargs["attn_implementation"] = "eager"
        processor = InstructBlipProcessor.from_pretrained(model_path, use_fast=False)
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        # Note: Manually change the model.generation_config.eos_token_id to the same as tokenizer.eos_token_id
        replace_instructblip_forward()  # support non-image inference
        template = "{question}"
        ans_start = None
        num_img_tokens = model.config.num_query_tokens
        image_token_id = None
    elif "llava-v1.6" in model_path or "llava-next" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "{question}"},
                    {"type": "image"},
                ],
            },
        ]
        template = processor.apply_chat_template(conversation, add_generation_prompt=True)
        ans_start = "ASSISTANT:" if "llava-v1.6" in model_path else "assistant"
        num_img_tokens = None   # TODO: get the num_img_tokens and image_start for the template
        image_token_id = model.config.image_token_index
    else:
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        template = "USER: <image>\n{question} ASSISTANT:"
        ans_start = "ASSISTANT:"
        num_img_tokens = 576  
        image_token_id = model.config.image_token_index
    model.to(device)
    return processor, model, template, ans_start, num_img_tokens, image_token_id


def get_layers(model):
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    else:
        return model.model.layers


def instructBLIP_forward(
    self,
    pixel_values: torch.FloatTensor = None,
    qformer_input_ids: torch.FloatTensor = None,
    qformer_attention_mask: Optional[torch.LongTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    labels: Optional[torch.LongTensor] = None,
    return_dict: Optional[bool] = None,
    interpolate_pos_encoding: bool = False,
) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
        1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
        config.vocab_size]`

    Returns:

    Examples:

    ```python
    >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    >>> import torch
    >>> from PIL import Image
    >>> import requests

    >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> model.to(device)  # doctest: +IGNORE_RESULT

    >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    >>> prompt = "What is unusual about this image?"
    >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=False,
    ...     num_beams=5,
    ...     max_length=256,
    ...     min_length=1,
    ...     top_p=0.9,
    ...     repetition_penalty=1.5,
    ...     length_penalty=1.0,
    ...     temperature=1,
    ... )
    >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    >>> print(generated_text)
    The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is not None:
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
    else:
        vision_outputs = None
        query_outputs = None

    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if pixel_values is not None:
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

    if self.config.use_decoder_only_language_model:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
    else:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

    if not return_dict:
        output = (logits, vision_outputs, query_outputs, outputs)
        return ((loss,) + output) if loss is not None else output

    return InstructBlipForConditionalGenerationModelOutput(
        loss=loss,
        logits=logits,
        vision_outputs=vision_outputs,
        qformer_outputs=query_outputs,
        language_model_outputs=outputs,
    )


def replace_instructblip_forward():
    InstructBlipForConditionalGeneration.forward = instructBLIP_forward


def add_diffusion_noise(image_tensor, noise_step, generator=None):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn(x_0.shape, dtype=x_0.dtype, device=x_0.device, generator=generator)
        alphas_t = alphas_bar_sqrt[t].to(device=x_0.device, dtype=x_0.dtype)
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t].to(device=x_0.device, dtype=x_0.dtype)
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd
