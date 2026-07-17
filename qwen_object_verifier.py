"""Qwen2.5-VL first-token object-existence verifier."""

import time

import torch


YES_TOKEN_ID = 9454
NO_TOKEN_ID = 2753
PROMPT_TEMPLATE = "Is there a visible {object_name} in this image? Answer only Yes or No."


def prepare_verifier_inputs(processor, image, object_name):
    question = PROMPT_TEMPLATE.format(object_name=object_name)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return question, inputs


@torch.inference_mode()
def score_object(model, processor, image, object_name, top_k=5):
    question, inputs = prepare_verifier_inputs(processor, image, object_name)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    model.rope_deltas = None
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    outputs = model(**inputs, use_cache=False, return_dict=True)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - start
    logits = outputs.logits[0, -1, :].float()
    yes_logit = float(logits[YES_TOKEN_ID].item())
    no_logit = float(logits[NO_TOKEN_ID].item())
    score = float(torch.sigmoid(logits[YES_TOKEN_ID] - logits[NO_TOKEN_ID]).item())
    top = logits.topk(top_k)
    top_candidates = [
        {
            "token_id": int(token_id),
            "token": processor.tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(top.values.tolist(), top.indices.tolist())
    ]
    if not all(torch.isfinite(torch.tensor(value)) for value in (yes_logit, no_logit, score)):
        raise FloatingPointError("non-finite object verifier score")
    model.rope_deltas = None
    return {
        "question": question,
        "yes_token_id": YES_TOKEN_ID,
        "no_token_id": NO_TOKEN_ID,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "score": score,
        "prediction": int(score >= 0.5),
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 1024**3,
        "image_grid_thw": inputs["image_grid_thw"][0].detach().cpu().tolist(),
        "top_candidates": top_candidates,
    }
