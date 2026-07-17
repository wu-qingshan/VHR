#!/usr/bin/env python3
"""Unit checks for Qwen vision-span handling and frozen layer policy."""

from types import SimpleNamespace

import torch

from qwen_scvhr import qwen_rr_mode, qwen_target_layers, remove_single_vision_span


def main():
    config = SimpleNamespace(
        vision_start_token_id=10,
        vision_end_token_id=11,
        image_token_id=12,
    )
    input_ids = torch.tensor([[1, 2, 10, 12, 12, 11, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    text_ids, text_mask, span = remove_single_vision_span(input_ids, attention_mask, config)
    assert span == (2, 6)
    assert text_ids.tolist() == [[1, 2, 3, 4]]
    assert text_mask.tolist() == [[1, 1, 1, 1]]

    model = SimpleNamespace(model=SimpleNamespace(layers=[object() for _ in range(28)]))
    assert qwen_target_layers(model) == [1] + list(range(14, 28))
    assert qwen_rr_mode("Is there a dog in the image?") == "vhr"
    assert qwen_rr_mode("Please describe this image in detail.") == "scvhr"
    print("Qwen SC-VHR helper checks passed")


if __name__ == "__main__":
    main()
