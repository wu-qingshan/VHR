#!/usr/bin/env python3
"""Unit checks for Qwen vision-span handling and frozen layer policy."""

from types import SimpleNamespace

import torch

from qwen_scvhr import (
    cycle_if_appended,
    highest_safe_token,
    qwen_rr_mode,
    qwen_target_layers,
    remove_single_vision_span,
)


class SpaceProcessor:
    def decode(self, token_ids, **kwargs):
        return " ".join(str(value) for value in token_ids)


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

    processor = SpaceProcessor()
    risk = cycle_if_appended(processor, [1, 2, 1, 2, 1], 2)
    assert risk == {"period": 2, "block": ["1", "2"], "text_token_count": 6}
    assert cycle_if_appended(processor, [1, 2, 1, 2, 1], 3) is None
    scores = torch.tensor([[0.0, 0.1, 0.9, 0.8]])
    safe, masked = highest_safe_token(scores, processor, [1, 2, 1, 2, 1])
    assert int(safe.item()) == 3
    assert masked == 1
    print("Qwen SC-VHR helper checks passed")


if __name__ == "__main__":
    main()
