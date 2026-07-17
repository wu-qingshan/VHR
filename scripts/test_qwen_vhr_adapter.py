#!/usr/bin/env python3
"""CPU unit checks for the Qwen2.5-VL eager attention adapter."""

from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention

from qwen_vhr import (
    install_qwen_vhr,
    qwen_vhr_state_report,
    reset_qwen_vhr_state,
    set_qwen_vhr_phase,
    uninstall_qwen_vhr,
)


def causal_mask(length):
    mask = torch.full((length, length), torch.finfo(torch.float32).min)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, length, length)


def position_embeddings(length, head_dim):
    return (
        torch.ones(3, 1, length, head_dim),
        torch.zeros(3, 1, length, head_dim),
    )


def main():
    torch.manual_seed(20260717)
    config = Qwen2_5_VLConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
    )
    attention = Qwen2_5_VLAttention(config, layer_idx=0).eval()
    layer = SimpleNamespace(self_attn=attention)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]), rope_deltas=None)

    hidden = torch.randn(1, 5, config.hidden_size)
    kwargs = {
        "attention_mask": causal_mask(5),
        "position_embeddings": position_embeddings(5, attention.head_dim),
    }
    expected = attention(hidden, **kwargs)[0]
    install_qwen_vhr(model, [0], aug_ratio=2.0, use_filter=True)
    actual_off = attention(hidden, **kwargs)[0]
    torch.testing.assert_close(actual_off, expected, atol=0, rtol=0)

    set_qwen_vhr_phase(model, "text")
    text_hidden = hidden[:, :-1, :]
    attention(
        text_hidden,
        attention_mask=causal_mask(4),
        position_embeddings=position_embeddings(4, attention.head_dim),
    )
    set_qwen_vhr_phase(model, "clean")
    clean = attention(hidden, **kwargs)[0]
    if torch.equal(clean, expected):
        raise AssertionError("clean VHR output unexpectedly equals vanilla output")

    report = qwen_vhr_state_report(model)
    assert report[0]["prefill_counts"] == {"text": 1, "clean": 1}
    assert 1 <= len(report[0]["selected_heads"]) <= config.num_attention_heads
    assert report[0]["finite_head_vary"] is True

    set_qwen_vhr_phase(model, "decode")
    decode_hidden = torch.randn(1, 1, config.hidden_size)
    decode_output = attention(
        decode_hidden,
        attention_mask=torch.zeros(1, 1, 1, 1),
        position_embeddings=position_embeddings(1, attention.head_dim),
    )[0]
    assert torch.isfinite(decode_output).all()

    set_qwen_vhr_phase(model, "vanilla")
    vanilla_after = attention(hidden, **kwargs)[0]
    torch.testing.assert_close(vanilla_after, expected, atol=0, rtol=0)

    reset_qwen_vhr_state(model)
    reset_report = qwen_vhr_state_report(model)[0]
    assert reset_report["phase"] == "off"
    assert reset_report["selected_heads"] is None
    assert reset_report["prefill_counts"] == {"text": 0, "clean": 0}

    uninstall_qwen_vhr(model)
    restored = attention(hidden, **kwargs)[0]
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    print("Qwen VHR adapter unit checks passed")


if __name__ == "__main__":
    main()
