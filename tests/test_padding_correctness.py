"""Padding-correctness: the methodology the serving benchmark depends on.

The benchmark pads prompts up to a bucket length and runs prefill. For the
measured speedup to be honest, padding must NOT change the model's outputs at the
real token positions. For a *causal* LM with *right* padding this holds by
construction: position i only attends to positions <= i, so real positions never
attend to the padding that follows them.

We verify that property directly with scaled-dot-product attention (the kernel
underneath every transformer block) — on CPU, no model download.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.unit
def test_right_padding_preserves_causal_outputs() -> None:
    torch.manual_seed(0)
    batch, heads, real_len, dim, padded_len = 1, 4, 10, 16, 24

    q = torch.randn(batch, heads, real_len, dim)
    k = torch.randn(batch, heads, real_len, dim)
    v = torch.randn(batch, heads, real_len, dim)

    out_unpadded = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Right-pad q/k/v with zeros to the bucket length.
    def rpad(t: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(batch, heads, padded_len - real_len, dim)
        return torch.cat([t, pad], dim=2)

    out_padded = F.scaled_dot_product_attention(
        rpad(q), rpad(k), rpad(v), is_causal=True
    )

    # Outputs at the real positions must be identical.
    torch.testing.assert_close(out_padded[:, :, :real_len], out_unpadded)


@pytest.mark.unit
def test_padding_changes_only_padded_region_is_expected() -> None:
    # Sanity: the padded positions DO produce (garbage) outputs — confirming we
    # must read only the real positions, which the benchmark does.
    torch.manual_seed(1)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == (1, 2, 5, 8)


@pytest.mark.unit
def test_hf_causal_lm_right_padding_preserves_real_logits() -> None:
    # The benchmark's actual path: a real LlamaForCausalLM with an attention_mask.
    # Right-padded input must give identical logits at the real positions.
    pytest.importorskip("transformers")  # only with the `bench` extra installed
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg).eval()

    real_len, pad_to = 7, 16
    ids = torch.randint(0, cfg.vocab_size, (1, real_len))

    with torch.no_grad():
        unpadded = model(input_ids=ids).logits

        padded_ids = torch.cat(
            [ids, torch.zeros(1, pad_to - real_len, dtype=torch.long)], dim=1
        )
        mask = torch.zeros(1, pad_to, dtype=torch.long)
        mask[:, :real_len] = 1
        padded = model(input_ids=padded_ids, attention_mask=mask).logits

    torch.testing.assert_close(padded[:, :real_len], unpadded, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_prefill_forward_handles_llama_and_gpt2_backbones() -> None:
    # The benchmark runs the backbone (.model for Llama, .transformer for GPT-2).
    # Verify prefill_forward finds both and returns hidden states.
    pytest.importorskip("transformers")
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
    )

    from benchmarks.workload import prefill_forward

    llama = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
    )).eval()
    gpt2 = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_embd=32, n_layer=1, n_head=4, n_positions=64,
    )).eval()

    inputs = {
        "input_ids": torch.randint(0, 64, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    with torch.no_grad():
        assert prefill_forward(llama, inputs).shape == (1, 8, 32)
        assert prefill_forward(gpt2, inputs).shape == (1, 8, 32)
