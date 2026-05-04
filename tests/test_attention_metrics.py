from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from attention_compression.attention_metrics import attention_kl, attention_logits, causal_attention, causal_logit_relative_mse, topk_overlap


def test_causal_attention_shapes_and_mask() -> None:
    q = torch.randn(2, 4, 3)
    k = torch.randn(2, 4, 3)
    v = torch.randn(2, 4, 5)

    logits, probs, context = causal_attention(q, k, v)

    assert logits.shape == (2, 4, 4)
    assert probs.shape == (2, 4, 4)
    assert context.shape == (2, 4, 5)
    assert torch.allclose(probs[:, 0, 1:], torch.zeros_like(probs[:, 0, 1:]))


def test_attention_kl_and_topk_overlap_identical_attention() -> None:
    probs = torch.softmax(torch.randn(2, 5, 5), dim=-1)

    assert attention_kl(probs, probs) < 1e-6
    assert topk_overlap(probs, probs, 3) == 1.0


def test_causal_logit_relative_mse_ignores_future_positions() -> None:
    q = torch.randn(1, 4, 3)
    k = torch.randn(1, 4, 3)
    logits = attention_logits(q, k)
    changed_future = logits.clone()
    changed_future[:, 0, 3] += 1000

    assert causal_logit_relative_mse(changed_future, logits) == 0.0
