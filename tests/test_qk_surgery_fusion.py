"""Fused vs naive MultiHeadQKLowRankProjection must match numerically."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from attention_compression.qk_surgery import (
    MultiHeadQKLowRankProjection,
    materialize_dense_linear_from_branch_states,
)


def _random_branch_states(*, input_dim: int, head_dim: int, rank: int, num_heads: int, dtype: torch.dtype):
    states = []
    for _ in range(num_heads):
        states.append(
            {
                "q.down": torch.randn(input_dim, rank, dtype=dtype),
                "q.up": torch.randn(rank, head_dim, dtype=dtype),
                "q.bias": torch.randn(head_dim, dtype=dtype),
            }
        )
    return states


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_matches_naive(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim, head_dim, rank, num_heads = 32, 8, 4, 3
    states = _random_branch_states(
        input_dim=input_dim, head_dim=head_dim, rank=rank, num_heads=num_heads, dtype=dtype
    )
    mod = MultiHeadQKLowRankProjection(
        branch_states=states,
        branch_name="q",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    ).to(device)
    mod.eval()
    bsz, seq = 2, 5
    x = torch.randn(bsz, seq, input_dim, device=device, dtype=dtype)

    mod._force_naive_forward = True
    y_naive = mod(x)
    mod._force_naive_forward = False
    y_fused = mod(x)

    assert y_naive.shape == y_fused.shape == (bsz, seq, num_heads * head_dim)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(y_naive, y_fused, rtol=tol, atol=tol)


def test_mixed_rank_falls_back_to_naive() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    states = [
        {
            "q.down": torch.randn(16, 3, dtype=dtype),
            "q.up": torch.randn(3, 8, dtype=dtype),
            "q.bias": torch.randn(8, dtype=dtype),
        },
        {
            "q.down": torch.randn(16, 5, dtype=dtype),
            "q.up": torch.randn(5, 8, dtype=dtype),
            "q.bias": torch.randn(8, dtype=dtype),
        },
    ]
    mod = MultiHeadQKLowRankProjection(
        branch_states=states,
        branch_name="q",
        input_dim=16,
        head_dim=8,
        device=device,
        dtype=dtype,
    )
    assert not mod._uniform_rank()
    x = torch.randn(1, 2, 16, dtype=dtype)
    mod._force_naive_forward = False
    y_fused_path = mod(x)
    mod._force_naive_forward = True
    y_naive = mod(x)
    assert torch.allclose(y_fused_path, y_naive)


def test_materialized_linear_matches_einsum() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    input_dim, head_dim, rank, num_heads = 64, 16, 8, 4
    states = _random_branch_states(
        input_dim=input_dim, head_dim=head_dim, rank=rank, num_heads=num_heads, dtype=dtype
    )
    for i, st in enumerate(states):
        for k, v in list(st.items()):
            states[i][k] = v.to(device=device)

    fused_mod = MultiHeadQKLowRankProjection(
        branch_states=states,
        branch_name="q",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    ).to(device)
    fused_mod.eval()
    lin = materialize_dense_linear_from_branch_states(
        states,
        branch_name="q",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    ).to(device)

    x = torch.randn(3, 7, input_dim, dtype=dtype, device=device)
    assert torch.allclose(fused_mod(x), lin(x), rtol=1e-4, atol=1e-4)
