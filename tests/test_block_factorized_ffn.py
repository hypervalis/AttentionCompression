"""Block factorized FFN shape and merge-mode smoke tests."""
from __future__ import annotations

import torch

from attention_compression.block_factorized_ffn import BlockFactorizedFFN, equal_pca_blocks


def _random_ortho(dim: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return q


def test_additive_and_concat_forward_shapes() -> None:
    dim = 64
    num_blocks = 4
    block_dims = equal_pca_blocks(dim, num_blocks)
    basis = _random_ortho(dim)
    mean = torch.zeros(dim)
    x = torch.randn(3, 7, dim)

    for mode in ("additive", "concat"):
        mod = BlockFactorizedFFN(
            dim=dim,
            block_dims=block_dims,
            basis=basis,
            mean=mean,
            merge_mode=mode,
            hidden_dims=[32] * num_blocks,
            teacher_intermediate=128,
        )
        y = mod(x)
        assert y.shape == x.shape


def test_additive_is_sum_of_block_contributions() -> None:
    dim = 32
    block_dims = [16, 16]
    basis = _random_ortho(dim)
    mean = torch.randn(dim)
    mod = BlockFactorizedFFN(
        dim=dim,
        block_dims=block_dims,
        basis=basis,
        mean=mean,
        merge_mode="additive",
        hidden_dims=[24, 24],
    )
    x = torch.randn(2, dim)
    xc = x - mean
    manual = torch.zeros_like(x)
    for (lo, hi), mlp in zip(mod.block_slices(), mod.blocks, strict=True):
        vj = basis[:, lo:hi]
        manual = manual + mlp(xc @ vj) @ vj.T
    torch.testing.assert_close(mod(x), manual, rtol=1e-4, atol=1e-4)
