"""Low-rank MLP init and forward smoke tests."""
from __future__ import annotations

import torch

from attention_compression.joint_qkv import LowRankBranch, init_branch_from_pca
from attention_compression.mlp_lowrank import (
    LowRankLinear,
    LowRankSwiGLU,
    StagedHybridSwiGLU,
    choose_rank,
    stage_train_targets,
    staged_hybrid_to_lowrank,
)


def test_lowrank_linear_matches_branch() -> None:
    mod = LowRankLinear(32, 16, rank=4)
    x = torch.randn(2, 3, 32)
    y = mod(x)
    assert y.shape == (2, 3, 16)


def test_swiglu_forward_shape() -> None:
    mlp = LowRankSwiGLU(
        hidden_size=32,
        intermediate_size=64,
        rank_gate=4,
        rank_up=4,
        rank_down=4,
        act_fn=torch.nn.SiLU(),
    )
    x = torch.randn(2, 32)
    y = mlp(x)
    assert y.shape == (2, 32)


def test_init_branch_from_pca_roundtrip() -> None:
    in_d, out_d, rank = 32, 16, 4
    w = torch.randn(in_d, out_d)
    branch = LowRankBranch(in_d, out_d, rank)
    x = torch.randn(100, in_d)
    y = x @ w
    mean = y.mean(0)
    cov = (y - mean).T @ (y - mean) / 99
    vals, vecs = torch.linalg.eigh(cov)
    basis = vecs[:, torch.argsort(vals, descending=True)]
    init_branch_from_pca(branch, projection=w, mean=mean, basis=basis)
    y_hat = branch(x)
    cos = torch.nn.functional.cosine_similarity(y_hat, y, dim=-1).mean()
    assert float(cos) > 0.9


def test_isolated_down_uses_hybrid_hidden() -> None:
    class _Teacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act_fn = torch.nn.SiLU()
            self.gate_proj = torch.nn.Linear(32, 64, bias=False)
            self.up_proj = torch.nn.Linear(32, 64, bias=False)
            self.down_proj = torch.nn.Linear(64, 32, bias=False)

    teacher = _Teacher()
    hybrid = StagedHybridSwiGLU(teacher, ranks={"gate": 4, "up": 4, "down": 4}, compressed=("gate", "up", "down"))
    x = torch.randn(2, 32)
    with torch.no_grad():
        hybrid.gate_proj.branch.down.fill_(0.01)
    pred, tgt = stage_train_targets(hybrid, teacher, x, stage="down", loss_target="isolated")
    h = hybrid.mlp_hidden(x)
    assert pred.shape == tgt.shape == (2, 32)
    assert torch.allclose(tgt, teacher.down_proj(h))


def test_staged_hybrid_merges_to_lowrank() -> None:
    class _Teacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act_fn = torch.nn.SiLU()
            self.gate_proj = torch.nn.Linear(32, 64)
            self.up_proj = torch.nn.Linear(32, 64)
            self.down_proj = torch.nn.Linear(64, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    teacher = _Teacher()
    ranks = {"gate": 4, "up": 4, "down": 4}
    hybrid = StagedHybridSwiGLU(teacher, ranks=ranks, compressed=("gate", "up", "down"))
    x = torch.randn(3, 32)
    assert hybrid(x).shape == (3, 32)
    student = staged_hybrid_to_lowrank(hybrid)
    assert student(x).shape == (3, 32)


def test_choose_rank_respects_cap() -> None:
    cumulative = torch.linspace(0, 1, 100)
    pca = {"cumulative": cumulative}
    assert choose_rank(pca, 0.95, cap=10) == 10
