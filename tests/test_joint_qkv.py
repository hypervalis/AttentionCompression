from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from attention_compression.joint_qkv import JointQKVBranches, LowRankBranch, init_branch_from_pca


def test_joint_qkv_branch_shapes() -> None:
    model = JointQKVBranches(input_dim=8, head_dim=4, q_rank=2, k_rank=3, v_rank=4)
    x = torch.randn(5, 7, 8)

    q, k, v = model(x)

    assert q.shape == (5, 7, 4)
    assert k.shape == (5, 7, 4)
    assert v.shape == (5, 7, 4)


def test_pca_initialization_matches_projection_for_full_rank() -> None:
    branch = LowRankBranch(input_dim=3, output_dim=3, rank=3)
    projection = torch.randn(3, 3)
    basis = torch.eye(3)
    mean = torch.randn(3)
    x = torch.randn(6, 3)

    init_branch_from_pca(branch, projection=projection, mean=mean, basis=basis)

    assert torch.allclose(branch(x), x @ projection, atol=1e-5)
