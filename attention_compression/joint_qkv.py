from __future__ import annotations

import torch


class LowRankBranch(torch.nn.Module):
    """Low-rank local branch: `x @ down @ up + bias`."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.down = torch.nn.Parameter(torch.empty(input_dim, rank))
        self.up = torch.nn.Parameter(torch.empty(rank, output_dim))
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        torch.nn.init.normal_(self.down, std=0.01)
        torch.nn.init.normal_(self.up, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.down @ self.up + self.bias


class JointQKVBranches(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, q_rank: int, k_rank: int, v_rank: int) -> None:
        super().__init__()
        self.q = LowRankBranch(input_dim, head_dim, q_rank)
        self.k = LowRankBranch(input_dim, head_dim, k_rank)
        self.v = LowRankBranch(input_dim, head_dim, v_rank)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x), self.v(x)


class JointQKBranches(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, q_rank: int, k_rank: int) -> None:
        super().__init__()
        self.q = LowRankBranch(input_dim, head_dim, q_rank)
        self.k = LowRankBranch(input_dim, head_dim, k_rank)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x)


def init_branch_from_pca(
    branch: LowRankBranch,
    *,
    projection: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> None:
    """Initialize branch to PCA reconstruction of `x @ projection`.

    `projection` is `[input_dim, output_dim]`; `basis` is `[output_dim, rank]`.
    """
    rank = branch.down.shape[1]
    u = basis[:, :rank].to(device=projection.device, dtype=projection.dtype)
    with torch.no_grad():
        branch.down.copy_(projection @ u)
        branch.up.copy_(u.T)
        branch.bias.copy_(mean - mean @ u @ u.T)
