"""FFN factorization: orthogonal PCA subspaces + one small MLP per block."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_compression.model_hub import get_transformer_layers

MERGE_MODES = ("additive", "concat")


def init_subspace_mlp_near_zero(mod: "SubspaceMLP") -> None:
    """Start each block MLP near zero so the bank begins as a small perturbation."""
    for layer in (mod.down, mod.fc, mod.up):
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.02)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


class SubspaceMLP(torch.nn.Module):
    """GELU MLP on a single subspace (``dim -> hidden -> dim``)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.down = torch.nn.Linear(dim, hidden)
        self.fc = torch.nn.Linear(hidden, hidden)
        self.up = torch.nn.Linear(hidden, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.down(z))
        h = torch.nn.functional.gelu(self.fc(h))
        return self.up(h)


class BlockFactorizedFFN(torch.nn.Module):
    """Split ``ffn_input`` along PCA blocks; per-block MLP; additive or concat merge."""

    def __init__(
        self,
        *,
        dim: int,
        block_dims: list[int],
        basis: torch.Tensor,
        mean: torch.Tensor,
        merge_mode: str,
        hidden_dims: list[int] | None = None,
        teacher_intermediate: int = 4096,
    ) -> None:
        super().__init__()
        if sum(block_dims) != dim:
            raise ValueError(f"block_dims {block_dims} must sum to dim={dim}")
        mode = merge_mode.strip().lower()
        if mode not in MERGE_MODES:
            raise ValueError(f"merge_mode must be one of {MERGE_MODES}")
        self.dim = dim
        self.block_dims = list(block_dims)
        self.num_blocks = len(block_dims)
        self.merge_mode = mode
        self.register_buffer("mean", mean.reshape(dim).float())
        self.register_buffer("basis", basis.float())
        if self.basis.shape != (dim, dim):
            raise ValueError(f"basis must be ({dim}, {dim}), got {tuple(self.basis.shape)}")

        if hidden_dims is None:
            hidden_dims = [
                max(int(round(teacher_intermediate * d / dim)), d * 2) for d in block_dims
            ]
        if len(hidden_dims) != len(block_dims):
            raise ValueError("hidden_dims length must match block_dims")

        self.blocks = torch.nn.ModuleList(
            [SubspaceMLP(d, h) for d, h in zip(block_dims, hidden_dims, strict=True)]
        )
        self.merge: torch.nn.Linear | None
        if mode == "concat":
            self.merge = torch.nn.Linear(dim, dim)
        else:
            self.merge = None

        for block in self.blocks:
            init_subspace_mlp_near_zero(block)
        if self.merge is not None:
            torch.nn.init.xavier_uniform_(self.merge.weight, gain=0.02)
            torch.nn.init.zeros_(self.merge.bias)

    def block_slices(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        start = 0
        for d in self.block_dims:
            out.append((start, start + d))
            start += d
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.mean
        if self.merge_mode == "additive":
            y = torch.zeros_like(x)
            for (lo, hi), mlp in zip(self.block_slices(), self.blocks, strict=True):
                vj = self.basis[:, lo:hi]
                z = xc @ vj
                yj = mlp(z)
                y = y + yj @ vj.T
            return y
        parts: list[torch.Tensor] = []
        for (lo, hi), mlp in zip(self.block_slices(), self.blocks, strict=True):
            vj = self.basis[:, lo:hi]
            z = xc @ vj
            parts.append(mlp(z))
        assert self.merge is not None
        return self.merge(torch.cat(parts, dim=-1))


def equal_pca_blocks(dim: int, num_blocks: int) -> list[int]:
    if dim % num_blocks != 0:
        raise ValueError(f"dim {dim} not divisible by num_blocks {num_blocks}")
    d = dim // num_blocks
    return [d] * num_blocks


@torch.no_grad()
def fit_ffn_input_pca(
    capture_paths: list[Path],
    *,
    max_tokens: int = 200_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean, basis, eigenvalues)`` from captured ``ffn_input`` shards."""
    chunks: list[torch.Tensor] = []
    n = 0
    dim: int | None = None
    for path in capture_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        x = shard["ffn_input"].reshape(-1, shard["ffn_input"].shape[-1]).float()
        if dim is None:
            dim = int(x.shape[-1])
        take = min(x.shape[0], max_tokens - n)
        if take <= 0:
            break
        chunks.append(x[:take])
        n += take
    if n < 2 or dim is None:
        raise RuntimeError("need at least 2 tokens and one shard for PCA")
    X = torch.cat(chunks, dim=0)
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (n - 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    basis = evecs[:, order]
    return mean, basis, evals


def build_block_factorized_ffn(
    *,
    dim: int,
    num_blocks: int,
    mean: torch.Tensor,
    basis: torch.Tensor,
    merge_mode: str,
    teacher_intermediate: int = 4096,
) -> BlockFactorizedFFN:
    block_dims = equal_pca_blocks(dim, num_blocks)
    return BlockFactorizedFFN(
        dim=dim,
        block_dims=block_dims,
        basis=basis,
        mean=mean,
        merge_mode=merge_mode,
        teacher_intermediate=teacher_intermediate,
    )


def load_block_factorized_ffn(
    artifact_dir: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BlockFactorizedFFN, dict]:
    artifact_dir = Path(artifact_dir)
    ckpt_path = artifact_dir / "block_factorized_ffn.pt"
    report_path = artifact_dir / "block_factorized_ffn_report.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dim = int(report["hidden_size"])
    block_dims = list(report["block_dims"])
    mod = BlockFactorizedFFN(
        dim=dim,
        block_dims=block_dims,
        basis=ckpt["basis"],
        mean=ckpt["mean"],
        merge_mode=str(report.get("merge_mode", "additive")),
        hidden_dims=list(report["block_hidden_dims"]),
        teacher_intermediate=int(report.get("teacher_intermediate", 4096)),
    )
    mod.load_state_dict(ckpt["student"], strict=True)
    mod = mod.to(device=device, dtype=dtype).eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod, report


def apply_block_factorized_ffn(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    student, report = load_block_factorized_ffn(artifact_dir, device=device, dtype=dtype)
    layer = get_transformer_layers(model)[layer_index]
    layer.mlp = student
    return {
        "layer": layer_index,
        "artifact_dir": str(artifact_dir),
        "merge_mode": report.get("merge_mode"),
        "num_blocks": report.get("num_blocks"),
        "student_ffn_params": sum(p.numel() for p in student.parameters()),
        "mlp_replaced": True,
    }
