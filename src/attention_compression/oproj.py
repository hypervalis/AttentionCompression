"""Apply trained low-rank or dense ``o_proj`` replacements (scripts 31–32)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_compression.model_hub import get_self_attn


class LowRankProjection(torch.nn.Module):
    """``in_dim -> rank -> out_dim`` (output dim stays full hidden size)."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, *, bias: bool = True) -> None:
        super().__init__()
        self.down = torch.nn.Linear(in_dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_dim, bias=bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(z))


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    plen = len(prefix)
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def build_projector_from_state(
    state: dict[str, torch.Tensor],
    *,
    in_dim: int,
    out_dim: int | None = None,
    report: dict | None = None,
) -> torch.nn.Module:
    """Build a projector from script-31 ``projector`` or script-32 ``mimic_oproj`` weights."""
    report = report or {}
    out_dim = out_dim if out_dim is not None else in_dim

    if any(k.startswith("proj.") for k in state):
        state = _strip_prefix(state, "proj.")

    if "weight" in state:
        layer = torch.nn.Linear(in_dim, out_dim, bias="bias" in state)
        layer.load_state_dict(state, strict=True)
        return layer

    if "down.weight" in state:
        rank = int(report.get("projection_rank") or report.get("oproj_rank") or state["down.weight"].shape[0])
        has_bias = "up.bias" in state
        mod = LowRankProjection(in_dim, out_dim, rank, bias=has_bias)
        mod.load_state_dict(state, strict=True)
        return mod

    raise ValueError(f"Unrecognized o_proj checkpoint keys: {sorted(state)[:8]}...")


def _load_report(artifact_dir: Path) -> dict:
    for name in ("compressed_oproj_report.json", "bottleneck_ffn_report.json"):
        path = artifact_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_mimic_oproj_projector(
    artifact_dir: str | Path,
    *,
    in_dim: int,
    out_dim: int | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, dict]:
    """Load mimic ``o_proj`` from script-31 dir or ``bottleneck_ffn.pt`` (script 32)."""
    artifact_dir = Path(artifact_dir)
    report = _load_report(artifact_dir)
    out_dim = out_dim if out_dim is not None else in_dim

    script31 = artifact_dir / "compressed_oproj.pt"
    script32 = artifact_dir / "bottleneck_ffn.pt"
    if script31.is_file():
        ckpt = torch.load(script31, map_location="cpu", weights_only=False)
        state = ckpt["projector"]
    elif script32.is_file():
        ckpt = torch.load(script32, map_location="cpu", weights_only=False)
        state = ckpt["mimic_oproj"]
        if any(k.startswith("proj.") for k in state):
            state = _strip_prefix(state, "proj.")
    else:
        raise FileNotFoundError(f"Need {script31} or {script32} under {artifact_dir}")

    projector = build_projector_from_state(state, in_dim=in_dim, out_dim=out_dim, report=report)
    projector = projector.to(device=device, dtype=dtype)
    for p in projector.parameters():
        p.requires_grad_(False)
    projector.eval()
    return projector, report


def build_projector_from_checkpoint(
    ckpt: dict[str, dict[str, torch.Tensor]],
    *,
    report: dict,
) -> torch.nn.Module:
    """Build from script-31 ``compressed_oproj.pt`` payload."""
    in_dim = int(report.get("in_dim", 0))
    if in_dim <= 0:
        proj_state = ckpt["projector"]
        w = proj_state.get("weight") or proj_state.get("up.weight")
        if w is None:
            raise ValueError("Cannot infer in_dim; add compressed_oproj_report.json")
        in_dim = int(w.shape[1])
    return build_projector_from_state(ckpt["projector"], in_dim=in_dim, report=report)


def apply_mimic_oproj(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    in_dim: int,
    out_dim: int | None = None,
) -> dict:
    """Replace ``self_attn.o_proj`` with a trained low-rank (or dense) mimic; output dim unchanged."""
    projector, report = load_mimic_oproj_projector(
        artifact_dir,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
        dtype=dtype,
    )
    attn = get_self_attn(model, layer_index)
    attn.o_proj = projector
    rank = getattr(projector, "down", None)
    oproj_rank = int(rank.weight.shape[0]) if rank is not None else 0
    return {
        "layer": layer_index,
        "artifact_dir": str(artifact_dir),
        "o_proj_swapped": True,
        "projection_kind": report.get("projection_kind", "lowrank"),
        "oproj_rank": int(report.get("oproj_rank") or report.get("projection_rank") or oproj_rank),
        "oproj_params": sum(p.numel() for p in projector.parameters()),
    }


def apply_compressed_oproj(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Replace ``self_attn.o_proj`` with a checkpoint from script 31 only."""
    artifact_dir = Path(artifact_dir)
    ckpt_path = artifact_dir / "compressed_oproj.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    report = _load_report(artifact_dir)
    in_dim = int(report.get("in_dim", 0))
    if in_dim <= 0:
        raise ValueError("compressed_oproj_report.json must include in_dim")
    return apply_mimic_oproj(
        model,
        layer_index=layer_index,
        artifact_dir=artifact_dir,
        device=device,
        dtype=dtype,
        in_dim=in_dim,
    )
