"""Bottleneck FFN + mimic o_proj apply (script 32 training / inference)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_compression.compress.artifacts import resolve_oproj_swap_dir
from attention_compression.model_hub import get_transformer_layers
from attention_compression.oproj import LowRankProjection, build_projector_from_state, load_mimic_oproj_projector


class MimicOProj(torch.nn.Module):
    def __init__(self, proj: torch.nn.Module) -> None:
        super().__init__()
        self.proj = proj

    def forward(self, heads_flat: torch.Tensor) -> torch.Tensor:
        return self.proj(heads_flat)


class BottleneckFFN(torch.nn.Module):
    def __init__(self, dim: int, bottleneck: int, hidden: int) -> None:
        super().__init__()
        self.down = torch.nn.Linear(dim, bottleneck)
        self.fc = torch.nn.Linear(bottleneck, hidden)
        self.up = torch.nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.gelu(self.down(x))
        h = torch.nn.functional.gelu(self.fc(z))
        return self.up(h)


class CoupledBottleneckFFNBlock(torch.nn.Module):
    """Mimic + student; ``heads`` from pre-``o_proj`` hook when the MLP bridge needs ``R_mimic``."""

    def __init__(self, mimic: MimicOProj, student: BottleneckFFN) -> None:
        super().__init__()
        self.mimic = mimic
        self.student = student
        self._heads_flat: torch.Tensor | None = None

    def set_heads(self, heads_flat: torch.Tensor | None) -> None:
        self._heads_flat = heads_flat

    def r_mimic(self, ffn_input: torch.Tensor) -> torch.Tensor:
        if self._heads_flat is None:
            raise RuntimeError("head contexts not set (o_proj pre-hook must run first)")
        heads = self._heads_flat
        if heads.dtype != ffn_input.dtype:
            heads = heads.to(dtype=ffn_input.dtype)
        return ffn_input - self.mimic(heads)


class _HeadCapture:
    def __init__(self) -> None:
        self.heads_flat: torch.Tensor | None = None

    def hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.heads_flat = inputs[0]


class _MLPBridge(torch.nn.Module):
    """Deploy script-32 student on ``R_mimic`` with teacher FFN delta scale (Olmo MLP is nonlinear)."""

    def __init__(self, block: CoupledBottleneckFFNBlock, teacher_mlp: torch.nn.Module) -> None:
        super().__init__()
        self.block = block
        self.teacher_mlp = teacher_mlp
        for p in self.teacher_mlp.parameters():
            p.requires_grad_(False)
        self.teacher_mlp.eval()

    def forward(self, ffn_input: torch.Tensor) -> torch.Tensor:
        r_mimic = self.block.r_mimic(ffn_input)
        student_delta = self.block.student(r_mimic)
        with torch.no_grad():
            teacher_on_ffn = self.teacher_mlp(ffn_input)
            teacher_on_r = self.teacher_mlp(r_mimic)
        scale = teacher_on_ffn.norm(dim=-1, keepdim=True) / teacher_on_r.norm(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        return teacher_on_ffn + (student_delta - teacher_on_r) * scale


def load_coupled_ffn_block(
    artifact_dir: str | Path,
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[CoupledBottleneckFFNBlock, dict, torch.nn.Module]:
    """Return ``(block, report, mimic_projector)``; projector is the inner ``o_proj`` map."""
    artifact_dir = Path(artifact_dir)
    ckpt_path = artifact_dir / "bottleneck_ffn.pt"
    report_path = artifact_dir / "bottleneck_ffn_report.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path} (run train --target ffn or layer first)")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    in_dim = num_heads * head_dim
    bottleneck_dim = int(report.get("bottleneck_dim", 1024))
    ffn_hidden_dim = int(report.get("ffn_hidden_dim", 4096))

    mimic_state = ckpt["mimic_oproj"]
    if any(k.startswith("proj.") for k in mimic_state):
        inner_state = {k[len("proj.") :]: v for k, v in mimic_state.items() if k.startswith("proj.")}
    else:
        inner_state = mimic_state

    projector = build_projector_from_state(inner_state, in_dim=in_dim, out_dim=hidden_size, report=report)
    mimic = MimicOProj(projector)

    student = BottleneckFFN(hidden_size, bottleneck_dim, ffn_hidden_dim)
    student.load_state_dict(ckpt["student_ffn"], strict=True)

    block = CoupledBottleneckFFNBlock(mimic, student)
    block = block.to(device=device, dtype=dtype)
    for p in block.parameters():
        p.requires_grad_(False)
    block.eval()
    projector = projector.to(device=device, dtype=dtype)
    for p in projector.parameters():
        p.requires_grad_(False)
    projector.eval()
    return block, report, projector


def _register_o_proj_head_hook(
    layer: torch.nn.Module,
    block: CoupledBottleneckFFNBlock,
) -> None:
    if not hasattr(layer, "_attention_compression_hooks"):
        layer._attention_compression_hooks = []  # type: ignore[attr-defined]

    def _pre_o_proj(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        block.set_heads(inputs[0])

    hook = layer.self_attn.o_proj.register_forward_pre_hook(_pre_o_proj)
    layer._attention_compression_hooks.append(hook)  # type: ignore[attr-defined]


def apply_mimic_oproj_layer(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    checkpoint_root: str | Path | None = None,
    projection_rank: int = 768,
) -> dict:
    """Swap ``o_proj`` for script-31 low-rank mimic (same output dim; ~75% params at rank 768)."""
    in_dim = num_heads * head_dim
    root = Path(checkpoint_root) if checkpoint_root is not None else Path(artifact_dir)
    swap_dir = resolve_oproj_swap_dir(
        root, layer=layer_index, projection_rank=projection_rank
    )
    projector, report = load_mimic_oproj_projector(
        swap_dir,
        in_dim=in_dim,
        out_dim=int(getattr(model.config, "hidden_size", in_dim)),
        device=device,
        dtype=dtype,
    )
    layers = get_transformer_layers(model)
    layers[layer_index].self_attn.o_proj = projector
    rank = int(projector.down.weight.shape[0]) if hasattr(projector, "down") else 0
    return {
        "layer": layer_index,
        "artifact_dir": str(swap_dir),
        "o_proj_swapped": True,
        "oproj_rank": int(report.get("oproj_rank") or report.get("projection_rank") or rank),
        "oproj_params": sum(p.numel() for p in projector.parameters()),
        "mlp_bridge": False,
        "oproj_source": "script31" if (Path(swap_dir) / "compressed_oproj.pt").is_file() else "script32_mimic",
    }


def apply_ffn_block(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    swap_o_proj: bool = True,
    install_mlp_bridge: bool = True,
    checkpoint_root: str | Path | None = None,
    projection_rank: int = 768,
) -> dict:
    """Install mimic ``o_proj`` (optional swap) and optional script-32 MLP bridge."""
    hidden_size = int(getattr(model.config, "hidden_size", num_heads * head_dim))
    block, report, _ = load_coupled_ffn_block(
        artifact_dir,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    layers = get_transformer_layers(model)
    layer = layers[layer_index]
    o_proj_swapped = False
    swap_dir = resolve_oproj_swap_dir(
        Path(checkpoint_root) if checkpoint_root is not None else Path(artifact_dir),
        layer=layer_index,
        projection_rank=projection_rank,
    )
    swap_proj, swap_report = load_mimic_oproj_projector(
        swap_dir,
        in_dim=num_heads * head_dim,
        out_dim=hidden_size,
        device=device,
        dtype=dtype,
    )

    if swap_o_proj:
        layer.self_attn.o_proj = swap_proj
        o_proj_swapped = True

    meta: dict = {
        "layer": layer_index,
        "artifact_dir": str(artifact_dir),
        "oproj_swap_dir": str(swap_dir),
        "oproj_rank": int(
            swap_report.get("oproj_rank") or swap_report.get("projection_rank") or report.get("oproj_rank", 0)
        ),
        "bottleneck_dim": int(report.get("bottleneck_dim", 1024)),
        "ffn_hidden_dim": int(report.get("ffn_hidden_dim", 4096)),
        "o_proj_swapped": o_proj_swapped,
        "oproj_params": sum(p.numel() for p in swap_proj.parameters()),
        "mlp_bridge": install_mlp_bridge,
        "oproj_source": "script31" if (swap_dir / "compressed_oproj.pt").is_file() else "script32_mimic",
    }

    if install_mlp_bridge:
        _register_o_proj_head_hook(layer, block)
        teacher_mlp = layer.mlp
        layer.mlp = _MLPBridge(block, teacher_mlp)
        meta["coupled_forward"] = (
            "teacher(ffn_in) + (student(R_mimic)-teacher(R_mimic)) * "
            "||teacher(ffn_in)||/||teacher(R_mimic)||"
        )
        meta["teacher_mlp_retained"] = True
    elif swap_o_proj:
        meta["note"] = "mimic o_proj only; teacher MLP unchanged"

    return meta
