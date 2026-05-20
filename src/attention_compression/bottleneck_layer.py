"""Coupled bottleneck width ``b`` for ``o_proj`` and FFN (e.g. b=1536 on hidden 2048)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_compression.model_hub import get_transformer_layers


class BottleneckMap(torch.nn.Module):
    """``in_dim -> bottleneck -> out_dim`` with GELU in the bottleneck (FFN student body)."""

    def __init__(self, in_dim: int, bottleneck: int, out_dim: int, *, hidden: int | None = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck = bottleneck
        self.out_dim = out_dim
        hidden = hidden if hidden is not None else max(bottleneck * 2, 4 * bottleneck)
        self.down = torch.nn.Linear(in_dim, bottleneck)
        self.fc = torch.nn.Linear(bottleneck, hidden)
        self.up = torch.nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.gelu(self.down(x))
        h = torch.nn.functional.gelu(self.fc(z))
        return self.up(h)


class BottleneckOProj(torch.nn.Module):
    """``heads [*, in_dim] -> [*, out_dim]`` via bottleneck (no nonlinearity between down/up)."""

    def __init__(self, in_dim: int, bottleneck: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.down = torch.nn.Linear(in_dim, bottleneck, bias=False)
        self.up = torch.nn.Linear(bottleneck, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    @classmethod
    def from_teacher_linear(
        cls, teacher: torch.nn.Linear, bottleneck: int, *, bias: bool = True
    ) -> BottleneckOProj:
        in_dim = int(teacher.in_features)
        out_dim = int(teacher.out_features)
        mod = cls(in_dim, bottleneck, out_dim, bias=bias and teacher.bias is not None)
        w = teacher.weight.detach().float().cpu()
        b = teacher.bias.detach().float().cpu() if teacher.bias is not None else None
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        r = min(bottleneck, s.numel())
        sqrt_s = s[:r].sqrt()
        mod.down.weight.data.copy_(sqrt_s.unsqueeze(1) * vh[:r, :])
        mod.up.weight.data.copy_(u[:, :r] * sqrt_s.unsqueeze(0))
        if b is not None and mod.up.bias is not None:
            mod.up.bias.data.copy_(b)
        return mod


def init_bottleneck_map_identityish(mod: BottleneckMap) -> None:
    """Start FFN student near zero delta on bottleneck path."""
    torch.nn.init.xavier_uniform_(mod.down.weight, gain=0.02)
    torch.nn.init.zeros_(mod.down.bias)
    torch.nn.init.xavier_uniform_(mod.fc.weight, gain=0.02)
    torch.nn.init.zeros_(mod.fc.bias)
    torch.nn.init.xavier_uniform_(mod.up.weight, gain=0.02)
    if mod.up.bias is not None:
        torch.nn.init.zeros_(mod.up.bias)


def load_coupled_b1536(
    artifact_dir: str | Path,
    *,
    hidden_size: int,
    bottleneck: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BottleneckOProj, BottleneckMap, dict]:
    artifact_dir = Path(artifact_dir)
    ckpt_path = artifact_dir / "coupled_b1536.pt"
    report_path = artifact_dir / "coupled_b1536_report.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    b = int(report.get("bottleneck_dim", bottleneck))
    ffn_hidden = int(report.get("ffn_hidden_dim", max(4 * b, 3072)))

    has_bias = "up.bias" in ckpt["o_proj"]
    o_proj = BottleneckOProj(hidden_size, b, hidden_size, bias=has_bias)
    o_proj.load_state_dict(ckpt["o_proj"], strict=True)
    student = BottleneckMap(hidden_size, b, hidden_size, hidden=ffn_hidden)
    student.load_state_dict(ckpt["student_ffn"], strict=True)

    o_proj = o_proj.to(device=device, dtype=dtype).eval()
    student = student.to(device=device, dtype=dtype).eval()
    for p in list(o_proj.parameters()) + list(student.parameters()):
        p.requires_grad_(False)
    return o_proj, student, report


def apply_coupled_b1536_layer(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    bottleneck: int = 1536,
) -> dict:
    """Replace ``o_proj`` and ``mlp`` with coupled ``b``-dim bottlenecks."""
    hidden_size = int(getattr(model.config, "hidden_size", 2048))
    o_proj, student, report = load_coupled_b1536(
        artifact_dir,
        hidden_size=hidden_size,
        bottleneck=bottleneck,
        device=device,
        dtype=dtype,
    )
    layer = get_transformer_layers(model)[layer_index]
    layer.self_attn.o_proj = o_proj
    layer.mlp = student
    return {
        "layer": layer_index,
        "artifact_dir": str(artifact_dir),
        "bottleneck_dim": int(report.get("bottleneck_dim", bottleneck)),
        "ffn_hidden_dim": int(report.get("ffn_hidden_dim", 0)),
        "o_proj_params": sum(p.numel() for p in o_proj.parameters()),
        "student_ffn_params": sum(p.numel() for p in student.parameters()),
        "o_proj_swapped": True,
        "mlp_replaced": True,
    }
