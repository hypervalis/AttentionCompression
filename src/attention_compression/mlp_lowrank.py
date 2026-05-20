"""Low-rank SwiGLU MLP with supervised output-PCA init (same recipe as Q/K branches)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from attention_compression.joint_qkv import LowRankBranch, init_branch_from_pca
from attention_compression.model_hub import get_transformer_layers
from attention_compression.pca import dims_for_thresholds, pca_spectrum


class LowRankLinear(torch.nn.Module):
    """Drop-in for ``nn.Linear``: ``x @ down @ up + bias`` with ``down [in,r]``, ``up [r,out]``."""

    def __init__(self, in_features: int, out_features: int, rank: int, *, bias: bool = True) -> None:
        super().__init__()
        self.branch = LowRankBranch(in_features, out_features, rank)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        if not bias:
            self.branch.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)

    @property
    def weight(self) -> torch.Tensor:
        """Dense weight ``[out_features, in_features]`` for inspection / materialization."""
        return (self.branch.up.T @ self.branch.down.T).T

    @property
    def bias(self) -> torch.Tensor:
        return self.branch.bias


class LowRankSwiGLU(torch.nn.Module):
    """OLMo-style gated MLP with low-rank ``gate``, ``up``, and ``down`` projections."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        rank_gate: int,
        rank_up: int,
        rank_down: int,
        act_fn: torch.nn.Module,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        self.gate_proj = LowRankLinear(hidden_size, intermediate_size, rank_gate, bias=bias)
        self.up_proj = LowRankLinear(hidden_size, intermediate_size, rank_up, bias=bias)
        self.down_proj = LowRankLinear(intermediate_size, hidden_size, rank_down, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def is_lowrank_linear(module: torch.nn.Module) -> bool:
    return isinstance(module, LowRankLinear)


def _freeze_linear_copy(linear: torch.nn.Linear) -> torch.nn.Linear:
    """Detached dense copy of ``linear`` (no grads)."""
    out = torch.nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    out.weight.data.copy_(linear.weight.data)
    if linear.bias is not None and out.bias is not None:
        out.bias.data.copy_(linear.bias.data)
    for p in out.parameters():
        p.requires_grad_(False)
    return out


class StagedHybridSwiGLU(torch.nn.Module):
    """SwiGLU with dense teacher maps plus optional low-rank replacements (staged training)."""

    def __init__(
        self,
        teacher_mlp: torch.nn.Module,
        *,
        ranks: dict[str, int],
        compressed: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.act_fn = teacher_mlp.act_fn
        self.hidden_size = int(teacher_mlp.gate_proj.in_features)
        self.intermediate_size = int(teacher_mlp.gate_proj.out_features)
        has_bias = teacher_mlp.gate_proj.bias is not None
        self._compressed: set[str] = set(compressed)
        rank_gate = int(ranks.get("gate", 0))
        rank_up = int(ranks.get("up", 0))
        rank_down = int(ranks.get("down", 0))

        def _proj(name: str, low: LowRankLinear | torch.nn.Linear) -> None:
            setattr(self, f"{name}_proj", low)

        if "gate" in self._compressed:
            if rank_gate < 1:
                raise ValueError("rank_gate required when gate is compressed")
            _proj("gate", LowRankLinear(self.hidden_size, self.intermediate_size, rank_gate, bias=has_bias))
        else:
            _proj("gate", _freeze_linear_copy(teacher_mlp.gate_proj))

        if "up" in self._compressed:
            if rank_up < 1:
                raise ValueError("rank_up required when up is compressed")
            _proj("up", LowRankLinear(self.hidden_size, self.intermediate_size, rank_up, bias=has_bias))
        else:
            _proj("up", _freeze_linear_copy(teacher_mlp.up_proj))

        if "down" in self._compressed:
            if rank_down < 1:
                raise ValueError("rank_down required when down is compressed")
            _proj(
                "down",
                LowRankLinear(self.intermediate_size, self.hidden_size, rank_down, bias=has_bias),
            )
        else:
            _proj("down", _freeze_linear_copy(teacher_mlp.down_proj))

    @property
    def compressed(self) -> frozenset[str]:
        return frozenset(self._compressed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.mlp_hidden(x))

    def mlp_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Post-gate, pre-``down_proj`` tensor: ``act(gate(x)) * up(x)`` (hybrid maps)."""
        return self.act_fn(self.gate_proj(x)) * self.up_proj(x)

    def rank_map(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for name in ("gate", "up", "down"):
            mod = getattr(self, f"{name}_proj")
            out[name] = mod.rank if is_lowrank_linear(mod) else None
        return out


def promote_branch_to_lowrank(
    hybrid: StagedHybridSwiGLU,
    teacher_mlp: torch.nn.Module,
    branch: str,
    rank: int,
    *,
    device: torch.device,
    paths: list[Path] | None = None,
    train_refs: list[tuple[int, int]] | None = None,
    init_pca: bool = True,
) -> StagedHybridSwiGLU:
    """Replace a dense frozen map with a low-rank branch (optional PCA init)."""
    if branch not in ("gate", "up", "down"):
        raise ValueError(f"branch must be gate|up|down, got {branch!r}")
    attr = f"{branch}_proj"
    current = getattr(hybrid, attr)
    if is_lowrank_linear(current):
        if current.rank != rank:
            raise ValueError(f"{branch} already low-rank rank {current.rank} != {rank}")
        return hybrid

    has_bias = teacher_mlp.gate_proj.bias is not None
    ref = teacher_mlp.gate_proj
    if branch == "gate":
        lr = LowRankLinear(hybrid.hidden_size, hybrid.intermediate_size, rank, bias=has_bias)
    elif branch == "up":
        lr = LowRankLinear(hybrid.hidden_size, hybrid.intermediate_size, rank, bias=has_bias)
    else:
        lr = LowRankLinear(hybrid.intermediate_size, hybrid.hidden_size, rank, bias=has_bias)
        ref = teacher_mlp.down_proj

    lr = lr.to(device=ref.weight.device, dtype=ref.weight.dtype)
    if init_pca:
        if paths is None or train_refs is None:
            raise ValueError("paths and train_refs required when init_pca=True")
        init_single_branch_from_pca(
            lr,
            teacher_mlp,
            branch,
            paths,
            train_refs,
            device=device,
            hybrid=hybrid,
        )
    setattr(hybrid, attr, lr)
    hybrid._compressed.add(branch)
    return hybrid


@torch.no_grad()
def init_single_branch_from_pca(
    branch_mod: LowRankLinear,
    teacher_mlp: torch.nn.Module,
    branch: str,
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    device: torch.device,
    hybrid: StagedHybridSwiGLU | None = None,
) -> dict:
    """PCA-init one low-rank map from teacher linear outputs on that map's exact I/O."""
    if branch == "gate":
        w = teacher_mlp.gate_proj.weight.detach().float().T.contiguous().to(device)
        pca = fit_linear_output_pca(
            paths, train_refs, projection=w, teacher_forward=teacher_mlp.gate_proj, device=device
        )
        proj = w
    elif branch == "up":
        w = teacher_mlp.up_proj.weight.detach().float().T.contiguous().to(device)
        pca = fit_linear_output_pca(
            paths, train_refs, projection=w, teacher_forward=teacher_mlp.up_proj, device=device
        )
        proj = w
    elif branch == "down":
        w = teacher_mlp.down_proj.weight.detach().float().T.contiguous().to(device)
        pca = fit_down_linear_output_pca(
            paths,
            train_refs,
            teacher_mlp=teacher_mlp,
            device=device,
            hybrid=hybrid,
        )
        proj = w
    else:
        raise ValueError(branch)
    init_branch_from_pca(branch_mod.branch, projection=proj, mean=pca["mean"], basis=pca["basis"])
    return {
        "branch": branch,
        "rank": branch_mod.rank,
        "n_tokens": int(pca["n_tokens"]),
        "hidden_source": pca.get("hidden_source"),
    }


def enable_stage_training(hybrid: StagedHybridSwiGLU, stage: str) -> None:
    """Train only the low-rank branch for ``gate`` / ``up`` / ``down``."""
    branch_to_attr = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    if stage not in branch_to_attr:
        raise ValueError(f"stage must be gate|up|down, got {stage!r}")
    for p in hybrid.parameters():
        p.requires_grad_(False)
    mod = getattr(hybrid, branch_to_attr[stage])
    if not is_lowrank_linear(mod):
        raise TypeError(f"Stage {stage!r} is still dense; call promote_branch_to_lowrank first")
    for p in mod.branch.parameters():
        p.requires_grad_(True)
    hybrid.train()


def enable_compressed_training(hybrid: StagedHybridSwiGLU) -> None:
    """Train all low-rank branches installed so far (dense teacher copies stay frozen)."""
    for p in hybrid.parameters():
        p.requires_grad_(False)
    for name in hybrid.compressed:
        mod = getattr(hybrid, f"{name}_proj")
        if not is_lowrank_linear(mod):
            continue
        for p in mod.branch.parameters():
            p.requires_grad_(True)
    hybrid.train()


def stage_train_targets(
    hybrid: StagedHybridSwiGLU,
    teacher_mlp: torch.nn.Module,
    x: torch.Tensor,
    *,
    stage: str,
    loss_target: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pred, tgt) for staged distillation.

    ``linear`` / ``isolated``: match the active map on its exact inputs — ``ffn_input``
    for gate/up; hybrid ``mlp_hidden`` for down (what ``down_proj`` sees at runtime).
    """
    if loss_target in ("linear", "isolated"):
        if stage == "gate":
            return hybrid.gate_proj(x), teacher_mlp.gate_proj(x)
        if stage == "up":
            return hybrid.up_proj(x), teacher_mlp.up_proj(x)
        if stage == "down":
            h = hybrid.mlp_hidden(x)
            return hybrid.down_proj(h), teacher_mlp.down_proj(h)
        raise ValueError(stage)
    if loss_target == "mlp":
        return hybrid(x), teacher_mlp(x)
    raise ValueError(f"loss_target must be mlp|linear|isolated, got {loss_target!r}")


def staged_hybrid_to_lowrank(hybrid: StagedHybridSwiGLU) -> LowRankSwiGLU:
    """Materialize a full low-rank SwiGLU when all three maps are compressed."""
    ranks = hybrid.rank_map()
    if any(ranks[k] is None for k in ("gate", "up", "down")):
        missing = [k for k, r in ranks.items() if r is None]
        raise ValueError(f"Cannot merge: still dense maps {missing}")
    student = LowRankSwiGLU(
        hidden_size=hybrid.hidden_size,
        intermediate_size=hybrid.intermediate_size,
        rank_gate=int(ranks["gate"]),
        rank_up=int(ranks["up"]),
        rank_down=int(ranks["down"]),
        act_fn=hybrid.act_fn,
        bias=hybrid.gate_proj.bias is not None,
    )
    student.load_state_dict(hybrid.state_dict(), strict=True)
    return student


def export_lowrank_mlp_artifact(
    hybrid: StagedHybridSwiGLU,
    out_dir: str | Path,
    *,
    report: dict,
    student_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write ``lowrank_mlp.pt`` + ``lowrank_mlp_report.json`` for script 49."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student = staged_hybrid_to_lowrank(hybrid)
    state = student_state or {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save({"student": state}, out_dir / "lowrank_mlp.pt")
    (out_dir / "lowrank_mlp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_dir


def _accumulate_output_stats(
    z: torch.Tensor,
    *,
    n: int,
    sum_y: torch.Tensor,
    yty: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    n += int(z.shape[0])
    sum_y = sum_y + z.sum(dim=0)
    yty = yty + z.T @ z
    return n, sum_y, yty


@torch.no_grad()
def fit_linear_output_pca(
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    projection: torch.Tensor,
    teacher_forward: torch.nn.Module,
    device: torch.device,
    batch_rows: int = 4096,
) -> dict[str, torch.Tensor]:
    """PCA of teacher linear outputs ``z = ffn_input @ projection`` on the train split."""
    out_dim = int(projection.shape[1])
    n = 0
    sum_y = torch.zeros(out_dim, device=device)
    yty = torch.zeros(out_dim, out_dim, device=device)
    by_path: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in train_refs:
        by_path[path_i].append(sample_i)

    for path_i, sample_ids in by_path.items():
        shard = torch.load(paths[path_i], map_location="cpu", weights_only=False)
        x_all = shard["ffn_input"][sample_ids].reshape(-1, shard["ffn_input"].shape[-1]).float()
        for start in range(0, x_all.shape[0], batch_rows):
            x = x_all[start : start + batch_rows].to(device=device)
            z = teacher_forward(x)
            n, sum_y, yty = _accumulate_output_stats(z, n=n, sum_y=sum_y, yty=yty)

    mean = sum_y / max(n, 1)
    cov = (yty - n * torch.outer(mean, mean)) / max(n - 1, 1)
    cov_cpu = cov.cpu()
    evals, cumulative = pca_spectrum(cov_cpu.numpy())
    vals, vecs = torch.linalg.eigh(cov_cpu)
    order = torch.argsort(vals, descending=True)
    basis = vecs[:, order].to(device=device, dtype=torch.float32)
    return {
        "mean": mean,
        "basis": basis,
        "eigenvalues": torch.from_numpy(evals).to(device=device),
        "cumulative": torch.from_numpy(cumulative).to(device=device),
        "n_tokens": n,
    }


def rank_from_pca(pca: dict[str, torch.Tensor], threshold: float) -> int:
    cumulative = pca["cumulative"]
    if isinstance(cumulative, torch.Tensor):
        cumulative = cumulative.cpu().numpy()
    dims = dims_for_thresholds(cumulative, [threshold])
    return int(dims[f"{threshold:.4f}"])


def choose_rank(
    pca: dict[str, torch.Tensor],
    threshold: float,
    *,
    cap: int | None = None,
    floor: int = 1,
) -> int:
    r = rank_from_pca(pca, threshold)
    if cap is not None:
        r = min(r, cap)
    return max(r, floor)


@torch.no_grad()
def fit_down_linear_output_pca(
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    teacher_mlp: torch.nn.Module,
    device: torch.device,
    hybrid: StagedHybridSwiGLU | None = None,
    batch_rows: int = 4096,
) -> dict[str, torch.Tensor]:
    """PCA of ``down_proj(h)`` with ``h`` from hybrid maps when provided, else teacher."""
    hidden_source = "hybrid" if hybrid is not None else "teacher"
    out_dim = int(teacher_mlp.down_proj.out_features)
    n = 0
    sum_y = torch.zeros(out_dim, device=device)
    yty = torch.zeros(out_dim, out_dim, device=device)
    by_path: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in train_refs:
        by_path[path_i].append(sample_i)
    for path_i, sample_ids in by_path.items():
        shard = torch.load(paths[path_i], map_location="cpu", weights_only=False)
        x_all = shard["ffn_input"][sample_ids].reshape(-1, shard["ffn_input"].shape[-1]).float()
        for start in range(0, x_all.shape[0], batch_rows):
            x = x_all[start : start + batch_rows].to(device=device)
            if hybrid is not None:
                h = hybrid.mlp_hidden(x)
            else:
                h = teacher_mlp.act_fn(teacher_mlp.gate_proj(x)) * teacher_mlp.up_proj(x)
            z = teacher_mlp.down_proj(h)
            n, sum_y, yty = _accumulate_output_stats(z, n=n, sum_y=sum_y, yty=yty)
    mean = sum_y / max(n, 1)
    cov = (yty - n * torch.outer(mean, mean)) / max(n - 1, 1)
    cov_cpu = cov.cpu()
    evals, cumulative = pca_spectrum(cov_cpu.numpy())
    vals, vecs = torch.linalg.eigh(cov_cpu)
    order = torch.argsort(vals, descending=True)
    return {
        "mean": mean,
        "basis": vecs[:, order].to(device=device, dtype=torch.float32),
        "eigenvalues": torch.from_numpy(evals).to(device=device),
        "cumulative": torch.from_numpy(cumulative).to(device=device),
        "n_tokens": n,
        "hidden_source": hidden_source,
    }


@torch.no_grad()
def fit_down_output_pca(
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    teacher_mlp: torch.nn.Module,
    device: torch.device,
    batch_rows: int = 4096,
) -> dict[str, torch.Tensor]:
    """Backward-compatible alias: down PCA with teacher-only ``mlp_hidden``."""
    return fit_down_linear_output_pca(
        paths,
        train_refs,
        teacher_mlp=teacher_mlp,
        device=device,
        hybrid=None,
        batch_rows=batch_rows,
    )


@torch.no_grad()
def estimate_mlp_ranks_from_pca(
    teacher_mlp: torch.nn.Module,
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    device: torch.device,
    variance_threshold: float = 0.95,
    ranks: dict[str, int] | None = None,
    rank_cap: int | None = 512,
) -> tuple[dict[str, int], dict]:
    """Choose ranks from output PCA (or explicit overrides)."""
    gate_w = teacher_mlp.gate_proj.weight.detach().float().T.contiguous().to(device)
    up_w = teacher_mlp.up_proj.weight.detach().float().T.contiguous().to(device)
    pca_gate = fit_linear_output_pca(
        paths, train_refs, projection=gate_w, teacher_forward=teacher_mlp.gate_proj, device=device
    )
    pca_up = fit_linear_output_pca(
        paths, train_refs, projection=up_w, teacher_forward=teacher_mlp.up_proj, device=device
    )
    pca_down = fit_down_output_pca(paths, train_refs, teacher_mlp=teacher_mlp, device=device)
    overrides = ranks or {}
    def _pick(name: str, pca: dict[str, torch.Tensor]) -> int:
        if overrides.get(name, 0):
            return int(overrides[name])
        return choose_rank(pca, variance_threshold, cap=rank_cap)

    chosen = {"gate": _pick("gate", pca_gate), "up": _pick("up", pca_up), "down": _pick("down", pca_down)}

    def _pca_entry(pca: dict[str, torch.Tensor], rank: int) -> dict:
        return {
            "rank": rank,
            "rank_cap": rank_cap,
            "pca_ranks": {f"var_{int(t * 100)}pct": rank_from_pca(pca, t) for t in (0.9, 0.95, 0.99)},
        }

    report = {
        "variance_threshold": variance_threshold,
        "rank_cap": rank_cap,
        "gate": _pca_entry(pca_gate, chosen["gate"]),
        "up": _pca_entry(pca_up, chosen["up"]),
        "down": _pca_entry(pca_down, chosen["down"]),
    }
    return chosen, report


@torch.no_grad()
def init_lowrank_mlp_from_supervised_pca(
    student: LowRankSwiGLU,
    teacher_mlp: torch.nn.Module,
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    device: torch.device,
    variance_threshold: float = 0.95,
    ranks: dict[str, int] | None = None,
    rank_cap: int | None = 512,
) -> dict:
    """Initialize ``gate`` / ``up`` / ``down`` branches from output PCA on teacher activations."""
    gate_w = teacher_mlp.gate_proj.weight.detach().float().T.contiguous().to(device)
    up_w = teacher_mlp.up_proj.weight.detach().float().T.contiguous().to(device)
    down_w = teacher_mlp.down_proj.weight.detach().float().T.contiguous().to(device)

    pca_gate = fit_linear_output_pca(
        paths, train_refs, projection=gate_w, teacher_forward=teacher_mlp.gate_proj, device=device
    )
    pca_up = fit_linear_output_pca(
        paths, train_refs, projection=up_w, teacher_forward=teacher_mlp.up_proj, device=device
    )
    pca_down = fit_down_output_pca(paths, train_refs, teacher_mlp=teacher_mlp, device=device)

    overrides = ranks or {}
    rank_gate = int(overrides.get("gate", 0)) or choose_rank(pca_gate, variance_threshold, cap=rank_cap)
    rank_up = int(overrides.get("up", 0)) or choose_rank(pca_up, variance_threshold, cap=rank_cap)
    rank_down = int(overrides.get("down", 0)) or choose_rank(pca_down, variance_threshold, cap=rank_cap)

    if student.gate_proj.rank != rank_gate:
        raise ValueError(f"student gate rank {student.gate_proj.rank} != estimated {rank_gate}")
    if student.up_proj.rank != rank_up:
        raise ValueError(f"student up rank {student.up_proj.rank} != estimated {rank_up}")
    if student.down_proj.rank != rank_down:
        raise ValueError(f"student down rank {student.down_proj.rank} != estimated {rank_down}")

    init_branch_from_pca(student.gate_proj.branch, projection=gate_w, mean=pca_gate["mean"], basis=pca_gate["basis"])
    init_branch_from_pca(student.up_proj.branch, projection=up_w, mean=pca_up["mean"], basis=pca_up["basis"])
    init_branch_from_pca(student.down_proj.branch, projection=down_w, mean=pca_down["mean"], basis=pca_down["basis"])

    return {
        "init": "supervised_output_pca",
        "variance_threshold": variance_threshold,
        "rank_cap": rank_cap,
        "rank_gate": rank_gate,
        "rank_up": rank_up,
        "rank_down": rank_down,
    }


def build_lowrank_mlp_from_teacher(
    teacher_mlp: torch.nn.Module,
    *,
    rank_gate: int,
    rank_up: int,
    rank_down: int,
) -> LowRankSwiGLU:
    hidden = int(teacher_mlp.gate_proj.in_features)
    inter = int(teacher_mlp.gate_proj.out_features)
    act = teacher_mlp.act_fn
    has_bias = teacher_mlp.gate_proj.bias is not None
    return LowRankSwiGLU(
        hidden_size=hidden,
        intermediate_size=inter,
        rank_gate=rank_gate,
        rank_up=rank_up,
        rank_down=rank_down,
        act_fn=act,
        bias=has_bias,
    )


def load_lowrank_mlp(
    artifact_dir: str | Path,
    *,
    teacher_mlp: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LowRankSwiGLU, dict]:
    artifact_dir = Path(artifact_dir)
    ckpt = torch.load(artifact_dir / "lowrank_mlp.pt", map_location="cpu", weights_only=False)
    report = json.loads((artifact_dir / "lowrank_mlp_report.json").read_text(encoding="utf-8"))
    student = build_lowrank_mlp_from_teacher(
        teacher_mlp,
        rank_gate=int(report["rank_gate"]),
        rank_up=int(report["rank_up"]),
        rank_down=int(report["rank_down"]),
    )
    student.load_state_dict(ckpt["student"], strict=True)
    student = student.to(device=device, dtype=dtype).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student, report


def apply_lowrank_mlp(
    model: torch.nn.Module,
    *,
    layer_index: int,
    artifact_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    layer = get_transformer_layers(model)[layer_index]
    teacher_mlp = layer.mlp
    student, report = load_lowrank_mlp(
        artifact_dir, teacher_mlp=teacher_mlp, device=device, dtype=dtype
    )
    layer.mlp = student
    return {
        "layer": layer_index,
        "artifact_dir": str(artifact_dir),
        "rank_gate": report.get("rank_gate"),
        "rank_up": report.get("rank_up"),
        "rank_down": report.get("rank_down"),
        "student_params": sum(p.numel() for p in student.parameters()),
        "mlp_replaced": True,
    }
