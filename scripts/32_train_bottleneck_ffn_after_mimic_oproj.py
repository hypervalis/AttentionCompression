#!/usr/bin/env python3
"""Train a bottleneck FFN after a mimic `o_proj`.

Pipeline on captured layer internals:

1. ``head_contexts`` is the concatenated head output **before** teacher ``o_proj``.
2. ``ffn_input`` is the normalized residual stream **entering** the teacher FFN, so it
   already includes ``teacher_o_proj(heads)`` in the residual.
3. The FFN input under a replacement ``o_proj`` is ``R_mimic = ffn_input - mimic_o_proj(heads)``
   when only the attention output map changes (residual add is ``+ o_proj(heads)``).

Then co-train a student FFN **and** the mimic ``o_proj`` (initialized from the script 31
checkpoint) against ``teacher_mlp(R_mimic)``, where ``teacher_mlp`` stays frozen. Gradients
flow through ``R_mimic = ffn_input - mimic(heads)`` into the mimic so the two modules
can settle a joint solution.

Training loss defaults to **directional** ``1 - cos(pred, target)`` per token (``--loss-kind cosine``),
which tracks residual-stream alignment better than pooled relative MSE; ``relative`` and
``both`` remain available.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention_compression.activations import find_transformer_layers
from attention_compression.attention_metrics import cosine_similarity_mean, relative_mse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bottleneck FFN distillation after mimic o_proj.")
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--compressed-oproj-pt", required=True)
    p.add_argument("--oproj-projection-kind", required=True, choices=["dense", "lowrank", "pca_lowrank"])
    p.add_argument("--oproj-rank", type=int, default=768)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--bottleneck-dim", type=int, default=1024)
    p.add_argument("--ffn-hidden-dim", type=int, default=4096)
    p.add_argument("--ae-state", required=True)
    p.add_argument("--ae-kind", default="decoder_residual_mlp", choices=["linear", "decoder_residual_mlp"])
    p.add_argument("--ae-hidden-dim", type=int, default=1536)
    p.add_argument("--train-windows-per-bin", type=int, default=128)
    p.add_argument("--eval-windows-per-bin", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--oproj-lr",
        type=float,
        default=None,
        help="LR for mimic o_proj; defaults to --lr.",
    )
    p.add_argument(
        "--loss-kind",
        default="cosine",
        choices=["relative", "cosine", "both"],
        help="Training objective: batch relative MSE to target, mean 1-cosine per token, or weighted mix.",
    )
    p.add_argument(
        "--loss-relative-weight",
        type=float,
        default=0.25,
        help="Weight on relative loss when --loss-kind both (cosine weight is 1 minus this).",
    )
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def refs_by_path(refs: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in refs:
        out[path_i].append(sample_i)
    return out


def group_rows_by_bin(paths: list[Path]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        for sample_i, row in enumerate(shard["rows"]):
            groups[row["rarity_bin"]].append((path_i, sample_i))
    return groups


def choose_train_eval(groups, train_n: int, eval_n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    train, eval_ = [], []
    for refs in groups.values():
        if len(refs) < train_n + eval_n:
            raise ValueError(f"Need {train_n + eval_n} rows per bin; got {len(refs)}")
        perm = torch.randperm(len(refs), generator=gen).tolist()
        train.extend(refs[i] for i in perm[:train_n])
        eval_.extend(refs[i] for i in perm[train_n : train_n + eval_n])
    return train, eval_


def batch_refs(paths: list[Path], refs: list[tuple[int, int]], batch_size: int, *, shuffle: bool = False, seed: int = 0):
    grouped = refs_by_path(refs)
    items = list(grouped.items())
    gen = torch.Generator().manual_seed(seed)
    if shuffle and items:
        order = torch.randperm(len(items), generator=gen).tolist()
        items = [items[i] for i in order]
    for path_i, ids in items:
        ids = list(ids)
        if shuffle and ids:
            order = torch.randperm(len(ids), generator=gen).tolist()
            ids = [ids[i] for i in order]
        shard = torch.load(paths[path_i], map_location="cpu")
        for start in range(0, len(ids), batch_size):
            yield shard, ids[start : start + batch_size]


class FrozenAutoencoder(torch.nn.Module):
    def __init__(self, dim: int, bottleneck: int, hidden_dim: int, kind: str, state_path: str) -> None:
        super().__init__()
        state = torch.load(state_path, map_location="cpu")
        self.kind = kind
        self.encoder = torch.nn.Linear(dim, bottleneck)
        self.decoder = torch.nn.Linear(bottleneck, dim)
        self.encoder.weight.data.copy_(state["encoder.weight"].float())
        self.encoder.bias.data.copy_(state["encoder.bias"].float())
        self.decoder.weight.data.copy_(state["decoder.weight"].float())
        self.decoder.bias.data.copy_(state["decoder.bias"].float())
        if kind == "decoder_residual_mlp":
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(bottleneck, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, dim),
            )
            self.residual[0].weight.data.copy_(state["residual.0.weight"].float())
            self.residual[0].bias.data.copy_(state["residual.0.bias"].float())
            self.residual[2].weight.data.copy_(state["residual.2.weight"].float())
            self.residual[2].bias.data.copy_(state["residual.2.bias"].float())
        else:
            self.residual = None
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z)
        if self.residual is not None:
            y = y + self.residual(z)
        return y

    def linearized_weight_bias(self, teacher_weight: torch.Tensor, teacher_bias: torch.Tensor | None):
        weight = teacher_weight @ self.decoder.weight.detach().cpu() @ self.encoder.weight.detach().cpu()
        decoded_bias = torch.nn.functional.linear(
            self.encoder.bias.detach().cpu(), self.decoder.weight.detach().cpu(), self.decoder.bias.detach().cpu()
        )
        if teacher_bias is None:
            bias = torch.nn.functional.linear(decoded_bias, teacher_weight, None)
        else:
            bias = torch.nn.functional.linear(decoded_bias, teacher_weight, teacher_bias)
        return weight, bias


class LowRankProjection(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, init_dense: torch.Tensor, init_bias: torch.Tensor | None):
        super().__init__()
        u, s, vh = torch.linalg.svd(init_dense.float(), full_matrices=False)
        rank = min(rank, s.numel())
        self.down = torch.nn.Linear(in_dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_dim, bias=True)
        self.up.weight.data.copy_(u[:, :rank] * s[:rank].sqrt().unsqueeze(0))
        self.down.weight.data.copy_(s[:rank].sqrt().unsqueeze(1) * vh[:rank, :])
        self.up.bias.data.copy_(init_bias.float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(z))


class MimicOProj(torch.nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        projection_kind: str,
        rank: int,
        state: dict[str, torch.Tensor],
        teacher_o_proj: torch.nn.Linear,
        autoencoder: FrozenAutoencoder,
    ) -> None:
        super().__init__()
        teacher_weight = teacher_o_proj.weight.detach().float().cpu()
        teacher_bias = None if teacher_o_proj.bias is None else teacher_o_proj.bias.detach().float().cpu()
        init_weight, init_bias = autoencoder.linearized_weight_bias(teacher_weight, teacher_bias)
        if projection_kind == "dense":
            self.proj = torch.nn.Linear(in_dim, in_dim, bias=True)
        elif projection_kind in ("lowrank", "pca_lowrank"):
            # ``pca_lowrank`` checkpoints use the same LowRankProjection parameterization as
            # ``lowrank``; supervised PCA init happens in script 31, not here.
            self.proj = LowRankProjection(in_dim, in_dim, rank, init_weight, init_bias)
        else:
            raise ValueError(projection_kind)
        missing, unexpected = self.proj.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Bad mimic o_proj checkpoint: missing={missing} unexpected={unexpected}")

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


def rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.float() - target.float()) ** 2) / torch.var(target.float()).clamp_min(1e-6)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean ``1 - cos(pred, target)`` over token positions (last dim is hidden)."""
    pred_f = pred.reshape(-1, pred.shape[-1]).float()
    target_f = target.reshape(-1, target.shape[-1]).float()
    cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1)
    return torch.mean(1.0 - cos)


def train_loss(pred: torch.Tensor, target: torch.Tensor, *, loss_kind: str, relative_weight: float) -> torch.Tensor:
    if loss_kind == "relative":
        return rel_loss(pred, target)
    if loss_kind == "cosine":
        return cosine_loss(pred, target)
    if loss_kind == "both":
        w = float(relative_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError("--loss-relative-weight must be in [0, 1] when using both")
        return w * rel_loss(pred, target) + (1.0 - w) * cosine_loss(pred, target)
    raise ValueError(loss_kind)


@torch.no_grad()
def evaluate(student, teacher_mlp, mimic_o_proj, paths, eval_refs, batch_size, device):
    student.eval()
    mimic_o_proj.eval()
    acc = defaultdict(float)
    count = 0
    for shard, ids in batch_refs(paths, eval_refs, batch_size):
        heads = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(start_dim=2)
        ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
        r_m = ffn_in - mimic_o_proj(heads)
        target = teacher_mlp(r_m)
        pred = student(r_m)
        b = heads.shape[0]
        metrics = {
            "relative_mse": relative_mse(pred, target),
            "cosine": cosine_similarity_mean(pred, target),
        }
        for key, val in metrics.items():
            acc[key] += val * b
        count += b
    return {k: v / count for k, v in acc.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(Path(args.internals_capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards in {args.internals_capture_dir}")
    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed)
    first = torch.load(paths[0], map_location="cpu")
    in_dim = int(first["head_contexts"].shape[2] * first["head_contexts"].shape[3])

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.eval().to(device)
    _layer_path, layers = find_transformer_layers(teacher)
    layer = layers[args.target_layer]
    teacher_o_proj = layer.self_attn.o_proj.to(device=device, dtype=torch.float32)
    teacher_mlp = layer.mlp.to(device=device, dtype=torch.float32)
    for param in teacher_mlp.parameters():
        param.requires_grad_(False)

    ckpt = torch.load(args.compressed_oproj_pt, map_location="cpu")
    ae = FrozenAutoencoder(in_dim, args.bottleneck_dim, args.ae_hidden_dim, args.ae_kind, args.ae_state)
    mimic = MimicOProj(
        in_dim=in_dim,
        projection_kind=args.oproj_projection_kind,
        rank=args.oproj_rank,
        state=ckpt["projector"],
        teacher_o_proj=teacher_o_proj,
        autoencoder=ae,
    ).to(device=device, dtype=torch.float32)

    hidden = int(teacher.config.hidden_size)
    student = BottleneckFFN(hidden, args.bottleneck_dim, args.ffn_hidden_dim).to(device=device, dtype=torch.float32)
    oproj_lr = args.oproj_lr if args.oproj_lr is not None else args.lr
    opt = torch.optim.AdamW(
        [
            {"params": list(student.parameters()), "lr": args.lr},
            {"params": list(mimic.parameters()), "lr": oproj_lr},
        ],
        weight_decay=1e-4,
    )

    teacher_ffn_params = sum(p.numel() for p in teacher_mlp.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    mimic_params = sum(p.numel() for p in mimic.parameters())

    history = [
        {
            "epoch": 0,
            "eval": evaluate(student, teacher_mlp, mimic, paths, eval_refs, args.batch_size, device),
        }
    ]
    print(
        "epoch=0",
        json.dumps(history[-1]["eval"], sort_keys=True),
        f"teacher_ffn_params={teacher_ffn_params} student_ffn_params={student_params} mimic_oproj_params={mimic_params}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        student.train()
        mimic.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            heads = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(start_dim=2)
            ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
            r_m = ffn_in - mimic(heads)
            target = teacher_mlp(r_m)
            pred = student(r_m)
            loss = train_loss(pred, target, loss_kind=args.loss_kind, relative_weight=args.loss_relative_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(student, teacher_mlp, mimic, paths, eval_refs, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"student_ffn": student.state_dict(), "mimic_oproj": mimic.state_dict()},
        out / "bottleneck_ffn.pt",
    )
    report = {
        "internals_capture_dir": args.internals_capture_dir,
        "compressed_oproj_pt": args.compressed_oproj_pt,
        "oproj_projection_kind": args.oproj_projection_kind,
        "oproj_rank": args.oproj_rank,
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "bottleneck_dim": args.bottleneck_dim,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "lr": args.lr,
        "oproj_lr": oproj_lr,
        "loss_kind": args.loss_kind,
        "loss_relative_weight": args.loss_relative_weight,
        "teacher_ffn_params": teacher_ffn_params,
        "student_ffn_params": student_params,
        "mimic_oproj_params": mimic_params,
        "history": history,
    }
    with (out / "bottleneck_ffn_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
