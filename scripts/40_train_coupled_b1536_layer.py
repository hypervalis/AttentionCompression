#!/usr/bin/env python3
"""Train coupled b=1536 bottlenecks for o_proj and FFN on layer internals (script 27)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from attention_compression.activations import find_transformer_layers
from attention_compression.attention_metrics import cosine_similarity_mean, relative_mse
from attention_compression.losses import LOSS_KINDS, compression_train_loss
from attention_compression.bottleneck_layer import (
    BottleneckMap,
    BottleneckOProj,
    init_bottleneck_map_identityish,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coupled o_proj + FFN at bottleneck width b (default 1536).")
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--bottleneck-dim", type=int, default=1536)
    p.add_argument(
        "--ffn-hidden-dim",
        type=int,
        default=0,
        help="Student FFN hidden (default: round(4096 * b/2048)).",
    )
    p.add_argument("--train-windows-per-bin", type=int, default=128)
    p.add_argument("--eval-windows-per-bin", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--oproj-lr", type=float, default=None)
    p.add_argument("--loss-kind", default="both", choices=list(LOSS_KINDS))
    p.add_argument("--loss-relative-weight", type=float, default=0.25)
    p.add_argument(
        "--cosine-weight",
        type=float,
        default=1.0,
        help="Multiplier on cosine when --loss-kind relative_plus_cosine.",
    )
    p.add_argument("--oproj-loss-weight", type=float, default=0.5, help="Weight on o_proj vs FFN loss.")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def group_rows_by_bin(paths: list[Path]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu", weights_only=False)
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


def batch_refs(paths, refs, batch_size, *, shuffle=False, seed=0):
    by_path: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in refs:
        by_path[path_i].append(sample_i)
    items = list(by_path.items())
    gen = torch.Generator().manual_seed(seed)
    if shuffle:
        order = torch.randperm(len(items), generator=gen).tolist()
        items = [items[i] for i in order]
    for path_i, ids in items:
        if shuffle and ids:
            perm = torch.randperm(len(ids), generator=gen).tolist()
            ids = [ids[i] for i in perm]
        shard = torch.load(paths[path_i], map_location="cpu", weights_only=False)
        for start in range(0, len(ids), batch_size):
            yield shard, ids[start : start + batch_size]


def train_loss(pred, target, *, loss_kind: str, relative_weight: float) -> torch.Tensor:
    if loss_kind == "relative":
        return torch.mean((pred.float() - target.float()) ** 2) / torch.var(target.float()).clamp_min(1e-6)
    if loss_kind == "cosine":
        pf = pred.reshape(-1, pred.shape[-1]).float()
        tf = target.reshape(-1, target.shape[-1]).float()
        return torch.mean(1.0 - torch.nn.functional.cosine_similarity(pf, tf, dim=-1))
    w = float(relative_weight)
    return w * train_loss(pred, target, loss_kind="relative", relative_weight=0) + (1.0 - w) * train_loss(
        pred, target, loss_kind="cosine", relative_weight=0
    )


@torch.no_grad()
def evaluate(o_proj, student, teacher_o, teacher_mlp, paths, eval_refs, batch_size, device):
    o_proj.eval()
    student.eval()
    acc: dict[str, float] = defaultdict(float)
    count = 0
    for shard, ids in batch_refs(paths, eval_refs, batch_size):
        heads = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(start_dim=2)
        ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
        o_pred = o_proj(heads)
        o_tgt = teacher_o(heads)
        m_pred = student(ffn_in)
        m_tgt = teacher_mlp(ffn_in)
        b = heads.shape[0]
        metrics = {
            "oproj_relative_mse": relative_mse(o_pred, o_tgt),
            "oproj_cosine": cosine_similarity_mean(o_pred, o_tgt),
            "ffn_relative_mse": relative_mse(m_pred, m_tgt),
            "ffn_cosine": cosine_similarity_mean(m_pred, m_tgt),
        }
        for k, v in metrics.items():
            acc[k] += v * b
        count += b
    return {k: v / count for k, v in acc.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(Path(args.internals_capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.internals_capture_dir)
    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(
        groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed
    )
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    in_dim = int(first["head_contexts"].shape[2] * first["head_contexts"].shape[3])
    hidden = in_dim
    b = args.bottleneck_dim
    ffn_hidden = args.ffn_hidden_dim or int(round(4096 * b / 2048))

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.eval().to(device)
    _path, layers = find_transformer_layers(teacher)
    layer = layers[args.target_layer]
    teacher_o = layer.self_attn.o_proj.to(device=device, dtype=torch.float32)
    teacher_mlp = layer.mlp.to(device=device, dtype=torch.float32)
    for p in teacher_mlp.parameters():
        p.requires_grad_(False)
    for p in teacher_o.parameters():
        p.requires_grad_(False)

    o_proj = BottleneckOProj.from_teacher_linear(teacher_o, b).to(device=device, dtype=torch.float32)
    student = BottleneckMap(hidden, b, hidden, hidden=ffn_hidden).to(device=device, dtype=torch.float32)
    init_bottleneck_map_identityish(student)

    o_lr = args.oproj_lr if args.oproj_lr is not None else args.lr
    opt = torch.optim.AdamW(
        [
            {"params": student.parameters(), "lr": args.lr},
            {"params": o_proj.parameters(), "lr": o_lr},
        ],
        weight_decay=1e-4,
    )
    ow = float(args.oproj_loss_weight)
    fw = 1.0 - ow

    def _score(metrics: dict[str, float]) -> float:
        """Higher is better. Ignore pre-train epoch-0 FFN (identity init has ~0 cosine)."""
        ffn_cos = float(metrics["ffn_cosine"])
        if ffn_cos < 0.2:
            return -1e9
        gamma = 80.0
        ffn_penalty = float(metrics["ffn_relative_mse"]) + gamma * (1.0 - ffn_cos)
        o_penalty = float(metrics["oproj_relative_mse"]) + gamma * (1.0 - float(metrics["oproj_cosine"]))
        return -(ffn_penalty + o_penalty)

    best_score = -1e18
    best_state: dict[str, dict] | None = None

    history = [
        {
            "epoch": 0,
            "eval": evaluate(o_proj, student, teacher_o, teacher_mlp, paths, eval_refs, args.batch_size, device),
        }
    ]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        o_proj.train()
        student.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            heads = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(start_dim=2)
            ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                o_tgt = teacher_o(heads)
                m_tgt = teacher_mlp(ffn_in)
            o_pred = o_proj(heads)
            m_pred = student(ffn_in)
            loss = ow * compression_train_loss(
                o_pred,
                o_tgt,
                loss_kind=args.loss_kind,
                relative_weight=args.loss_relative_weight,
                cosine_weight=args.cosine_weight,
            ) + fw * compression_train_loss(
                m_pred,
                m_tgt,
                loss_kind=args.loss_kind,
                relative_weight=args.loss_relative_weight,
                cosine_weight=args.cosine_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(o_proj, student, teacher_o, teacher_mlp, paths, eval_refs, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)
        sc = _score(metrics)
        if sc > best_score:
            best_score = sc
            best_state = {
                "o_proj": {k: v.detach().cpu() for k, v in o_proj.state_dict().items()},
                "student_ffn": {k: v.detach().cpu() for k, v in student.state_dict().items()},
            }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {
            "o_proj": {k: v.detach().cpu() for k, v in o_proj.state_dict().items()},
            "student_ffn": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        }
    torch.save(best_state, out / "coupled_b1536.pt")

    dense_o = sum(p.numel() for p in teacher_o.parameters())
    dense_m = sum(p.numel() for p in teacher_mlp.parameters())
    report = {
        "internals_capture_dir": str(args.internals_capture_dir),
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "hidden_size": hidden,
        "bottleneck_dim": b,
        "ffn_hidden_dim": ffn_hidden,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "loss_kind": args.loss_kind,
        "loss_relative_weight": args.loss_relative_weight,
        "cosine_weight": args.cosine_weight,
        "oproj_loss_weight": ow,
        "teacher_o_proj_params": dense_o,
        "teacher_ffn_params": dense_m,
        "o_proj_params": sum(p.numel() for p in o_proj.parameters()),
        "student_ffn_params": sum(p.numel() for p in student.parameters()),
        "best_eval_score": best_score,
        "history": history,
    }
    (out / "coupled_b1536_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
