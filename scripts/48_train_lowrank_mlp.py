#!/usr/bin/env python3
"""Train low-rank SwiGLU MLP (gate/up/down) with supervised output-PCA init, like Q/K."""
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
from attention_compression.mlp_lowrank import (
    build_lowrank_mlp_from_teacher,
    estimate_mlp_ranks_from_pca,
    init_lowrank_mlp_from_supervised_pca,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-rank MLP distillation with supervised PCA init.")
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=8)
    p.add_argument("--rank-gate", type=int, default=0, help="0 = PCA estimate (capped).")
    p.add_argument("--rank-up", type=int, default=0)
    p.add_argument("--rank-down", type=int, default=0)
    p.add_argument("--pca-variance-threshold", type=float, default=0.95)
    p.add_argument("--rank-cap", type=int, default=512, help="Max rank per map from PCA estimate.")
    p.add_argument(
        "--train-windows-per-bin",
        type=int,
        default=144,
        help="Max 144 if captures have 160/bin (leave 16 for eval).",
    )
    p.add_argument("--eval-windows-per-bin", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss-kind", default="mse_cosine", choices=list(LOSS_KINDS))
    p.add_argument("--loss-relative-weight", type=float, default=0.1)
    p.add_argument(
        "--mse-weight",
        type=float,
        default=0.25,
        help="Weight on plain MSE when --loss-kind mse_cosine (cosine weight is 1 minus this).",
    )
    p.add_argument("--cosine-weight", type=float, default=1.0)
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


@torch.no_grad()
def evaluate(student, teacher_mlp, paths, eval_refs, batch_size, device):
    student.eval()
    acc: dict[str, float] = defaultdict(float)
    count = 0
    for shard, ids in batch_refs(paths, eval_refs, batch_size):
        ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
        pred = student(ffn_in)
        tgt = teacher_mlp(ffn_in)
        b = ffn_in.shape[0]
        for k, v in {
            "ffn_relative_mse": relative_mse(pred, tgt),
            "ffn_cosine": cosine_similarity_mean(pred, tgt),
        }.items():
            acc[k] += v * b
        count += b
    return {k: v / count for k, v in acc.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(args.internals_capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.internals_capture_dir)

    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(
        groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed
    )

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.eval().to(device)
    _path, layers = find_transformer_layers(teacher)
    teacher_mlp = layers[args.target_layer].mlp.to(device=device, dtype=torch.float32)
    for p in teacher_mlp.parameters():
        p.requires_grad_(False)

    overrides = {
        "gate": args.rank_gate,
        "up": args.rank_up,
        "down": args.rank_down,
    }
    rank_cap = args.rank_cap if args.rank_cap > 0 else None
    ranks, rank_report = estimate_mlp_ranks_from_pca(
        teacher_mlp,
        paths,
        train_refs,
        device=device,
        variance_threshold=args.pca_variance_threshold,
        ranks=overrides,
        rank_cap=rank_cap,
    )
    print("PCA rank estimate:", json.dumps(rank_report, indent=2), flush=True)

    student = build_lowrank_mlp_from_teacher(
        teacher_mlp,
        rank_gate=ranks["gate"],
        rank_up=ranks["up"],
        rank_down=ranks["down"],
    ).to(device=device, dtype=torch.float32)

    init_report = init_lowrank_mlp_from_supervised_pca(
        student,
        teacher_mlp,
        paths,
        train_refs,
        device=device,
        variance_threshold=args.pca_variance_threshold,
        ranks=ranks,
        rank_cap=rank_cap,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = -1e18
    best_state: dict[str, dict] | None = None
    history = [{"epoch": 0, "eval": evaluate(student, teacher_mlp, paths, eval_refs, args.batch_size, device)}]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                tgt = teacher_mlp(ffn_in)
            pred = student(ffn_in)
            loss = compression_train_loss(
                pred,
                tgt,
                loss_kind=args.loss_kind,
                relative_weight=args.loss_relative_weight,
                cosine_weight=args.cosine_weight,
                mse_weight=args.mse_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(student, teacher_mlp, paths, eval_refs, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)
        ffn_cos = float(metrics["ffn_cosine"])
        if ffn_cos >= 0.2:
            sc = -(float(metrics["ffn_relative_mse"]) + 80.0 * (1.0 - ffn_cos))
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save({"student": best_state}, out / "lowrank_mlp.pt")

    dense_p = sum(p.numel() for p in teacher_mlp.parameters())
    student_p = sum(p.numel() for p in student.parameters())
    report = {
        "internals_capture_dir": str(args.internals_capture_dir),
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "rank_gate": ranks["gate"],
        "rank_up": ranks["up"],
        "rank_down": ranks["down"],
        "pca_variance_threshold": args.pca_variance_threshold,
        "rank_cap": rank_cap,
        "rank_report": rank_report,
        "init_report": init_report,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "loss_kind": args.loss_kind,
        "mse_weight": args.mse_weight,
        "teacher_mlp_params": dense_p,
        "student_mlp_params": student_p,
        "param_ratio": student_p / dense_p,
        "best_eval_score": best_score,
        "history": history,
    }
    (out / "lowrank_mlp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
