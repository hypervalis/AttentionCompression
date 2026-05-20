#!/usr/bin/env python3
"""Train PCA-block factorized FFN (multiple subspace MLPs) vs frozen teacher MLP."""
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
from attention_compression.block_factorized_ffn import (
    MERGE_MODES,
    BlockFactorizedFFN,
    equal_pca_blocks,
    fit_ffn_input_pca,
)
from attention_compression.mlp_operator import load_operator_basis
from attention_compression.losses import LOSS_KINDS, compression_train_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Block factorized FFN on PCA subspaces of ffn_input.")
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument(
        "--merge-mode",
        default="additive",
        choices=list(MERGE_MODES),
        help="additive: sum V_j MLP_j(V_j^T x); concat: Linear(cat_j MLP_j(...)).",
    )
    p.add_argument("--pca-max-tokens", type=int, default=200_000)
    p.add_argument(
        "--basis",
        default="input-pca",
        choices=["input-pca", "operator-jacobian", "linear-rrr"],
        help="Subspace basis: input PCA or operator artifact (45).",
    )
    p.add_argument(
        "--operator-artifact",
        type=Path,
        default=None,
        help="mlp_operator_svd.pt from script 45 when --basis is not input-pca.",
    )
    p.add_argument("--train-windows-per-bin", type=int, default=128)
    p.add_argument("--eval-windows-per-bin", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss-kind", default="both", choices=list(LOSS_KINDS))
    p.add_argument("--loss-relative-weight", type=float, default=0.1)
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


def _score(metrics: dict[str, float]) -> float:
    ffn_cos = float(metrics["ffn_cosine"])
    if ffn_cos < 0.2:
        return -1e9
    gamma = 80.0
    return -(float(metrics["ffn_relative_mse"]) + gamma * (1.0 - ffn_cos))


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
    hidden = int(first["ffn_input"].shape[-1])

    basis_kind = args.basis.replace("-", "_")
    if args.basis == "input-pca":
        print("fitting PCA on ffn_input...", flush=True)
        mean, basis, evals = fit_ffn_input_pca(paths, max_tokens=args.pca_max_tokens)
    else:
        if args.operator_artifact is None:
            raise ValueError("--operator-artifact required when --basis is not input-pca")
        print(f"loading basis from {args.operator_artifact} ({basis_kind})...", flush=True)
        mean, basis, evals = load_operator_basis(args.operator_artifact, kind=basis_kind)

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.eval().to(device)
    _path, layers = find_transformer_layers(teacher)
    layer = layers[args.target_layer]
    teacher_mlp = layer.mlp.to(device=device, dtype=torch.float32)
    for p in teacher_mlp.parameters():
        p.requires_grad_(False)

    inter = int(getattr(teacher.config, "intermediate_size", 4096))
    student = BlockFactorizedFFN(
        dim=hidden,
        block_dims=equal_pca_blocks(hidden, args.num_blocks),
        mean=mean,
        basis=basis,
        merge_mode=args.merge_mode,
        teacher_intermediate=inter,
    ).to(device=device, dtype=torch.float32)

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
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(student, teacher_mlp, paths, eval_refs, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)
        sc = _score(metrics)
        if sc > best_score:
            best_score = sc
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    block_dims = student.block_dims
    block_hidden = [b.hidden for b in student.blocks]
    torch.save(
        {
            "student": best_state,
            "mean": mean.cpu(),
            "basis": basis.cpu(),
            "eigenvalues": evals.cpu(),
        },
        out / "block_factorized_ffn.pt",
    )

    dense_m = sum(p.numel() for p in teacher_mlp.parameters())
    report = {
        "internals_capture_dir": str(args.internals_capture_dir),
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "hidden_size": hidden,
        "num_blocks": args.num_blocks,
        "block_dims": block_dims,
        "block_hidden_dims": block_hidden,
        "merge_mode": args.merge_mode,
        "basis": args.basis,
        "operator_artifact": str(args.operator_artifact) if args.operator_artifact else None,
        "pca_max_tokens": args.pca_max_tokens,
        "basis_top_eigenvalues": evals[:16].tolist(),
        "teacher_intermediate": inter,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "loss_kind": args.loss_kind,
        "loss_relative_weight": args.loss_relative_weight,
        "cosine_weight": args.cosine_weight,
        "teacher_ffn_params": dense_m,
        "student_ffn_params": sum(p.numel() for p in student.parameters()),
        "best_eval_score": best_score,
        "top_eigenvalues": evals[:16].tolist(),
        "history": history,
    }
    (out / "block_factorized_ffn_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
