#!/usr/bin/env python3
"""Staged low-rank MLP: compress and train gate, then up, then down (one map at a time)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from attention_compression.activations import find_transformer_layers
from attention_compression.attention_metrics import cosine_similarity_mean, relative_mse
from attention_compression.losses import LOSS_KINDS, compression_train_loss
from attention_compression.mlp_lowrank import (
    StagedHybridSwiGLU,
    enable_compressed_training,
    enable_stage_training,
    estimate_mlp_ranks_from_pca,
    export_lowrank_mlp_artifact,
    promote_branch_to_lowrank,
    stage_train_targets,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged low-rank SwiGLU (one linear map per stage).")
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=8)
    p.add_argument("--rank-gate", type=int, default=0)
    p.add_argument("--rank-up", type=int, default=0)
    p.add_argument("--rank-down", type=int, default=0)
    p.add_argument("--pca-variance-threshold", type=float, default=0.95)
    p.add_argument("--rank-cap", type=int, default=512)
    p.add_argument(
        "--train-windows-per-bin",
        type=int,
        default=0,
        help="0 = use all captured rows per bin minus eval (max data).",
    )
    p.add_argument("--eval-windows-per-bin", type=int, default=8)
    p.add_argument(
        "--pipeline",
        default="gate,up,down",
        help="Comma-separated stages to run in order (subset allowed).",
    )
    p.add_argument("--stage", default="", help="Run a single stage only (must match resume state).")
    p.add_argument("--resume-dir", type=Path, default=None, help="Load staged_mlp.pt from a prior run.")
    p.add_argument("--epochs-per-stage", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss-kind", default="both", choices=list(LOSS_KINDS))
    p.add_argument("--loss-relative-weight", type=float, default=0.25)
    p.add_argument("--mse-weight", type=float, default=0.25)
    p.add_argument("--cosine-weight", type=float, default=1.0)
    p.add_argument(
        "--loss-target",
        default="isolated",
        choices=("mlp", "linear", "isolated"),
        help="isolated/linear: train each map on its exact I/O (down uses hybrid mlp_hidden).",
    )
    p.add_argument(
        "--mlp-refresh-epochs",
        type=int,
        default=0,
        help="After each stage: full-MLP distill on all compressed branches so far.",
    )
    p.add_argument(
        "--mlp-refresh-lr",
        type=float,
        default=0.0,
        help="LR for MLP refresh (0 = use --lr).",
    )
    p.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="Optional joint MLP finetune after all stages (loss-target mlp).",
    )
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--benchmark", action="store_true", help="Run script 49 after export.")
    p.add_argument("--checkpoints", type=Path, default=Path("/mnt/sdb1/dolma-v1_6-sample"))
    return p.parse_args()


def group_rows_by_bin(paths: list[Path]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        for sample_i, row in enumerate(shard["rows"]):
            groups[row["rarity_bin"]].append((path_i, sample_i))
    return groups


def resolve_split_sizes(groups: dict[str, list], train_n: int, eval_n: int) -> tuple[int, int]:
    min_per_bin = min(len(refs) for refs in groups.values())
    if train_n <= 0:
        train_n = min_per_bin - eval_n
    if train_n < 1 or eval_n < 1:
        raise ValueError(f"Need positive train/eval split; got train={train_n} eval={eval_n}")
    if train_n + eval_n > min_per_bin:
        raise ValueError(
            f"Need {train_n + eval_n} rows per bin; min captured is {min_per_bin}"
        )
    return train_n, eval_n


def choose_train_eval(groups, train_n: int, eval_n: int, seed: int):
    train_n, eval_n = resolve_split_sizes(groups, train_n, eval_n)
    gen = torch.Generator().manual_seed(seed)
    train, eval_ = [], []
    for refs in groups.values():
        perm = torch.randperm(len(refs), generator=gen).tolist()
        train.extend(refs[i] for i in perm[:train_n])
        eval_.extend(refs[i] for i in perm[train_n : train_n + eval_n])
    return train, eval_, train_n, eval_n


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
def evaluate(
    hybrid,
    teacher_mlp,
    paths,
    eval_refs,
    batch_size,
    device,
    *,
    stage: str | None = None,
    loss_target: str = "mlp",
):
    hybrid.eval()
    acc: dict[str, float] = defaultdict(float)
    count = 0
    isolated = loss_target in ("linear", "isolated")
    for shard, ids in batch_refs(paths, eval_refs, batch_size):
        ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
        pred = hybrid(ffn_in)
        tgt = teacher_mlp(ffn_in)
        b = ffn_in.shape[0]
        for k, v in {
            "ffn_relative_mse": relative_mse(pred, tgt),
            "ffn_cosine": cosine_similarity_mean(pred, tgt),
        }.items():
            acc[k] += v * b
        if isolated and stage is not None:
            lp, lt = stage_train_targets(
                hybrid, teacher_mlp, ffn_in, stage=stage, loss_target=loss_target
            )
            acc["stage_linear_cosine"] += cosine_similarity_mean(lp, lt) * b
            acc["stage_linear_relative_mse"] += relative_mse(lp, lt) * b
        count += b
    out = {k: v / count for k, v in acc.items()}
    return out


def train_stage(
    hybrid: StagedHybridSwiGLU,
    teacher_mlp: torch.nn.Module,
    *,
    stage: str,
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    eval_refs: list[tuple[int, int]],
    args: argparse.Namespace,
    device: torch.device,
    train_mode: str = "single",
) -> tuple[dict[str, torch.Tensor] | None, list[dict]]:
    """train_mode: ``single`` (one branch) or ``compressed_all`` (all low-rank maps so far)."""
    if train_mode == "compressed_all":
        enable_compressed_training(hybrid)
    elif train_mode == "single":
        enable_stage_training(hybrid, stage.split("_")[0])
    else:
        raise ValueError(train_mode)
    lr = args.lr
    if train_mode == "compressed_all" and getattr(args, "mlp_refresh_lr", 0.0) > 0:
        lr = args.mlp_refresh_lr
    trainable = [p for p in hybrid.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    best_score = -1e18
    best_state: dict[str, torch.Tensor] | None = None
    isolated = args.loss_target in ("linear", "isolated")
    eval_stage = stage.split("_")[0] if isolated and train_mode == "single" else None
    history: list[dict] = [
        {
            "stage": stage,
            "epoch": 0,
            "eval": evaluate(
                hybrid,
                teacher_mlp,
                paths,
                eval_refs,
                args.batch_size,
                device,
                stage=eval_stage,
                loss_target=args.loss_target,
            ),
        }
    ]
    print(f"stage={stage} epoch=0 {json.dumps(history[-1]['eval'], sort_keys=True)}", flush=True)

    for epoch in range(1, args.epochs_per_stage + 1):
        hybrid.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(
            paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch
        ):
            ffn_in = shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
            pred, target = stage_train_targets(
                hybrid,
                teacher_mlp,
                ffn_in,
                stage=stage,
                loss_target=args.loss_target,
            )
            loss = compression_train_loss(
                pred,
                target,
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
        metrics = evaluate(
            hybrid,
            teacher_mlp,
            paths,
            eval_refs,
            args.batch_size,
            device,
            stage=eval_stage,
            loss_target=args.loss_target,
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": total / max(steps, 1),
            "eval": metrics,
        }
        history.append(row)
        print(
            f"stage={stage} epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"{json.dumps(metrics, sort_keys=True)}",
            flush=True,
        )
        if isolated:
            lin_cos = float(metrics.get("stage_linear_cosine", 0.0))
            lin_rel = float(metrics.get("stage_linear_relative_mse", 1.0))
            if lin_cos >= 0.2:
                sc = -(lin_rel + 80.0 * (1.0 - lin_cos))
                if sc > best_score:
                    best_score = sc
                    best_state = {k: v.detach().cpu() for k, v in hybrid.state_dict().items()}
        else:
            ffn_cos = float(metrics["ffn_cosine"])
            if ffn_cos >= 0.2:
                sc = -(float(metrics["ffn_relative_mse"]) + 80.0 * (1.0 - ffn_cos))
                if sc > best_score:
                    best_score = sc
                    best_state = {k: v.detach().cpu() for k, v in hybrid.state_dict().items()}

    if best_state is not None:
        hybrid.load_state_dict(best_state, strict=True)
    for p in hybrid.parameters():
        p.requires_grad_(False)
    hybrid.eval()
    return best_state, history


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(args.internals_capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.internals_capture_dir)

    pipeline = [s.strip() for s in args.pipeline.split(",") if s.strip()]
    if args.stage:
        pipeline = [args.stage.strip()]

    groups = group_rows_by_bin(paths)
    train_refs, eval_refs, train_n, eval_n = choose_train_eval(
        groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed
    )
    print(
        f"split train={len(train_refs)} eval={len(eval_refs)} "
        f"({train_n}+ {eval_n} per bin, bins={len(groups)})",
        flush=True,
    )

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.eval().to(device)
    _path, layers = find_transformer_layers(teacher)
    teacher_mlp = layers[args.target_layer].mlp.to(device=device, dtype=torch.float32)
    for p in teacher_mlp.parameters():
        p.requires_grad_(False)

    overrides = {"gate": args.rank_gate, "up": args.rank_up, "down": args.rank_down}
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

    stages_done: list[str] = []
    stage_histories: dict[str, list] = {}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.resume_dir is not None:
        ckpt = torch.load(args.resume_dir / "staged_mlp.pt", map_location="cpu", weights_only=False)
        stages_done = list(ckpt.get("stages_completed", []))
        hybrid = StagedHybridSwiGLU(
            teacher_mlp,
            ranks=ranks,
            compressed=tuple(stages_done),
        ).to(device=device, dtype=torch.float32)
        hybrid.load_state_dict(ckpt["hybrid"], strict=True)
        print(f"Resumed from {args.resume_dir} stages_completed={stages_done}", flush=True)
    else:
        hybrid = StagedHybridSwiGLU(
            teacher_mlp, ranks=ranks, compressed=()
        ).to(device=device, dtype=torch.float32)

    for stage in pipeline:
        if stage in stages_done:
            print(f"Skipping stage {stage} (already in checkpoint)", flush=True)
            continue
        promote_branch_to_lowrank(
            hybrid,
            teacher_mlp,
            stage,
            ranks[stage],
            device=device,
            paths=paths,
            train_refs=train_refs,
            init_pca=True,
        )
        _best, hist = train_stage(
            hybrid,
            teacher_mlp,
            stage=stage,
            paths=paths,
            train_refs=train_refs,
            eval_refs=eval_refs,
            args=args,
            device=device,
        )
        if stage not in stages_done:
            stages_done.append(stage)
        stage_histories[stage] = hist

        torch.save(
            {
                "hybrid": {k: v.detach().cpu() for k, v in hybrid.state_dict().items()},
                "stages_completed": stages_done,
                "ranks": ranks,
            },
            out / "staged_mlp.pt",
        )
        (out / f"stage_{stage}_history.json").write_text(
            json.dumps(hist, indent=2), encoding="utf-8"
        )

        if args.mlp_refresh_epochs > 0:
            refresh_key = f"{stage}_mlp_refresh"
            print(
                f"mlp_refresh after {stage} epochs={args.mlp_refresh_epochs} "
                f"compressed={sorted(hybrid.compressed)}",
                flush=True,
            )
            refresh_lr = args.mlp_refresh_lr if args.mlp_refresh_lr > 0 else args.lr
            refresh_args = argparse.Namespace(
                **{
                    **vars(args),
                    "loss_target": "mlp",
                    "epochs_per_stage": args.mlp_refresh_epochs,
                    "mlp_refresh_lr": refresh_lr,
                }
            )
            _rbest, refresh_hist = train_stage(
                hybrid,
                teacher_mlp,
                stage=refresh_key,
                paths=paths,
                train_refs=train_refs,
                eval_refs=eval_refs,
                args=refresh_args,
                device=device,
                train_mode="compressed_all",
            )
            stage_histories[refresh_key] = refresh_hist
            torch.save(
                {
                    "hybrid": {k: v.detach().cpu() for k, v in hybrid.state_dict().items()},
                    "stages_completed": stages_done,
                    "ranks": ranks,
                },
                out / "staged_mlp.pt",
            )
            (out / f"stage_{refresh_key}_history.json").write_text(
                json.dumps(refresh_hist, indent=2), encoding="utf-8"
            )

    if args.finetune_epochs > 0 and hybrid.compressed >= frozenset({"gate", "up", "down"}):
        print(f"finetune epochs={args.finetune_epochs} loss_target=mlp", flush=True)
        finetune_args = argparse.Namespace(
            **{**vars(args), "epochs_per_stage": args.finetune_epochs, "loss_target": "mlp"}
        )
        _best, finetune_hist = train_stage(
            hybrid,
            teacher_mlp,
            stage="finetune_mlp",
            paths=paths,
            train_refs=train_refs,
            eval_refs=eval_refs,
            args=finetune_args,
            device=device,
            train_mode="compressed_all",
        )
        stage_histories["finetune_mlp"] = finetune_hist

    final_eval = evaluate(hybrid, teacher_mlp, paths, eval_refs, args.batch_size, device)
    dense_p = sum(p.numel() for p in teacher_mlp.parameters())
    student = None
    student_p = None
    if hybrid.compressed >= frozenset({"gate", "up", "down"}):
        from attention_compression.mlp_lowrank import staged_hybrid_to_lowrank

        student = staged_hybrid_to_lowrank(hybrid)
        student_p = sum(p.numel() for p in student.parameters())

    report = {
        "training": "staged_lowrank_mlp",
        "internals_capture_dir": str(args.internals_capture_dir),
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "pipeline": pipeline,
        "stages_completed": stages_done,
        "rank_gate": ranks["gate"],
        "rank_up": ranks["up"],
        "rank_down": ranks["down"],
        "rank_report": rank_report,
        "loss_kind": args.loss_kind,
        "loss_relative_weight": args.loss_relative_weight,
        "loss_target": args.loss_target,
        "epochs_per_stage": args.epochs_per_stage,
        "mlp_refresh_epochs": args.mlp_refresh_epochs,
        "mlp_refresh_lr": args.mlp_refresh_lr,
        "finetune_epochs": args.finetune_epochs,
        "train_windows_per_bin": train_n,
        "eval_windows_per_bin": eval_n,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "final_eval": final_eval,
        "stage_histories": {k: v[-1] if v else {} for k, v in stage_histories.items()},
        "teacher_mlp_params": dense_p,
        "student_mlp_params": student_p,
        "param_ratio": (student_p / dense_p) if student_p else None,
    }
    (out / "staged_mlp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if student is not None:
        export_dir = out / "lowrank_export"
        export_lowrank_mlp_artifact(hybrid, export_dir, report=report)
        print(f"Exported {export_dir / 'lowrank_mlp.pt'}", flush=True)
        if args.benchmark:
            bench_script = Path(__file__).resolve().parent / "49_lowrank_mlp_benchmark.py"
            subprocess.run(
                [
                    sys.executable,
                    str(bench_script),
                    "--checkpoints",
                    str(args.checkpoints),
                    "--artifact-dir",
                    str(export_dir),
                    "--layer",
                    str(args.target_layer),
                    "--output-json",
                    str(out / "lowrank_mlp_benchmark.json"),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
