#!/usr/bin/env python3
"""Run capture (27) → mimic o_proj (31) → bottleneck FFN co-train (32) for each decoder layer.

Designed for the Ubuntu host + Dolma sample layout used in FINDINGS.md. Resumes safely:
skips capture if the expected shard count is present, skips 31 / 32 if their outputs exist.

Example::

    python scripts/33_run_bottleneck_pipeline_all_layers.py \\
      --base-dir /mnt/sdb1/dolma-v1_6-sample \\
      --first-layer 0 --last-layer 15
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-layer internals → o_proj mimic → bottleneck FFN pipeline.")
    p.add_argument("--base-dir", default="/mnt/sdb1/dolma-v1_6-sample", help="Artifact root (Dolma sample).")
    p.add_argument(
        "--selected-csv",
        default="/mnt/sdb1/dolma-v1_6-sample/selected_windows/selected_train_windows.csv",
    )
    p.add_argument(
        "--ae-state",
        default="/mnt/sdb1/dolma-v1_6-sample/layer00_head_concat_ae_half_residual_mlp/head_context_concat_autoencoder.pt",
        help="Frozen head-concat AE weights (same init as layer-0 mimic runs).",
    )
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--first-layer", type=int, default=0)
    p.add_argument("--last-layer", type=int, default=15)
    p.add_argument("--windows-per-bin", type=int, default=160, help="Must match existing 160pb captures.")
    p.add_argument("--capture-batch-size", type=int, default=2)
    p.add_argument("--samples-per-shard", type=int, default=16)
    p.add_argument("--expected-capture-shards", type=int, default=60)
    p.add_argument("--train-windows-per-bin", type=int, default=128)
    p.add_argument("--eval-windows-per-bin", type=int, default=32)
    p.add_argument("--bottleneck-dim", type=int, default=1024)
    p.add_argument("--ffn-hidden-dim", type=int, default=4096)
    p.add_argument("--ae-kind", default="decoder_residual_mlp", choices=["linear", "decoder_residual_mlp"])
    p.add_argument("--ae-hidden-dim", type=int, default=1536)
    p.add_argument("--projection-kind", default="lowrank", choices=["dense", "lowrank", "pca_lowrank"])
    p.add_argument("--projection-rank", type=int, default=768)
    p.add_argument("--oproj-epochs", type=int, default=10)
    p.add_argument("--ffn-epochs", type=int, default=5)
    p.add_argument(
        "--ffn-loss-kind",
        default="cosine",
        choices=["relative", "cosine", "both"],
        help="Passed to script 32 --loss-kind.",
    )
    p.add_argument(
        "--ffn-loss-relative-weight",
        type=float,
        default=0.25,
        help="Passed to script 32 --loss-relative-weight when loss-kind is both.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force-capture", action="store_true")
    p.add_argument("--force-oproj", action="store_true")
    p.add_argument("--force-ffn", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_step(cmd: list[str], *, dry_run: bool) -> None:
    print("CMD:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root()))


def count_capture_shards(capture_dir: Path) -> int:
    return len(list(capture_dir.glob("layer_*_internals_*.pt")))


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    py = sys.executable
    root = repo_root()

    for layer in range(args.first_layer, args.last_layer + 1):
        tag = f"layer{layer:02d}"
        capture_dir = base / f"{tag}_internals_160pb"
        oproj_dir = base / f"{tag}_oproj_mimic_ae_lowrank{args.projection_rank}_160pb"
        ffn_dir = base / f"{tag}_bottleneck_ffn_lowrank{args.projection_rank}_cotrain_{args.ffn_epochs}ep"

        print(f"\n========== {tag} ==========", flush=True)

        n_shards = count_capture_shards(capture_dir) if capture_dir.is_dir() else 0
        need_capture = args.force_capture or n_shards < args.expected_capture_shards
        if need_capture:
            capture_dir.mkdir(parents=True, exist_ok=True)
            run_step(
                [
                    py,
                    str(root / "scripts" / "27_capture_layer_internals.py"),
                    "--selected-csv",
                    args.selected_csv,
                    "--output-dir",
                    str(capture_dir),
                    "--model-name",
                    args.model_name,
                    "--target-layer",
                    str(layer),
                    "--windows-per-bin",
                    str(args.windows_per_bin),
                    "--batch-size",
                    str(args.capture_batch_size),
                    "--samples-per-shard",
                    str(args.samples_per_shard),
                    "--device",
                    "cuda",
                ],
                dry_run=args.dry_run,
            )
        else:
            print(f"skip capture ({n_shards} shards >= {args.expected_capture_shards})", flush=True)

        oproj_pt = oproj_dir / "compressed_oproj.pt"
        if args.force_oproj or not oproj_pt.is_file():
            oproj_dir.mkdir(parents=True, exist_ok=True)
            run_step(
                [
                    py,
                    str(root / "scripts" / "31_train_compressed_oproj_from_bottleneck.py"),
                    "--capture-dir",
                    str(capture_dir),
                    "--output-dir",
                    str(oproj_dir),
                    "--ae-state",
                    args.ae_state,
                    "--model-name",
                    args.model_name,
                    "--target-layer",
                    str(layer),
                    "--train-windows-per-bin",
                    str(args.train_windows_per_bin),
                    "--eval-windows-per-bin",
                    str(args.eval_windows_per_bin),
                    "--bottleneck-dim",
                    str(args.bottleneck_dim),
                    "--ae-kind",
                    args.ae_kind,
                    "--ae-hidden-dim",
                    str(args.ae_hidden_dim),
                    "--projection-kind",
                    args.projection_kind,
                    "--projection-rank",
                    str(args.projection_rank),
                    "--epochs",
                    str(args.oproj_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--seed",
                    str(args.seed),
                ],
                dry_run=args.dry_run,
            )
        else:
            print("skip o_proj (compressed_oproj.pt exists)", flush=True)

        ffn_report = ffn_dir / "bottleneck_ffn_report.json"
        if args.force_ffn or not ffn_report.is_file():
            ffn_dir.mkdir(parents=True, exist_ok=True)
            run_step(
                [
                    py,
                    str(root / "scripts" / "32_train_bottleneck_ffn_after_mimic_oproj.py"),
                    "--internals-capture-dir",
                    str(capture_dir),
                    "--compressed-oproj-pt",
                    str(oproj_pt),
                    "--oproj-projection-kind",
                    args.projection_kind,
                    "--oproj-rank",
                    str(args.projection_rank),
                    "--output-dir",
                    str(ffn_dir),
                    "--model-name",
                    args.model_name,
                    "--target-layer",
                    str(layer),
                    "--bottleneck-dim",
                    str(args.bottleneck_dim),
                    "--ffn-hidden-dim",
                    str(args.ffn_hidden_dim),
                    "--ae-state",
                    args.ae_state,
                    "--ae-kind",
                    args.ae_kind,
                    "--ae-hidden-dim",
                    str(args.ae_hidden_dim),
                    "--train-windows-per-bin",
                    str(args.train_windows_per_bin),
                    "--eval-windows-per-bin",
                    str(args.eval_windows_per_bin),
                    "--epochs",
                    str(args.ffn_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--seed",
                    str(args.seed),
                    "--loss-kind",
                    args.ffn_loss_kind,
                    "--loss-relative-weight",
                    str(args.ffn_loss_relative_weight),
                ],
                dry_run=args.dry_run,
            )
        else:
            print("skip FFN (bottleneck_ffn_report.json exists)", flush=True)

    print("\nDone all layers in range.", flush=True)


if __name__ == "__main__":
    main()
