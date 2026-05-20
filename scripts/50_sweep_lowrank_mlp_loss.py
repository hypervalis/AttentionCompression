#!/usr/bin/env python3
"""Sweep low-rank MLP training losses (script 48): mse_cosine weights and both(w)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep script-48 loss kinds / weights on one layer.")
    p.add_argument("--output-parent", type=Path, required=True)
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--checkpoints", type=Path, default=Path("/mnt/sdb1/dolma-v1_6-sample"))
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=8)
    p.add_argument("--rank-cap", type=int, default=512)
    p.add_argument("--rank-gate", type=int, default=512)
    p.add_argument("--rank-up", type=int, default=512)
    p.add_argument("--rank-down", type=int, default=512)
    p.add_argument("--train-windows-per-bin", type=int, default=144)
    p.add_argument("--eval-windows-per-bin", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--benchmark-windows-per-bin", type=int, default=6)
    p.add_argument(
        "--mse-weights",
        default="0,0.1,0.25,0.5,0.75",
        help="Comma-separated --mse-weight for mse_cosine runs.",
    )
    p.add_argument(
        "--both-weights",
        default="0.1,0.25,0.5",
        help="Comma-separated --loss-relative-weight for both runs.",
    )
    p.add_argument("--skip-mse-cosine", action="store_true")
    p.add_argument("--skip-both", action="store_true")
    p.add_argument("--skip-baselines", action="store_true", help="Skip relative and cosine-only.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root = repo_root()
    src_pp = str(root / "src")
    old_pp = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_pp if not old_pp else f"{src_pp}{os.pathsep}{old_pp}"
    return env


def train_cmd(args: argparse.Namespace, out_dir: Path, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        str(repo_root() / "scripts" / "48_train_lowrank_mlp.py"),
        "--internals-capture-dir",
        str(args.internals_capture_dir),
        "--output-dir",
        str(out_dir),
        "--model-name",
        args.model_name,
        "--target-layer",
        str(args.target_layer),
        "--rank-cap",
        str(args.rank_cap),
        "--rank-gate",
        str(args.rank_gate),
        "--rank-up",
        str(args.rank_up),
        "--rank-down",
        str(args.rank_down),
        "--train-windows-per-bin",
        str(args.train_windows_per_bin),
        "--eval-windows-per-bin",
        str(args.eval_windows_per_bin),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        *extra,
    ]


def benchmark_cmd(args: argparse.Namespace, artifact_dir: Path, bench_json: Path) -> list[str]:
    return [
        sys.executable,
        str(repo_root() / "scripts" / "49_lowrank_mlp_benchmark.py"),
        "--checkpoints",
        str(args.checkpoints),
        "--artifact-dir",
        str(artifact_dir),
        "--layer",
        str(args.target_layer),
        "--windows-per-bin",
        str(args.benchmark_windows_per_bin),
        "--output-json",
        str(bench_json),
    ]


def best_eval_from_report(report: dict) -> dict:
    best_cos = -1.0
    best: dict | None = None
    for row in report.get("history", []):
        if int(row.get("epoch", 0)) == 0:
            continue
        ev = row.get("eval") or {}
        cos = float(ev.get("ffn_cosine", 0.0))
        if cos > best_cos:
            best_cos = cos
            best = ev
    return best or {}


def run_one(args: argparse.Namespace, name: str, extra: list[str], results: list[dict]) -> None:
    parent = Path(args.output_parent)
    out = parent / name
    bench_json = parent / f"{name}_benchmark.json"
    train = train_cmd(args, out, extra)
    print("TRAIN", " ".join(train), flush=True)
    if args.dry_run:
        results.append({"name": name, "dry_run": True, "train_cmd": train})
        return
    subprocess.run(train, check=True, cwd=str(repo_root()), env=_subprocess_env())
    report = json.loads((out / "lowrank_mlp_report.json").read_text(encoding="utf-8"))
    bench = benchmark_cmd(args, out, bench_json)
    print("BENCH", " ".join(bench), flush=True)
    subprocess.run(bench, check=True, cwd=str(repo_root()), env=_subprocess_env())
    bench_data = json.loads(bench_json.read_text(encoding="utf-8"))
    last = report["history"][-1]
    best = best_eval_from_report(report)
    ppl_ratio = float(bench_data["lowrank_mlp"]["perplexity_ratio_vs_baseline"])
    results.append(
        {
            "name": name,
            "output_dir": str(out),
            "loss_kind": report.get("loss_kind"),
            "mse_weight": report.get("mse_weight"),
            "loss_relative_weight": report.get("loss_relative_weight"),
            "last_epoch": last.get("epoch"),
            "last_eval": last.get("eval"),
            "best_eval": best,
            "best_ffn_cosine": float(best.get("ffn_cosine", 0.0)),
            "perplexity_ratio": ppl_ratio,
            "perplexity": float(bench_data["lowrank_mlp"]["perplexity"]),
        }
    )


def main() -> None:
    args = parse_args()
    mse_weights = [float(x.strip()) for x in args.mse_weights.split(",") if x.strip()]
    both_weights = [float(x.strip()) for x in args.both_weights.split(",") if x.strip()]
    for w in mse_weights + both_weights:
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"Weight must be in [0, 1]: {w}")

    parent = Path(args.output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    if not args.skip_baselines:
        run_one(args, "loss_cosine", ["--loss-kind", "cosine"], results)
        run_one(args, "loss_relative", ["--loss-kind", "relative"], results)

    if not args.skip_mse_cosine:
        for w in mse_weights:
            wtag = f"{w:g}".replace(".", "p")
            run_one(
                args,
                f"mse_cosine_w{wtag}",
                ["--loss-kind", "mse_cosine", "--mse-weight", str(w)],
                results,
            )

    if not args.skip_both:
        for w in both_weights:
            wtag = f"{w:g}".replace(".", "p")
            run_one(
                args,
                f"both_w{wtag}",
                ["--loss-kind", "both", "--loss-relative-weight", str(w)],
                results,
            )

    ranked = sorted(results, key=lambda r: (r.get("perplexity_ratio", 1e9), -r.get("best_ffn_cosine", 0)))
    summary = {
        "target_layer": args.target_layer,
        "internals_capture_dir": str(args.internals_capture_dir),
        "train_windows_per_bin": args.train_windows_per_bin,
        "eval_windows_per_bin": args.eval_windows_per_bin,
        "rank_cap": args.rank_cap,
        "runs": results,
        "by_ppl_ratio": [r["name"] for r in ranked],
        "best_ppl": ranked[0] if ranked else None,
    }
    out_path = parent / "lowrank_mlp_loss_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
