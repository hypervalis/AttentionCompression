#!/usr/bin/env python3
"""Run a minimal end-to-end smoke of the main compression path.

Steps:
  1) Write a tiny token shard + selected CSV (one split covers all ``RARITY_BIN_NAMES``).
  2) ``07_activation_smoke_test.py`` — forward pass + layer hook.
  3) ``09_capture_head_activations.py`` — one-head activation shards.
  4) ``16_train_qk_dense_v.py`` — one epoch, one train/eval window per bin.

Requires optional deps: ``torch``, ``transformers`` (same as the experiment scripts).

Example::

  python3 scripts/17_end_to_end_smoke.py --output-dir /tmp/ac_smoke

"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_args() -> argparse.Namespace:
    root = _REPO
    parser = argparse.ArgumentParser(description="End-to-end smoke: fixture → 07 → 09 → 16.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "e2e_smoke",
        help="Working directory (created; cleared unless --no-clean).",
    )
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--head-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64, help="Window length in tokens (keep small for speed).")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete output-dir before running.")
    parser.add_argument(
        "--with-pytest",
        action="store_true",
        help="Also run a small subset of unit tests (requires pytest).",
    )
    return parser.parse_args()


def write_fixture(*, out_dir: Path, seq_len: int) -> Path:
    import numpy as np

    from attention_compression.windows import RARITY_BIN_NAMES

    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / "smoke_tokens.npy"
    windows_per_bin = 2
    n_bins = len(RARITY_BIN_NAMES)
    n_rows = n_bins * windows_per_bin
    total_tokens = n_rows * seq_len
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, 32000, size=(total_tokens,), dtype=np.int64)
    np.save(shard, tokens)

    csv_path = out_dir / "smoke_selected.csv"
    fields = ["window_id", "rarity_bin", "path", "start", "seq_len"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        wid = 0
        for bin_name in RARITY_BIN_NAMES:
            name = str(bin_name)
            for _ in range(windows_per_bin):
                start = wid * seq_len
                w.writerow(
                    {
                        "window_id": wid,
                        "rarity_bin": name,
                        "path": str(shard.resolve()),
                        "start": start,
                        "seq_len": seq_len,
                    }
                )
                wid += 1
    return csv_path


def run_step(cmd: list[str], *, cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing torch (and you need transformers for OLMo). "
            "Example: pip install torch transformers; or use the same venv as your sweeps."
        ) from e

    args = parse_args()
    repo = _REPO
    work: Path = args.output_dir.resolve()
    if not args.no_clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    csv_path = write_fixture(out_dir=work, seq_len=args.seq_len)
    py = sys.executable

    run_step(
        [
            py,
            str(repo / "scripts" / "07_activation_smoke_test.py"),
            "--selected-csv",
            str(csv_path),
            "--model-name",
            args.model_name,
            "--target-layer",
            str(args.target_layer),
            "--windows-per-bin",
            "2",
            "--max-batch-size",
            "2",
            "--output",
            str(work / "smoke_07_activation.json"),
        ],
        cwd=repo,
    )

    cap_dir = work / "capture"
    run_step(
        [
            py,
            str(repo / "scripts" / "09_capture_head_activations.py"),
            "--selected-csv",
            str(csv_path),
            "--output-dir",
            str(cap_dir),
            "--model-name",
            args.model_name,
            "--target-layer",
            str(args.target_layer),
            "--head-index",
            str(args.head_index),
            "--windows-per-bin",
            "2",
            "--batch-size",
            "2",
            "--samples-per-shard",
            "8",
            "--device",
            "auto",
        ],
        cwd=repo,
    )

    train_out = work / "qk_dense_v_smoke"
    run_step(
        [
            py,
            str(repo / "scripts" / "16_train_qk_dense_v.py"),
            "--capture-dir",
            str(cap_dir),
            "--output-dir",
            str(train_out),
            "--model-name",
            args.model_name,
            "--target-layer",
            str(args.target_layer),
            "--head-index",
            str(args.head_index),
            "--train-windows-per-bin",
            "1",
            "--eval-windows-per-bin",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "1",
        ],
        cwd=repo,
    )

    report = train_out / "qk_dense_v_report.json"
    if not report.is_file():
        raise SystemExit(f"Missing report: {report}")
    print("OK:", report, flush=True)

    if args.with_pytest:
        tests = [
            "tests/test_activations.py",
            "tests/test_joint_qkv.py",
            "tests/test_attention_metrics.py",
        ]
        run_step([py, "-m", "pytest", "-q", *tests], cwd=repo)


if __name__ == "__main__":
    main()
