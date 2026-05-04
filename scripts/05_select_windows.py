#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_compression.config import load_config
from attention_compression.selection import select_train_eval_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select shard-aware train/eval windows by rarity bin.")
    parser.add_argument("--scores-path", required=True, help="Path to window_scores.npz.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/experiment_defaults.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--per-shard-cap", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary = select_train_eval_windows(
        scores_path=args.scores_path,
        output_dir=args.output_dir,
        train_targets=config["train_windows_per_bin"],
        eval_targets=config["eval_windows_per_bin"],
        seed=args.seed,
        per_shard_cap=args.per_shard_cap,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
