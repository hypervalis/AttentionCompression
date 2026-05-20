#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_compression.diagnostics import run_sampling_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize rarity-selected windows.")
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-per-split", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_sampling_diagnostics(
        scores_path=args.scores_path,
        selected_dir=args.selected_dir,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        validate_per_split=args.validate_per_split,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
