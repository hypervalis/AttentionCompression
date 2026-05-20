#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_compression.corpus import inspect_jsonl_corpus, write_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Dolma JSONL corpus files.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--file-glob", default="*.json")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_jsonl_corpus(
        data_dir=args.data_dir,
        file_glob=args.file_glob,
        text_field=args.text_field,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_inspection(args.output, inspection)
    print(json.dumps(asdict(inspection), indent=2))


if __name__ == "__main__":
    main()
