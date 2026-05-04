#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_compression.windows import score_token_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score token windows by empirical transition rarity.")
    parser.add_argument("--tokens-dir", required=True, help="Directory containing flat .npy token shards.")
    parser.add_argument("--counts-dir", required=True, help="Directory containing unigram/bigram count files.")
    parser.add_argument("--windows-dir", required=True, help="Output directory for window score metadata.")
    parser.add_argument("--vocab-size", required=True, type=int, help="Tokenizer/model vocab size.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--window-batch-size", type=int, default=8192)
    parser.add_argument("--write-csv", action="store_true", help="Also write an inspectable CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = score_token_windows(
        tokens_dir=args.tokens_dir,
        counts_dir=args.counts_dir,
        windows_dir=args.windows_dir,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        stride=args.stride,
        window_batch_size=args.window_batch_size,
        write_csv=args.write_csv,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
