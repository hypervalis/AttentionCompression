#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_compression.counts import build_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unigram and sparse observed-bigram counts.")
    parser.add_argument("--tokens-dir", required=True, help="Directory containing flat .npy token shards.")
    parser.add_argument("--counts-dir", required=True, help="Output directory for counts.")
    parser.add_argument("--vocab-size", required=True, type=int, help="Tokenizer/model vocab size.")
    parser.add_argument("--chunk-tokens", type=int, default=5_000_000)
    parser.add_argument("--merge-batch-size", type=int, default=32)
    parser.add_argument("--cleanup-intermediates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_counts(
        tokens_dir=args.tokens_dir,
        counts_dir=args.counts_dir,
        vocab_size=args.vocab_size,
        chunk_tokens=args.chunk_tokens,
        merge_batch_size=args.merge_batch_size,
        cleanup_intermediates=args.cleanup_intermediates,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
