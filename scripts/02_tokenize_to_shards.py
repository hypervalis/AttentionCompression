#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_compression.tokenization import tokenize_jsonl_corpus_to_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize Dolma JSONL files into flat .npy shards.")
    parser.add_argument("--data-dir", required=True, help="Directory containing decompressed Dolma .json files.")
    parser.add_argument("--tokens-dir", required=True, help="Output directory for token shards.")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--file-glob", default="*.json")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--no-eos", action="store_true", help="Do not append EOS between documents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = tokenize_jsonl_corpus_to_shards(
        data_dir=args.data_dir,
        tokens_dir=args.tokens_dir,
        model_name=args.model_name,
        file_glob=args.file_glob,
        text_field=args.text_field,
        tokens_per_shard=args.tokens_per_shard,
        append_eos=not args.no_eos,
        max_documents=args.max_documents,
        progress_every=args.progress_every,
    )
    print(json.dumps(asdict(info), indent=2))


if __name__ == "__main__":
    main()
