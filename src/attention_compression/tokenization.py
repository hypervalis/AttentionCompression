from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from attention_compression.corpus import discover_corpus_files, iter_jsonl_documents


@dataclass(frozen=True)
class TokenizerInfo:
    model_name: str
    tokenizer_len: int
    tokenizer_vocab_size: int | None
    config_vocab_size: int | None
    effective_vocab_size: int
    eos_token: str | None
    eos_token_id: int | None
    token_id_dtype: str


@dataclass(frozen=True)
class TokenShardRecord:
    shard_id: int
    path: str
    token_count: int
    dtype: str
    first_source: str | None
    source_counts: dict[str, int]


def choose_token_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return np.dtype(np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32)


def load_tokenizer_info(model_name: str) -> tuple[Any, TokenizerInfo]:
    """Load the model tokenizer/config and return the effective ID space size."""
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install the tokenize extra: pip install -e '.[tokenize]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config_vocab_size = None
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config_vocab_size = getattr(config, "vocab_size", None)
    except Exception:
        config = None  # noqa: F841 - useful when debugging locally.

    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    tokenizer_len = len(tokenizer)
    candidates = [tokenizer_len]
    if isinstance(tokenizer_vocab_size, int):
        candidates.append(tokenizer_vocab_size)
    if isinstance(config_vocab_size, int):
        candidates.append(config_vocab_size)
    effective_vocab_size = max(candidates)
    dtype = choose_token_dtype(effective_vocab_size)
    info = TokenizerInfo(
        model_name=model_name,
        tokenizer_len=tokenizer_len,
        tokenizer_vocab_size=tokenizer_vocab_size if isinstance(tokenizer_vocab_size, int) else None,
        config_vocab_size=config_vocab_size if isinstance(config_vocab_size, int) else None,
        effective_vocab_size=effective_vocab_size,
        eos_token=tokenizer.eos_token,
        eos_token_id=tokenizer.eos_token_id,
        token_id_dtype=dtype.name,
    )
    return tokenizer, info


class TokenShardWriter:
    def __init__(
        self,
        *,
        tokens_dir: str | Path,
        dtype: np.dtype,
        tokens_per_shard: int,
    ) -> None:
        if tokens_per_shard <= 0:
            raise ValueError("tokens_per_shard must be positive")
        self.tokens_dir = Path(tokens_dir)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        self.tokens_per_shard = tokens_per_shard
        self.records: list[TokenShardRecord] = []
        self._buffer: list[np.ndarray] = []
        self._buffer_tokens = 0
        self._shard_id = 0
        self._source_counts: dict[str, int] = {}
        self._first_source: str | None = None

    def add(self, token_ids: list[int], *, source: str | None) -> None:
        if not token_ids:
            return
        arr = np.asarray(token_ids, dtype=self.dtype)
        self._buffer.append(arr)
        self._buffer_tokens += int(arr.size)
        source_key = source or "unknown"
        self._source_counts[source_key] = self._source_counts.get(source_key, 0) + 1
        if self._first_source is None:
            self._first_source = source
        if self._buffer_tokens >= self.tokens_per_shard:
            self.flush()

    def flush(self) -> None:
        if self._buffer_tokens == 0:
            return
        out = self.tokens_dir / f"shard_{self._shard_id:06d}.npy"
        tokens = np.concatenate(self._buffer).astype(self.dtype, copy=False)
        np.save(out, tokens)
        self.records.append(
            TokenShardRecord(
                shard_id=self._shard_id,
                path=str(out),
                token_count=int(tokens.size),
                dtype=self.dtype.name,
                first_source=self._first_source,
                source_counts=dict(sorted(self._source_counts.items())),
            )
        )
        self._buffer = []
        self._buffer_tokens = 0
        self._source_counts = {}
        self._first_source = None
        self._shard_id += 1


def write_manifest(tokens_dir: str | Path, records: list[TokenShardRecord], tokenizer_info: TokenizerInfo) -> None:
    root = Path(tokens_dir)
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (root / "tokenizer_info.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(tokenizer_info), f, indent=2)


def tokenize_jsonl_corpus_to_shards(
    *,
    data_dir: str | Path,
    tokens_dir: str | Path,
    model_name: str,
    file_glob: str = "*.json",
    text_field: str = "text",
    tokens_per_shard: int = 50_000_000,
    append_eos: bool = True,
    max_documents: int | None = None,
    progress_every: int = 10_000,
) -> TokenizerInfo:
    tokenizer, info = load_tokenizer_info(model_name)
    dtype = choose_token_dtype(info.effective_vocab_size)
    writer = TokenShardWriter(tokens_dir=tokens_dir, dtype=dtype, tokens_per_shard=tokens_per_shard)
    eos = [int(info.eos_token_id)] if append_eos and info.eos_token_id is not None else []

    doc_count = 0
    token_count = 0
    started_at = time.monotonic()
    corpus_files = discover_corpus_files(data_dir, file_glob)
    for _, _, obj, text in iter_jsonl_documents(corpus_files, text_field=text_field):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos:
            token_ids = [*token_ids, *eos]
        token_count += len(token_ids)
        writer.add(token_ids, source=obj.get("source"))
        doc_count += 1
        if progress_every > 0 and doc_count % progress_every == 0:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            docs_per_sec = doc_count / elapsed
            tokens_per_sec = token_count / elapsed
            print(
                f"tokenized docs={doc_count} tokens={token_count} "
                f"docs_per_sec={docs_per_sec:.2f} tokens_per_sec={tokens_per_sec:.2f}",
                flush=True,
            )
        if max_documents is not None and doc_count >= max_documents:
            break

    writer.flush()
    write_manifest(tokens_dir, writer.records, info)
    return info
