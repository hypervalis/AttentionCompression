from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attention_compression.token_shards import iter_token_shards


@dataclass(frozen=True)
class CountBuildSummary:
    tokens_dir: str
    counts_dir: str
    vocab_size: int
    total_tokens: int
    total_transitions: int
    unigram_path: str
    bigram_path: str
    run_count: int


def _save_sparse_counts(path: Path, keys: np.ndarray, counts: np.ndarray) -> None:
    np.savez_compressed(
        path,
        keys=keys.astype(np.uint64, copy=False),
        counts=counts.astype(np.uint64, copy=False),
    )


def _load_sparse_counts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["keys"].astype(np.uint64, copy=False), data["counts"].astype(np.uint64, copy=False)


def _merge_sparse_arrays(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    counts = counts[order]
    unique_keys, first_idx = np.unique(keys, return_index=True)
    merged_counts = np.add.reduceat(counts, first_idx).astype(np.uint64, copy=False)
    return unique_keys.astype(np.uint64, copy=False), merged_counts


def write_bigram_count_runs(
    *,
    tokens_dir: str | Path,
    counts_dir: str | Path,
    vocab_size: int,
    chunk_tokens: int = 5_000_000,
) -> tuple[np.ndarray, int, int, int]:
    """Create per-chunk sparse observed-bigram run files and unigram counts.

    Bigrams are counted inside each shard. This is enough for the rough empirical
    rarity distribution we want, while keeping raw corpus processing sequential.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if chunk_tokens < 2:
        raise ValueError("chunk_tokens must be at least 2")

    counts_root = Path(counts_dir)
    runs_dir = counts_root / "bigram_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    unigram = np.zeros(vocab_size, dtype=np.uint64)
    run_idx = 0
    total_tokens = 0
    total_transitions = 0
    started_at = time.monotonic()

    for shard, tokens in iter_token_shards(tokens_dir):
        if tokens.size == 0:
            continue
        max_token = int(np.max(tokens))
        if max_token >= vocab_size:
            raise ValueError(
                f"Token ID {max_token} in {shard.path} is >= vocab_size={vocab_size}"
            )

        total_tokens += int(tokens.size)
        unigram += np.bincount(tokens, minlength=vocab_size).astype(np.uint64, copy=False)

        if tokens.size < 2:
            continue
        for start in range(0, tokens.size - 1, chunk_tokens):
            stop = min(tokens.size - 1, start + chunk_tokens)
            prev = tokens[start:stop].astype(np.uint64, copy=False)
            nxt = tokens[start + 1 : stop + 1].astype(np.uint64, copy=False)
            keys = prev * np.uint64(vocab_size) + nxt
            unique_keys, counts = np.unique(keys, return_counts=True)
            run_path = runs_dir / f"run_{run_idx:08d}.npz"
            _save_sparse_counts(run_path, unique_keys, counts)
            total_transitions += int(keys.size)
            run_idx += 1

        elapsed = max(time.monotonic() - started_at, 1e-6)
        print(
            f"counted shard_id={shard.shard_id} tokens={total_tokens} "
            f"transitions={total_transitions} runs={run_idx} "
            f"tokens_per_sec={total_tokens / elapsed:.2f}",
            flush=True,
        )

    counts_root.mkdir(parents=True, exist_ok=True)
    np.save(counts_root / "unigram_counts.npy", unigram)
    return unigram, total_tokens, total_transitions, run_idx


def merge_bigram_runs(
    *,
    counts_dir: str | Path,
    batch_size: int = 32,
    final_name: str = "bigram_counts_sparse.npz",
    cleanup_intermediates: bool = False,
) -> Path:
    """Merge sparse bigram run files into one observed sparse count table."""
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")

    counts_root = Path(counts_dir)
    current = sorted((counts_root / "bigram_runs").glob("*.npz"))
    if not current:
        raise FileNotFoundError(f"No bigram run files found under {counts_root / 'bigram_runs'}")

    round_idx = 0
    while len(current) > 1:
        previous = current
        print(
            f"merge round={round_idx} input_files={len(previous)} batch_size={batch_size}",
            flush=True,
        )
        next_dir = counts_root / f"bigram_merge_round_{round_idx:02d}"
        next_dir.mkdir(parents=True, exist_ok=True)
        next_paths: list[Path] = []

        for batch_idx, start in enumerate(range(0, len(previous), batch_size)):
            batch = previous[start : start + batch_size]
            key_parts = []
            count_parts = []
            for path in batch:
                keys, counts = _load_sparse_counts(path)
                key_parts.append(keys)
                count_parts.append(counts)
            merged_keys, merged_counts = _merge_sparse_arrays(
                np.concatenate(key_parts), np.concatenate(count_parts)
            )
            out_path = next_dir / f"run_{batch_idx:08d}.npz"
            _save_sparse_counts(out_path, merged_keys, merged_counts)
            next_paths.append(out_path)
            print(
                f"merge round={round_idx} batch={batch_idx} files={len(batch)} "
                f"unique_bigrams={merged_keys.size}",
                flush=True,
            )

        if cleanup_intermediates:
            for path in previous:
                path.unlink(missing_ok=True)
            # Remove empty run/round directories as soon as they are no longer needed.
            for path in sorted({p.parent for p in previous}, reverse=True):
                try:
                    path.rmdir()
                except OSError:
                    pass

        current = next_paths
        round_idx += 1

    final_path = counts_root / final_name
    keys, counts = _load_sparse_counts(current[0])
    _save_sparse_counts(final_path, keys, counts)
    print(f"merge final unique_bigrams={keys.size} path={final_path}", flush=True)
    if cleanup_intermediates:
        current[0].unlink(missing_ok=True)
        for path in sorted(counts_root.glob("bigram_merge_round_*"), reverse=True):
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(counts_root / "bigram_runs", ignore_errors=True)
    return final_path


def build_counts(
    *,
    tokens_dir: str | Path,
    counts_dir: str | Path,
    vocab_size: int,
    chunk_tokens: int = 5_000_000,
    merge_batch_size: int = 32,
    cleanup_intermediates: bool = False,
) -> CountBuildSummary:
    unigram, total_tokens, total_transitions, run_count = write_bigram_count_runs(
        tokens_dir=tokens_dir,
        counts_dir=counts_dir,
        vocab_size=vocab_size,
        chunk_tokens=chunk_tokens,
    )
    bigram_path = merge_bigram_runs(
        counts_dir=counts_dir,
        batch_size=merge_batch_size,
        cleanup_intermediates=cleanup_intermediates,
    )
    summary = CountBuildSummary(
        tokens_dir=str(tokens_dir),
        counts_dir=str(counts_dir),
        vocab_size=vocab_size,
        total_tokens=total_tokens,
        total_transitions=total_transitions,
        unigram_path=str(Path(counts_dir) / "unigram_counts.npy"),
        bigram_path=str(bigram_path),
        run_count=run_count,
    )
    with (Path(counts_dir) / "counts_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    # Keep a cheap invariant close to the outputs.
    if int(unigram.sum()) != total_tokens:
        raise RuntimeError("Unigram count total does not match token total")
    return summary


def load_bigram_lookup(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load sorted sparse bigram keys and counts for scoring."""
    keys, counts = _load_sparse_counts(Path(path))
    order = np.argsort(keys, kind="stable")
    return keys[order], counts[order]
