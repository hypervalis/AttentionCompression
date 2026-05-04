from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attention_compression.counts import load_bigram_lookup
from attention_compression.token_shards import discover_token_shards, load_tokens


RARITY_BIN_NAMES = np.array(
    ["very_common", "common", "typical", "rare", "very_rare", "extreme_rare"],
    dtype=object,
)


@dataclass(frozen=True)
class WindowScoreSummary:
    tokens_dir: str
    counts_dir: str
    windows_dir: str
    seq_len: int
    stride: int
    window_count: int
    scores_path: str
    csv_path: str | None


def rarity_bin_ids(rarity_percentile: np.ndarray) -> np.ndarray:
    """Map rarity percentile [0, 100] to coarse distribution bins."""
    p = rarity_percentile
    bins = np.empty(p.shape, dtype=np.uint8)
    bins[p < 5.0] = 0
    bins[(p >= 5.0) & (p < 20.0)] = 1
    bins[(p >= 20.0) & (p < 80.0)] = 2
    bins[(p >= 80.0) & (p < 95.0)] = 3
    bins[(p >= 95.0) & (p < 99.0)] = 4
    bins[p >= 99.0] = 5
    return bins


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return stable empirical percentile ranks in [0, 100]."""
    if values.size == 0:
        return np.array([], dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    if values.size == 1:
        ranks[order] = 100.0
    else:
        ranks[order] = np.arange(values.size, dtype=np.float64) * (100.0 / (values.size - 1))
    return ranks.astype(np.float32)


def _lookup_counts(
    window_keys: np.ndarray,
    sparse_keys: np.ndarray,
    sparse_counts: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(sparse_keys, window_keys)
    in_bounds = idx < sparse_keys.size
    ok = np.zeros(window_keys.shape, dtype=bool)
    ok[in_bounds] = sparse_keys[idx[in_bounds]] == window_keys[in_bounds]
    if not np.all(ok):
        missing = int(np.count_nonzero(~ok))
        raise KeyError(f"{missing} window transitions were not present in sparse bigram counts")
    return sparse_counts[idx]


def score_token_windows(
    *,
    tokens_dir: str | Path,
    counts_dir: str | Path,
    windows_dir: str | Path,
    vocab_size: int,
    seq_len: int,
    stride: int,
    write_csv: bool = False,
    window_batch_size: int = 8192,
) -> WindowScoreSummary:
    """Score fixed windows by negative mean empirical log bigram probability."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")

    counts_root = Path(counts_dir)
    windows_root = Path(windows_dir)
    windows_root.mkdir(parents=True, exist_ok=True)

    unigram = np.load(counts_root / "unigram_counts.npy", mmap_mode="r")
    if unigram.size != vocab_size:
        raise ValueError(f"unigram size {unigram.size} does not match vocab_size={vocab_size}")

    sparse_keys, sparse_counts = load_bigram_lookup(counts_root / "bigram_counts_sparse.npz")
    shards = discover_token_shards(tokens_dir)
    shard_paths = np.array([str(s.path) for s in shards], dtype=object)
    started_at = time.monotonic()

    shard_ids: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    mean_logs: list[np.ndarray] = []
    p05_logs: list[np.ndarray] = []
    min_logs: list[np.ndarray] = []
    p10_logs: list[np.ndarray] = []

    for shard in shards:
        tokens = load_tokens(shard.path, mmap=True)
        if tokens.size < seq_len:
            continue
        starts_arr = np.arange(0, tokens.size - seq_len + 1, stride, dtype=np.uint64)
        if starts_arr.size == 0:
            continue

        shard_mean = np.empty(starts_arr.size, dtype=np.float32)
        shard_p05 = np.empty(starts_arr.size, dtype=np.float32)
        shard_p10 = np.empty(starts_arr.size, dtype=np.float32)
        shard_min = np.empty(starts_arr.size, dtype=np.float32)

        transition_count = seq_len - 1
        for batch_start in range(0, starts_arr.size, window_batch_size):
            batch_stop = min(starts_arr.size, batch_start + window_batch_size)
            batch_starts = starts_arr[batch_start:batch_stop]
            offsets = np.arange(transition_count, dtype=np.uint64)
            prev_idx = batch_starts[:, None] + offsets[None, :]
            nxt_idx = prev_idx + np.uint64(1)
            prev = np.asarray(tokens[prev_idx], dtype=np.uint64)
            nxt = np.asarray(tokens[nxt_idx], dtype=np.uint64)
            keys = prev * np.uint64(vocab_size) + nxt
            cxy = _lookup_counts(keys.ravel(), sparse_keys, sparse_counts).reshape(prev.shape)
            cx = np.asarray(unigram[prev], dtype=np.float64)
            logps = np.log(cxy.astype(np.float64, copy=False)) - np.log(cx)
            shard_mean[batch_start:batch_stop] = np.mean(logps, axis=1)
            shard_p05[batch_start:batch_stop] = np.percentile(logps, 5, axis=1)
            shard_p10[batch_start:batch_stop] = np.percentile(logps, 10, axis=1)
            shard_min[batch_start:batch_stop] = np.min(logps, axis=1)

        shard_ids.append(np.full(starts_arr.size, shard.shard_id, dtype=np.uint32))
        starts.append(starts_arr)
        mean_logs.append(shard_mean)
        p05_logs.append(shard_p05)
        p10_logs.append(shard_p10)
        min_logs.append(shard_min)
        elapsed = max(time.monotonic() - started_at, 1e-6)
        windows_done = sum(part.size for part in starts)
        print(
            f"scored shard_id={shard.shard_id} windows={windows_done} "
            f"windows_per_sec={windows_done / elapsed:.2f}",
            flush=True,
        )

    if not shard_ids:
        raise ValueError("No scoreable windows found")

    all_shard_ids = np.concatenate(shard_ids)
    all_starts = np.concatenate(starts)
    all_mean_logs = np.concatenate(mean_logs)
    all_p05_logs = np.concatenate(p05_logs)
    all_p10_logs = np.concatenate(p10_logs)
    all_min_logs = np.concatenate(min_logs)
    rarity_scores = (-all_mean_logs).astype(np.float32, copy=False)
    rarity_percentiles = percentile_ranks(rarity_scores)
    bin_ids = rarity_bin_ids(rarity_percentiles)

    scores_path = windows_root / "window_scores.npz"
    np.savez_compressed(
        scores_path,
        shard_id=all_shard_ids,
        shard_paths=shard_paths,
        start=all_starts,
        seq_len=np.uint32(seq_len),
        stride=np.uint32(stride),
        mean_log_transition=all_mean_logs,
        p05_log_transition=all_p05_logs,
        p10_log_transition=all_p10_logs,
        min_log_transition=all_min_logs,
        rarity_score=rarity_scores,
        rarity_percentile=rarity_percentiles,
        rarity_bin_id=bin_ids,
        rarity_bin_names=RARITY_BIN_NAMES,
    )

    csv_path = None
    if write_csv:
        csv_path = windows_root / "window_scores.csv"
        write_window_scores_csv(scores_path, csv_path)

    summary = WindowScoreSummary(
        tokens_dir=str(tokens_dir),
        counts_dir=str(counts_dir),
        windows_dir=str(windows_dir),
        seq_len=seq_len,
        stride=stride,
        window_count=int(all_starts.size),
        scores_path=str(scores_path),
        csv_path=str(csv_path) if csv_path is not None else None,
    )
    with (windows_root / "window_score_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary


def write_window_scores_csv(scores_path: str | Path, csv_path: str | Path) -> None:
    """Write an inspectable CSV from the compact NPZ score table."""
    scores = np.load(scores_path, allow_pickle=True)
    shard_paths = scores["shard_paths"]
    bin_names = scores["rarity_bin_names"]
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_id",
                "shard_id",
                "path",
                "start",
                "seq_len",
                "mean_log_transition",
                "p05_log_transition",
                "p10_log_transition",
                "min_log_transition",
                "rarity_score",
                "rarity_percentile",
                "rarity_bin",
            ]
        )
        for window_id in range(scores["start"].size):
            shard_id = int(scores["shard_id"][window_id])
            bin_id = int(scores["rarity_bin_id"][window_id])
            writer.writerow(
                [
                    window_id,
                    shard_id,
                    shard_paths[shard_id],
                    int(scores["start"][window_id]),
                    int(scores["seq_len"]),
                    float(scores["mean_log_transition"][window_id]),
                    float(scores["p05_log_transition"][window_id]),
                    float(scores["p10_log_transition"][window_id]),
                    float(scores["min_log_transition"][window_id]),
                    float(scores["rarity_score"][window_id]),
                    float(scores["rarity_percentile"][window_id]),
                    bin_names[bin_id],
                ]
            )
