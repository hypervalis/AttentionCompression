from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attention_compression.counts import build_counts, load_bigram_lookup
from attention_compression.token_shards import discover_token_shards, load_tokens
from attention_compression.windows import RARITY_BIN_NAMES, rarity_bin_ids, score_token_windows


def write_synthetic_shards(tokens_dir: Path) -> None:
    tokens_dir.mkdir(parents=True)
    np.save(
        tokens_dir / "shard_000000.npy",
        np.array([1, 2, 3, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 2, 3, 4], dtype=np.uint16),
    )
    np.save(
        tokens_dir / "shard_000001.npy",
        np.array([1, 2, 1, 2, 3, 5, 5, 5, 1, 2, 3, 4, 4, 4, 1, 2], dtype=np.uint16),
    )


def test_discover_and_load_token_shards(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    write_synthetic_shards(tokens_dir)

    shards = discover_token_shards(tokens_dir)

    assert [s.shard_id for s in shards] == [0, 1]
    assert [s.path.name for s in shards] == ["shard_000000.npy", "shard_000001.npy"]
    assert load_tokens(shards[0].path).shape == (16,)


def test_load_tokens_rejects_non_flat_arrays(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[1, 2], [3, 4]], dtype=np.uint16))

    with pytest.raises(ValueError, match="flat/1-D"):
        load_tokens(path)


def test_build_counts_creates_expected_sparse_outputs(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    counts_dir = tmp_path / "counts"
    write_synthetic_shards(tokens_dir)

    summary = build_counts(
        tokens_dir=tokens_dir,
        counts_dir=counts_dir,
        vocab_size=8,
        chunk_tokens=5,
        merge_batch_size=3,
    )

    assert summary.total_tokens == 32
    assert summary.total_transitions == 30
    assert summary.run_count == 6
    assert (counts_dir / "unigram_counts.npy").exists()
    assert (counts_dir / "bigram_counts_sparse.npz").exists()
    assert (counts_dir / "counts_summary.json").exists()

    unigram = np.load(counts_dir / "unigram_counts.npy")
    assert int(unigram.sum()) == 32
    assert int(unigram[1]) == 7
    assert int(unigram[2]) == 9

    keys, counts = load_bigram_lookup(counts_dir / "bigram_counts_sparse.npz")
    lookup = dict(zip(keys.tolist(), counts.tolist(), strict=True))
    vocab_size = 8
    assert lookup[1 * vocab_size + 2] == 7
    assert lookup[2 * vocab_size + 3] == 7
    assert lookup[3 * vocab_size + 4] == 3


def test_build_counts_rejects_token_ids_outside_vocab(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    np.save(tokens_dir / "shard_000000.npy", np.array([1, 2, 99], dtype=np.uint16))

    with pytest.raises(ValueError, match=">= vocab_size"):
        build_counts(tokens_dir=tokens_dir, counts_dir=tmp_path / "counts", vocab_size=8)


def test_score_token_windows_outputs_rarity_metadata(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    counts_dir = tmp_path / "counts"
    windows_dir = tmp_path / "windows"
    write_synthetic_shards(tokens_dir)
    build_counts(
        tokens_dir=tokens_dir,
        counts_dir=counts_dir,
        vocab_size=8,
        chunk_tokens=5,
        merge_batch_size=3,
    )

    summary = score_token_windows(
        tokens_dir=tokens_dir,
        counts_dir=counts_dir,
        windows_dir=windows_dir,
        vocab_size=8,
        seq_len=4,
        stride=4,
        write_csv=True,
    )

    assert summary.window_count == 8
    assert (windows_dir / "window_scores.npz").exists()
    assert (windows_dir / "window_scores.csv").exists()
    assert (windows_dir / "window_score_summary.json").exists()

    scores = np.load(windows_dir / "window_scores.npz", allow_pickle=True)
    assert scores["start"].tolist() == [0, 4, 8, 12, 0, 4, 8, 12]
    assert scores["shard_id"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert int(scores["seq_len"]) == 4
    assert scores["rarity_score"].shape == (8,)
    assert np.all(np.isfinite(scores["mean_log_transition"]))
    assert np.all(np.isfinite(scores["p10_log_transition"]))
    assert np.all(scores["rarity_percentile"] >= 0)
    assert np.all(scores["rarity_percentile"] <= 100)
    assert set(scores["rarity_bin_names"].tolist()) == set(RARITY_BIN_NAMES.tolist())


def test_rarity_bin_boundaries() -> None:
    percentiles = np.array([0.0, 4.99, 5.0, 19.99, 20.0, 79.99, 80.0, 94.99, 95.0, 98.99, 99.0, 100.0])

    bin_ids = rarity_bin_ids(percentiles)

    assert RARITY_BIN_NAMES[bin_ids].tolist() == [
        "very_common",
        "very_common",
        "common",
        "common",
        "typical",
        "typical",
        "rare",
        "rare",
        "very_rare",
        "very_rare",
        "extreme_rare",
        "extreme_rare",
    ]
