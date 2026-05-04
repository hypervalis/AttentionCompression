from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from attention_compression.corpus import inspect_jsonl_corpus, iter_jsonl_documents
from attention_compression.selection import select_train_eval_windows


def test_inspect_and_iter_jsonl_corpus(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "sample.json"
    docs = [
        {"id": "a", "source": "books", "metadata": {"domain": "x"}, "text": "hello"},
        {"id": "b", "source": "web", "metadata": {"domain": "y"}, "text": "world"},
    ]
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    inspection = inspect_jsonl_corpus(data_dir=data_dir)
    seen = list(iter_jsonl_documents([path]))

    assert inspection.file_count == 1
    assert inspection.text_present is True
    assert inspection.sample_keys == ["id", "metadata", "source", "text"]
    assert inspection.metadata_keys == ["domain"]
    assert [text for _, _, _, text in seen] == ["hello", "world"]


def make_scores_npz(path: Path) -> None:
    shard_id = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint32)
    rarity_bin_names = np.array(["very_common", "common", "typical"], dtype=object)
    rarity_bin_id = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 1], dtype=np.uint8)
    np.savez_compressed(
        path,
        shard_id=shard_id,
        shard_paths=np.array(["a.npy", "b.npy", "c.npy"], dtype=object),
        start=np.arange(10, dtype=np.uint64) * 1024,
        seq_len=np.uint32(1024),
        stride=np.uint32(1024),
        mean_log_transition=np.linspace(-3.0, -1.0, 10, dtype=np.float32),
        p05_log_transition=np.linspace(-4.0, -2.0, 10, dtype=np.float32),
        p10_log_transition=np.linspace(-3.8, -1.8, 10, dtype=np.float32),
        min_log_transition=np.linspace(-8.0, -5.0, 10, dtype=np.float32),
        rarity_score=np.linspace(1.0, 3.0, 10, dtype=np.float32),
        rarity_percentile=np.linspace(0.0, 100.0, 10, dtype=np.float32),
        rarity_bin_id=rarity_bin_id,
        rarity_bin_names=rarity_bin_names,
    )


def test_select_train_eval_windows_is_disjoint_and_writes_manifests(tmp_path: Path) -> None:
    scores_path = tmp_path / "window_scores.npz"
    out = tmp_path / "selected"
    make_scores_npz(scores_path)

    summary = select_train_eval_windows(
        scores_path=scores_path,
        output_dir=out,
        train_targets={"very_common": 1, "common": 1, "typical": 1},
        eval_targets={"very_common": 1, "common": 1, "typical": 1},
        seed=123,
        per_shard_cap=2,
    )

    selected = np.load(out / "selected_windows.npz")
    train = set(selected["train_indices"].tolist())
    eval_ = set(selected["eval_indices"].tolist())
    assert summary.train_count == 3
    assert summary.eval_count == 3
    assert train.isdisjoint(eval_)
    assert (out / "selected_train_windows.csv").exists()
    assert (out / "selected_eval_windows.csv").exists()
    assert (out / "selection_summary.json").exists()

    with (out / "selected_train_windows.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {"very_common", "common", "typical"} == {row["rarity_bin"] for row in rows}
