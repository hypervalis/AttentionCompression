from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from attention_compression.activations import find_transformer_layers, load_selected_rows_by_bin, rows_to_token_batch


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["window_id", "rarity_bin", "path", "start", "seq_len"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_selected_rows_by_bin_and_materialize_batch(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.npy"
    np.save(shard_path, np.arange(32, dtype=np.uint16))
    csv_path = tmp_path / "selected.csv"
    rows = [
        {"window_id": 0, "rarity_bin": "rare", "path": shard_path, "start": 0, "seq_len": 8},
        {"window_id": 1, "rarity_bin": "rare", "path": shard_path, "start": 8, "seq_len": 8},
        {"window_id": 2, "rarity_bin": "typical", "path": shard_path, "start": 16, "seq_len": 8},
    ]
    write_rows(csv_path, rows)

    selected = load_selected_rows_by_bin(csv_path, windows_per_bin=1)
    batch, bins = rows_to_token_batch(selected)

    assert [row["window_id"] for row in selected] == ["0", "2"]
    assert batch.shape == (2, 8)
    assert bins == ["rare", "typical"]
    assert batch[0].tolist() == list(range(8))
    assert batch[1].tolist() == list(range(16, 24))


class DummyLayerList(list):
    pass


class DummyModelInner:
    def __init__(self) -> None:
        self.layers = DummyLayerList([object(), object()])


class DummyModel:
    def __init__(self) -> None:
        self.model = DummyModelInner()


def test_find_transformer_layers_common_path() -> None:
    path, layers = find_transformer_layers(DummyModel())

    assert path == "model.layers"
    assert len(layers) == 2
