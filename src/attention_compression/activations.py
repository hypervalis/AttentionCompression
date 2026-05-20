from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_selected_rows_by_bin(
    path: str | Path,
    *,
    windows_per_bin: int,
) -> list[dict[str, str]]:
    """Load up to `windows_per_bin` selected-window rows from each rarity bin."""
    if windows_per_bin <= 0:
        raise ValueError("windows_per_bin must be positive")
    rows_by_bin: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bucket = rows_by_bin[row["rarity_bin"]]
            if len(bucket) < windows_per_bin:
                bucket.append(row)
    rows: list[dict[str, str]] = []
    for bin_name in sorted(rows_by_bin):
        rows.extend(rows_by_bin[bin_name])
    return rows


def rows_to_token_batch(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    """Materialize selected windows into a `[batch, seq_len]` NumPy token array."""
    if not rows:
        raise ValueError("No selected window rows provided")
    seq_lens = {int(row["seq_len"]) for row in rows}
    if len(seq_lens) != 1:
        raise ValueError(f"Rows must share one seq_len, got {sorted(seq_lens)}")
    seq_len = seq_lens.pop()
    loaded_shards: dict[str, np.ndarray] = {}
    batch = np.empty((len(rows), seq_len), dtype=np.int64)
    bins: list[str] = []
    for i, row in enumerate(rows):
        path = row["path"]
        if path not in loaded_shards:
            loaded_shards[path] = np.load(path, mmap_mode="r")
        tokens = loaded_shards[path]
        start = int(row["start"])
        stop = start + seq_len
        if stop > tokens.size:
            raise ValueError(f"Window exceeds shard bounds: {row}")
        batch[i] = tokens[start:stop]
        bins.append(row["rarity_bin"])
    return batch, bins


def find_transformer_layers(model: Any) -> tuple[str, Any]:
    """Find the transformer block list for common Hugging Face causal LMs."""
    candidates = [
        "model.layers",
        "model.model.layers",
        "model.transformer.blocks",
        "model.transformer.h",
        "transformer.blocks",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "__len__") and hasattr(cur, "__getitem__"):
            return path, cur
    raise ValueError("Could not locate transformer layer list on model")


@dataclass(frozen=True)
class CaptureShardMetadata:
    shard_id: int
    path: str
    sample_count: int
    seq_len: int
    hidden_size: int
    dtype: str
    target_layer: int
    rarity_bin_counts: dict[str, int]


def write_capture_manifest(
    *,
    output_dir: str | Path,
    records: list[CaptureShardMetadata],
    run_config: dict[str, Any],
) -> None:
    out = Path(output_dir)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (out / "capture_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
