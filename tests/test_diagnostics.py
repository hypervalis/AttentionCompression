from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from attention_compression.diagnostics import run_sampling_diagnostics


def write_selected(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "window_id",
        "split",
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_run_sampling_diagnostics_writes_report_and_validates_windows(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    selected_dir = tmp_path / "selected"
    report_dir = tmp_path / "reports"
    tokens_dir.mkdir()
    selected_dir.mkdir()
    shard_path = tokens_dir / "shard_000000.npy"
    np.save(shard_path, np.arange(32, dtype=np.uint16))

    manifest_path = tokens_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "shard_id": 0,
                "path": str(shard_path),
                "token_count": 32,
                "dtype": "uint16",
                "first_source": "synthetic",
                "source_counts": {"synthetic": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    scores_path = tmp_path / "window_scores.npz"
    np.savez_compressed(
        scores_path,
        shard_id=np.array([0, 0], dtype=np.uint32),
        shard_paths=np.array([str(shard_path)], dtype=object),
        start=np.array([0, 8], dtype=np.uint64),
        seq_len=np.uint32(8),
        stride=np.uint32(8),
        mean_log_transition=np.array([-1.0, -2.0], dtype=np.float32),
        p05_log_transition=np.array([-1.5, -2.5], dtype=np.float32),
        p10_log_transition=np.array([-1.4, -2.4], dtype=np.float32),
        min_log_transition=np.array([-2.0, -3.0], dtype=np.float32),
        rarity_score=np.array([1.0, 2.0], dtype=np.float32),
        rarity_percentile=np.array([0.0, 100.0], dtype=np.float32),
        rarity_bin_id=np.array([0, 1], dtype=np.uint8),
        rarity_bin_names=np.array(["very_common", "extreme_rare"], dtype=object),
    )

    row = {
        "window_id": 0,
        "split": "train",
        "shard_id": 0,
        "path": str(shard_path),
        "start": 0,
        "seq_len": 8,
        "mean_log_transition": -1.0,
        "p05_log_transition": -1.5,
        "p10_log_transition": -1.4,
        "min_log_transition": -2.0,
        "rarity_score": 1.0,
        "rarity_percentile": 0.0,
        "rarity_bin": "very_common",
    }
    write_selected(selected_dir / "selected_train_windows.csv", [row])
    write_selected(selected_dir / "selected_eval_windows.csv", [{**row, "split": "eval", "window_id": 1, "start": 8}])

    summary = run_sampling_diagnostics(
        scores_path=scores_path,
        selected_dir=selected_dir,
        manifest_path=manifest_path,
        output_dir=report_dir,
        validate_per_split=10,
    )

    assert summary.total_windows == 2
    assert summary.selected_train_windows == 1
    assert summary.selected_eval_windows == 1
    assert summary.validated_windows == 2
    assert summary.validation_errors == 0
    assert (report_dir / "sampling_diagnostics.json").exists()
    assert (report_dir / "rarity_bin_summary.csv").exists()
    assert (report_dir / "selection_bin_summary.csv").exists()
