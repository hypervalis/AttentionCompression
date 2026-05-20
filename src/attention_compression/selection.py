from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SelectionSummary:
    scores_path: str
    output_dir: str
    train_count: int
    eval_count: int
    seed: int
    per_shard_cap: int | None


def load_window_scores(scores_path: str | Path) -> dict[str, np.ndarray]:
    scores = np.load(scores_path, allow_pickle=True)
    return {key: scores[key] for key in scores.files}


def _sample_indices(
    candidates: np.ndarray,
    *,
    target: int,
    rng: np.random.Generator,
    shard_ids: np.ndarray,
    per_shard_cap: int | None,
) -> np.ndarray:
    if target <= 0 or candidates.size == 0:
        return np.array([], dtype=np.int64)
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    if per_shard_cap is None:
        return shuffled[:target]

    counts: dict[int, int] = {}
    selected: list[int] = []
    for idx in shuffled:
        shard = int(shard_ids[idx])
        if counts.get(shard, 0) >= per_shard_cap:
            continue
        selected.append(int(idx))
        counts[shard] = counts.get(shard, 0) + 1
        if len(selected) >= target:
            break
    return np.array(selected, dtype=np.int64)


def select_train_eval_windows(
    *,
    scores_path: str | Path,
    output_dir: str | Path,
    train_targets: Mapping[str, int],
    eval_targets: Mapping[str, int],
    seed: int = 13,
    per_shard_cap: int | None = None,
) -> SelectionSummary:
    scores = load_window_scores(scores_path)
    shard_ids = scores["shard_id"]
    bin_ids = scores["rarity_bin_id"]
    bin_names = scores["rarity_bin_names"]
    rng = np.random.default_rng(seed)

    used: set[int] = set()
    train_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []

    for bin_id, bin_name_obj in enumerate(bin_names):
        bin_name = str(bin_name_obj)
        candidates = np.flatnonzero(bin_ids == bin_id)
        train = _sample_indices(
            candidates,
            target=int(train_targets.get(bin_name, 0)),
            rng=rng,
            shard_ids=shard_ids,
            per_shard_cap=per_shard_cap,
        )
        used.update(train.tolist())
        remaining = np.array([idx for idx in candidates if int(idx) not in used], dtype=np.int64)
        eval_selected = _sample_indices(
            remaining,
            target=int(eval_targets.get(bin_name, 0)),
            rng=rng,
            shard_ids=shard_ids,
            per_shard_cap=per_shard_cap,
        )
        used.update(eval_selected.tolist())
        train_parts.append(train)
        eval_parts.append(eval_selected)

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    eval_indices = np.concatenate(eval_parts) if eval_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_selection(out / "selected_train_windows.csv", scores, train_indices, split="train")
    _write_selection(out / "selected_eval_windows.csv", scores, eval_indices, split="eval")
    np.savez_compressed(
        out / "selected_windows.npz",
        train_indices=train_indices.astype(np.uint64),
        eval_indices=eval_indices.astype(np.uint64),
    )

    summary = SelectionSummary(
        scores_path=str(scores_path),
        output_dir=str(out),
        train_count=int(train_indices.size),
        eval_count=int(eval_indices.size),
        seed=seed,
        per_shard_cap=per_shard_cap,
    )
    with (out / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary


def _write_selection(
    path: Path,
    scores: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    split: str,
) -> None:
    shard_paths = scores["shard_paths"]
    bin_names = scores["rarity_bin_names"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
        for idx in indices:
            shard_id = int(scores["shard_id"][idx])
            bin_id = int(scores["rarity_bin_id"][idx])
            writer.writerow(
                [
                    int(idx),
                    split,
                    shard_id,
                    shard_paths[shard_id],
                    int(scores["start"][idx]),
                    int(scores["seq_len"]),
                    float(scores["mean_log_transition"][idx]),
                    float(scores["p05_log_transition"][idx]),
                    float(scores["p10_log_transition"][idx]),
                    float(scores["min_log_transition"][idx]),
                    float(scores["rarity_score"][idx]),
                    float(scores["rarity_percentile"][idx]),
                    bin_names[bin_id],
                ]
            )
