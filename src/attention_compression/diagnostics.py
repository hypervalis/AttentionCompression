from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiagnosticsSummary:
    scores_path: str
    selected_dir: str
    output_dir: str
    total_windows: int
    selected_train_windows: int
    selected_eval_windows: int
    validated_windows: int
    validation_errors: int


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_selected_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_quantiles(values: np.ndarray) -> dict[str, float]:
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    out: dict[str, float] = {}
    for q, value in zip(qs, np.percentile(values, qs), strict=True):
        out[f"p{q:02d}"] = float(value)
    return out


def summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    rarity = scores["rarity_score"]
    bin_ids = scores["rarity_bin_id"]
    bin_names = scores["rarity_bin_names"]
    summary: dict[str, Any] = {
        "total_windows": int(rarity.size),
        "rarity_score_quantiles": score_quantiles(rarity),
        "bins": {},
    }
    for bin_id, bin_name_obj in enumerate(bin_names):
        name = str(bin_name_obj)
        mask = bin_ids == bin_id
        values = rarity[mask]
        summary["bins"][name] = {
            "count": int(values.size),
            "rarity_score_min": float(np.min(values)) if values.size else None,
            "rarity_score_max": float(np.max(values)) if values.size else None,
            "rarity_score_mean": float(np.mean(values)) if values.size else None,
            "rarity_percentile_min": float(np.min(scores["rarity_percentile"][mask])) if values.size else None,
            "rarity_percentile_max": float(np.max(scores["rarity_percentile"][mask])) if values.size else None,
        }
    return summary


def summarize_selection(rows: list[dict[str, str]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    by_bin = Counter(row["rarity_bin"] for row in rows)
    by_shard = Counter(int(row["shard_id"]) for row in rows)
    manifest_by_shard = {int(row["shard_id"]): row for row in manifest}
    by_first_source: Counter[str] = Counter()
    source_doc_counts: Counter[str] = Counter()
    for shard_id, selected_count in by_shard.items():
        record = manifest_by_shard.get(shard_id, {})
        by_first_source[str(record.get("first_source") or "unknown")] += selected_count
        for source, doc_count in record.get("source_counts", {}).items():
            source_doc_counts[str(source)] += int(doc_count)

    return {
        "count": len(rows),
        "by_rarity_bin": dict(sorted(by_bin.items())),
        "shard_count": len(by_shard),
        "max_windows_from_one_shard": max(by_shard.values()) if by_shard else 0,
        "top_shards": [{"shard_id": shard, "windows": count} for shard, count in by_shard.most_common(20)],
        "by_first_source": dict(by_first_source.most_common()),
        "manifest_source_doc_counts_for_touched_shards": dict(source_doc_counts.most_common()),
    }


def validate_selected_windows(
    rows: list[dict[str, str]],
    *,
    max_windows: int,
) -> tuple[int, list[str]]:
    """Load selected token spans and verify length/range without materializing all rows."""
    errors: list[str] = []
    loaded_shards: dict[str, np.ndarray] = {}
    checked = 0
    for row in rows[:max_windows]:
        path = row["path"]
        if path not in loaded_shards:
            loaded_shards[path] = np.load(path, mmap_mode="r")
        tokens = loaded_shards[path]
        start = int(row["start"])
        seq_len = int(row["seq_len"])
        if start < 0 or seq_len <= 0 or start + seq_len > tokens.size:
            errors.append(
                f"window_id={row['window_id']} path={path} start={start} seq_len={seq_len} tokens={tokens.size}"
            )
            continue
        window = tokens[start : start + seq_len]
        if window.shape != (seq_len,):
            errors.append(f"window_id={row['window_id']} returned shape={window.shape}")
        checked += 1
    return checked, errors


def run_sampling_diagnostics(
    *,
    scores_path: str | Path,
    selected_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    validate_per_split: int = 1000,
) -> DiagnosticsSummary:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_raw = np.load(scores_path, allow_pickle=True)
    scores = {key: scores_raw[key] for key in scores_raw.files}
    manifest = load_jsonl(manifest_path)
    train_rows = load_selected_csv(Path(selected_dir) / "selected_train_windows.csv")
    eval_rows = load_selected_csv(Path(selected_dir) / "selected_eval_windows.csv")

    score_summary = summarize_scores(scores)
    train_summary = summarize_selection(train_rows, manifest)
    eval_summary = summarize_selection(eval_rows, manifest)
    checked_train, train_errors = validate_selected_windows(train_rows, max_windows=validate_per_split)
    checked_eval, eval_errors = validate_selected_windows(eval_rows, max_windows=validate_per_split)

    report = {
        "score_summary": score_summary,
        "selection": {
            "train": train_summary,
            "eval": eval_summary,
        },
        "validation": {
            "train_checked": checked_train,
            "eval_checked": checked_eval,
            "errors": train_errors + eval_errors,
        },
    }
    with (out / "sampling_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_bin_summary_csv(out / "rarity_bin_summary.csv", score_summary)
    write_selection_summary_csv(out / "selection_bin_summary.csv", train_summary, eval_summary)

    return DiagnosticsSummary(
        scores_path=str(scores_path),
        selected_dir=str(selected_dir),
        output_dir=str(out),
        total_windows=int(scores["start"].size),
        selected_train_windows=len(train_rows),
        selected_eval_windows=len(eval_rows),
        validated_windows=checked_train + checked_eval,
        validation_errors=len(train_errors) + len(eval_errors),
    )


def write_bin_summary_csv(path: str | Path, score_summary: dict[str, Any]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rarity_bin", "count", "rarity_score_min", "rarity_score_max", "rarity_score_mean"])
        for name, row in score_summary["bins"].items():
            writer.writerow(
                [
                    name,
                    row["count"],
                    row["rarity_score_min"],
                    row["rarity_score_max"],
                    row["rarity_score_mean"],
                ]
            )


def write_selection_summary_csv(
    path: str | Path,
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
) -> None:
    all_bins = sorted(set(train_summary["by_rarity_bin"]) | set(eval_summary["by_rarity_bin"]))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rarity_bin", "train_count", "eval_count"])
        for name in all_bins:
            writer.writerow(
                [
                    name,
                    train_summary["by_rarity_bin"].get(name, 0),
                    eval_summary["by_rarity_bin"].get(name, 0),
                ]
            )
