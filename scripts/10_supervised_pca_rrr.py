#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from attention_compression.supervised_pca import centered_stats_from_raw, fit_ridge_rrr, rrr_coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit supervised PCA / reduced-rank regression baseline.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,96,128")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--train-tokens-per-shard", type=int, default=4096)
    parser.add_argument("--eval-tokens-per-shard", type=int, default=1024)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def load_capture_paths(capture_dir: str | Path) -> list[Path]:
    paths = sorted(Path(capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt activation shards found under {capture_dir}")
    return paths


def sample_indices(n_rows: int, train_n: int, eval_n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    take = min(n_rows, train_n + eval_n)
    idx = rng.choice(n_rows, size=take, replace=False)
    return idx[: min(train_n, take)], idx[min(train_n, take) :]


def add_stats_from_shards(
    *,
    paths: list[Path],
    train_tokens_per_shard: int,
    eval_tokens_per_shard: int,
    chunk_rows: int,
    seed: int,
    device: str,
) -> tuple[dict[str, np.ndarray | int], dict[str, list[np.ndarray]]]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)
    sum_x = sum_y = xtx = xty = yty = None
    n = 0
    eval_refs: dict[str, list[np.ndarray]] = {"path_index": [], "indices": []}

    for path_index, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        x = shard["x_attn"]
        y = shard["head_context"]
        flat_rows = x.shape[0] * x.shape[1]
        train_idx, eval_idx = sample_indices(flat_rows, train_tokens_per_shard, eval_tokens_per_shard, rng)
        eval_refs["path_index"].append(np.full(eval_idx.shape, path_index, dtype=np.int32))
        eval_refs["indices"].append(eval_idx.astype(np.int64))

        x_flat = x.reshape(flat_rows, x.shape[-1])
        y_flat = y.reshape(flat_rows, y.shape[-1])
        if sum_x is None:
            x_dim = x_flat.shape[-1]
            y_dim = y_flat.shape[-1]
            sum_x = np.zeros(x_dim, dtype=np.float64)
            sum_y = np.zeros(y_dim, dtype=np.float64)
            xtx = np.zeros((x_dim, x_dim), dtype=np.float64)
            xty = np.zeros((x_dim, y_dim), dtype=np.float64)
            yty = np.zeros((y_dim, y_dim), dtype=np.float64)

        for start in range(0, train_idx.size, chunk_rows):
            idx = train_idx[start : start + chunk_rows]
            xb = x_flat[idx].to(device=device, dtype=torch.float32)
            yb = y_flat[idx].to(device=device, dtype=torch.float32)
            sum_x += xb.sum(dim=0).cpu().numpy().astype(np.float64)
            sum_y += yb.sum(dim=0).cpu().numpy().astype(np.float64)
            xtx += (xb.T @ xb).cpu().numpy().astype(np.float64)
            xty += (xb.T @ yb).cpu().numpy().astype(np.float64)
            yty += (yb.T @ yb).cpu().numpy().astype(np.float64)
            n += int(xb.shape[0])

        print(f"stats shard={path_index} train_rows={train_idx.size} eval_rows={eval_idx.size} n={n}", flush=True)

    if sum_x is None or sum_y is None or xtx is None or xty is None or yty is None:
        raise RuntimeError("No training rows accumulated")
    eval_refs = {
        "path_index": [np.concatenate(eval_refs["path_index"])],
        "indices": [np.concatenate(eval_refs["indices"])],
    }
    return {"n": n, "sum_x": sum_x, "sum_y": sum_y, "xtx": xtx, "xty": xty, "yty": yty}, eval_refs


def evaluate_ranks(
    *,
    paths: list[Path],
    eval_refs: dict[str, list[np.ndarray]],
    b_full: np.ndarray,
    eigvecs: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    ranks: list[int],
    chunk_rows: int,
) -> list[dict[str, float | int]]:
    import torch

    refs_path = eval_refs["path_index"][0]
    refs_idx = eval_refs["indices"][0]
    coeffs = {rank: rrr_coefficients(b_full, eigvecs, rank=rank).astype(np.float32) for rank in ranks}
    metrics = {
        rank: {"sse": 0.0, "target_sse": 0.0, "dot": 0.0, "pred_norm": 0.0, "target_norm": 0.0, "n_elements": 0}
        for rank in ranks
    }

    for path_index, path in enumerate(paths):
        mask = refs_path == path_index
        if not np.any(mask):
            continue
        idx_all = refs_idx[mask]
        shard = torch.load(path, map_location="cpu")
        x = shard["x_attn"]
        y = shard["head_context"]
        flat_rows = x.shape[0] * x.shape[1]
        x_flat = x.reshape(flat_rows, x.shape[-1])
        y_flat = y.reshape(flat_rows, y.shape[-1])
        for start in range(0, idx_all.size, chunk_rows):
            idx = idx_all[start : start + chunk_rows]
            xb = x_flat[idx].float().numpy()
            yb = y_flat[idx].float().numpy()
            y_centered = yb - y_mean
            target_sse = float(np.square(y_centered).sum())
            for rank in ranks:
                pred = (xb - x_mean) @ coeffs[rank] + y_mean
                err = pred - yb
                centered_pred = pred - y_mean
                metrics[rank]["sse"] += float(np.square(err).sum())
                metrics[rank]["target_sse"] += target_sse
                metrics[rank]["dot"] += float((centered_pred * y_centered).sum())
                metrics[rank]["pred_norm"] += float(np.square(centered_pred).sum())
                metrics[rank]["target_norm"] += target_sse
                metrics[rank]["n_elements"] += int(yb.size)
        print(f"eval shard={path_index} rows={idx_all.size}", flush=True)

    results = []
    for rank in ranks:
        m = metrics[rank]
        mse = m["sse"] / m["n_elements"]
        rel_mse = m["sse"] / max(m["target_sse"], 1e-12)
        cosine = m["dot"] / max((m["pred_norm"] * m["target_norm"]) ** 0.5, 1e-12)
        results.append(
            {
                "rank": rank,
                "mse": mse,
                "relative_mse": rel_mse,
                "centered_cosine": cosine,
                "eval_elements": m["n_elements"],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    paths = load_capture_paths(args.capture_dir)
    ranks = [int(part) for part in args.ranks.split(",") if part.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw, eval_refs = add_stats_from_shards(
        paths=paths,
        train_tokens_per_shard=args.train_tokens_per_shard,
        eval_tokens_per_shard=args.eval_tokens_per_shard,
        chunk_rows=args.chunk_rows,
        seed=args.seed,
        device=args.device,
    )
    stats = centered_stats_from_raw(
        n=int(raw["n"]),
        sum_x=raw["sum_x"],
        sum_y=raw["sum_y"],
        xtx=raw["xtx"],
        xty=raw["xty"],
        yty=raw["yty"],
    )
    b_full, eigvecs, eigvals = fit_ridge_rrr(stats, ridge=args.ridge)
    results = evaluate_ranks(
        paths=paths,
        eval_refs=eval_refs,
        b_full=b_full,
        eigvecs=eigvecs,
        x_mean=stats.x_mean,
        y_mean=stats.y_mean,
        ranks=ranks,
        chunk_rows=args.chunk_rows,
    )

    np.savez_compressed(
        out / "rrr_model_stats.npz",
        x_mean=stats.x_mean.astype(np.float32),
        y_mean=stats.y_mean.astype(np.float32),
        b_full=b_full.astype(np.float32),
        eigvecs=eigvecs.astype(np.float32),
        eigvals=eigvals.astype(np.float32),
        ranks=np.array(ranks, dtype=np.int32),
    )
    report = {
        "capture_dir": str(args.capture_dir),
        "activation_shards": len(paths),
        "train_rows": int(raw["n"]),
        "eval_rows": int(eval_refs["indices"][0].size),
        "ridge": args.ridge,
        "ranks": ranks,
        "results": results,
    }
    with (out / "rrr_rank_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
