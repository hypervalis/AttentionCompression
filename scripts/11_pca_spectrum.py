#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from attention_compression.pca import covariance_from_raw, dims_for_thresholds, pca_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PCA explained-variance spectrum for captured tensors.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-key", default="head_context")
    parser.add_argument("--tokens-per-shard", type=int, default=16384)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--thresholds", default="0.5,0.75,0.8,0.9,0.95,0.99")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def load_capture_paths(capture_dir: str | Path) -> list[Path]:
    paths = sorted(Path(capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt activation shards found under {capture_dir}")
    return paths


def main() -> None:
    args = parse_args()
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    rng = np.random.default_rng(args.seed)
    paths = load_capture_paths(args.capture_dir)
    sum_x = xtx = None
    n = 0

    for path_index, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        x = shard[args.tensor_key]
        rows = x.shape[0] * x.shape[1]
        take = min(rows, args.tokens_per_shard)
        idx = rng.choice(rows, size=take, replace=False)
        flat = x.reshape(rows, x.shape[-1])
        if sum_x is None:
            dim = flat.shape[-1]
            sum_x = np.zeros(dim, dtype=np.float64)
            xtx = np.zeros((dim, dim), dtype=np.float64)
        for start in range(0, idx.size, args.chunk_rows):
            sub = idx[start : start + args.chunk_rows]
            xb = flat[sub].to(device=device, dtype=torch.float32)
            sum_x += xb.sum(dim=0).cpu().numpy().astype(np.float64)
            xtx += (xb.T @ xb).cpu().numpy().astype(np.float64)
            n += int(xb.shape[0])
        print(f"pca shard={path_index} sampled_rows={take} n={n}", flush=True)

    if sum_x is None or xtx is None:
        raise RuntimeError("No samples accumulated")
    mean, cov = covariance_from_raw(n=n, sum_x=sum_x, xtx=xtx)
    eigvals, cumulative = pca_spectrum(cov)
    thresholds = [float(part) for part in args.thresholds.split(",") if part.strip()]
    dims = dims_for_thresholds(cumulative, thresholds)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.tensor_key}_pca_spectrum.npz",
        mean=mean.astype(np.float32),
        eigenvalues=eigvals.astype(np.float32),
        cumulative_explained_variance=cumulative.astype(np.float32),
    )
    report = {
        "capture_dir": str(args.capture_dir),
        "tensor_key": args.tensor_key,
        "activation_shards": len(paths),
        "sampled_rows": n,
        "dimension": int(eigvals.size),
        "threshold_dims": dims,
        "top_eigenvalues": eigvals[:20].tolist(),
        "top_cumulative_explained_variance": cumulative[:20].tolist(),
    }
    with (out / f"{args.tensor_key}_pca_spectrum.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
