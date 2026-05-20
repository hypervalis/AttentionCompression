#!/usr/bin/env python3
"""PCA / covariance spectrum of captured ``ffn_input`` (script 27) for factorized FFN design."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def rank_for_variance(ratios: torch.Tensor, threshold: float) -> int:
    cum = torch.cumsum(ratios, dim=0)
    idx = int(torch.searchsorted(cum, threshold).item())
    return min(idx + 1, ratios.numel())


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.capture_dir)

    chunks: list[torch.Tensor] = []
    n = 0
    dim = None
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        x = shard["ffn_input"].reshape(-1, shard["ffn_input"].shape[-1]).float()
        if dim is None:
            dim = int(x.shape[-1])
        take = min(x.shape[0], args.max_tokens - n)
        if take <= 0:
            break
        chunks.append(x[:take])
        n += take

    if n < 2:
        raise RuntimeError("need at least 2 tokens")
    X = torch.cat(chunks, dim=0)
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (n - 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    evecs = evecs[:, order]
    total = evals.sum().clamp_min(1e-12)
    ratios = evals / total

    ranks = {f"var_{int(t * 100)}pct": rank_for_variance(ratios, t) for t in (0.9, 0.95, 0.99, 0.999)}
    k_ref = [256, 512, 768, 1024, 1536]
    param_est = {str(k): int(2 * dim * k + k * 4096 + k * dim) for k in k_ref}

    out = {
        "capture_dir": str(args.capture_dir),
        "tokens_used": n,
        "hidden_size": dim,
        "mean_norm": float(mean.norm()),
        "top_eigenvalues": evals[:16].tolist(),
        "ranks_for_variance": ranks,
        "reference_bottleneck_params": param_est,
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
