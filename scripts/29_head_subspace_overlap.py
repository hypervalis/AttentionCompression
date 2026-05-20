#!/usr/bin/env python3
"""Compare attention-head PCA subspaces after each head's o_proj slice.

Raw head-context coordinates are private to each head. This script lifts each
head context through its own ``o_proj`` block, then compares the resulting PCA
subspaces in the shared residual/embedding space.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from attention_compression.activations import find_transformer_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare shared residual-space head PCA dimensions.")
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--ranks", default="32,64,96,112")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return normalized projection overlap in [0, 1] for two orthonormal bases."""
    k = min(a.shape[1], b.shape[1])
    return float((a.T @ b).pow(2).sum().item() / max(k, 1))


def summarize_pairs(matrix: list[list[float]]) -> dict[str, object]:
    vals = []
    pairs = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i < j:
                vals.append(value)
                pairs.append({"heads": [i, j], "overlap": value})
    vals_t = torch.tensor(vals)
    pairs_sorted = sorted(pairs, key=lambda x: x["overlap"], reverse=True)
    return {
        "min": float(vals_t.min().item()),
        "median": float(vals_t.median().item()),
        "mean": float(vals_t.mean().item()),
        "max": float(vals_t.max().item()),
        "top_pairs": pairs_sorted[:10],
        "bottom_pairs": pairs_sorted[-10:],
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No capture shards found in {args.capture_dir}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, trust_remote_code=True)
    model.eval().to(device)
    _layer_path, layers = find_transformer_layers(model)
    attn = layers[args.target_layer].self_attn
    o_weight = attn.o_proj.weight.detach().to(device=device, dtype=torch.float32)

    first = torch.load(paths[0], map_location="cpu")
    num_heads = int(first["head_contexts"].shape[2])
    head_dim = int(first["head_contexts"].shape[3])
    hidden_size = int(o_weight.shape[0])

    n = torch.zeros(num_heads, device=device)
    sums = torch.zeros(num_heads, head_dim, device=device)
    xtx = torch.zeros(num_heads, head_dim, head_dim, device=device)
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        h = shard["head_contexts"].to(device=device, dtype=torch.float32)
        flat = h.reshape(-1, num_heads, head_dim)
        n += flat.shape[0]
        sums += flat.sum(dim=0)
        xtx += torch.einsum("nhd,nhe->hde", flat, flat)

    mean = sums / n[:, None]
    cov = (xtx - n[:, None, None] * torch.einsum("hd,he->hde", mean, mean)) / (n[:, None, None] - 1).clamp_min(1)

    bases_by_rank: dict[int, list[torch.Tensor]] = defaultdict(list)
    spectra = []
    for head in range(num_heads):
        vals, vecs = torch.linalg.eigh(cov[head])
        order = torch.argsort(vals, descending=True)
        vals = vals[order].clamp_min(0)
        vecs = vecs[:, order]
        total = vals.sum().clamp_min(1e-12)
        csum = torch.cumsum(vals, dim=0) / total
        w_h = o_weight[:, head * head_dim : (head + 1) * head_dim]
        # Left singular vectors of W_h @ C_ctx^0.5 are the exact PCA basis for
        # the head contribution in residual space, up to the retained ctx rank.
        weighted = w_h @ (vecs * vals.sqrt().unsqueeze(0))
        u_out, s_out, _ = torch.linalg.svd(weighted, full_matrices=False)
        out_var = s_out.pow(2)
        out_csum = torch.cumsum(out_var, dim=0) / out_var.sum().clamp_min(1e-12)
        spectra.append(
            {
                "head": head,
                "raw_head_rank_for_variance": {
                    str(frac): int(torch.searchsorted(csum.cpu(), torch.tensor(frac)).item() + 1)
                    for frac in (0.9, 0.95, 0.99)
                },
                "o_projected_rank_for_variance": {
                    str(frac): int(torch.searchsorted(out_csum.cpu(), torch.tensor(frac)).item() + 1)
                    for frac in (0.9, 0.95, 0.99)
                },
            }
        )
        for rank in ranks:
            bases_by_rank[rank].append(u_out[:, : min(rank, u_out.shape[1])].contiguous())

    overlap_reports = {}
    for rank, bases in bases_by_rank.items():
        matrix = []
        for i in range(num_heads):
            row = []
            for j in range(num_heads):
                row.append(1.0 if i == j else subspace_overlap(bases[i], bases[j]))
            matrix.append(row)
        overlap_reports[str(rank)] = {
            "matrix": matrix,
            "summary": summarize_pairs(matrix),
        }

    report = {
        "capture_dir": args.capture_dir,
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "ranks": ranks,
        "spectra": spectra,
        "overlaps": overlap_reports,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "head_subspace_overlap_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
