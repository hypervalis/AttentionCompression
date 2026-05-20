#!/usr/bin/env python3
"""Estimate MLP operator spectra: Cov(x), E[J^T J], SVD(ridge B); save bases for block FFN."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from attention_compression.activations import find_transformer_layers
from attention_compression.mlp_operator import (
    accumulate_jacobian_gram,
    accumulate_moment_stats,
    analyze_operator_artifact,
    finalize_operator_stats,
    save_operator_artifact,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jacobian / linear operator SVD of layer MLP on captures.")
    p.add_argument("--internals-capture-dir", required=True)
    p.add_argument("--output-pt", type=Path, required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=8)
    p.add_argument("--max-moment-tokens", type=int, default=200_000)
    p.add_argument(
        "--max-jacobian-tokens",
        type=int,
        default=256,
        help="Tokens for per-token jacrev on CPU MLP copy.",
    )
    p.add_argument("--moment-batch-size", type=int, default=256)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(args.internals_capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.internals_capture_dir)

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.eval().to(device)
    _path, layers = find_transformer_layers(model)
    mlp = layers[args.target_layer].mlp.to(device=device, dtype=torch.float32)
    for p in mlp.parameters():
        p.requires_grad_(False)

    print(
        f"pass 1: moments on {device} (max {args.max_moment_tokens}, batch {args.moment_batch_size})...",
        flush=True,
    )
    moments, jac_batches = accumulate_moment_stats(
        paths,
        mlp,
        device,
        max_moment_tokens=args.max_moment_tokens,
        max_jacobian_tokens=args.max_jacobian_tokens,
        moment_batch_size=args.moment_batch_size,
        seed=args.seed,
    )

    print("copy MLP to CPU for Jacobian pass...", flush=True)
    mlp_cpu = copy.deepcopy(mlp).cpu().float()
    del mlp, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.max_jacobian_tokens > 0 and jac_batches:
        print(f"pass 2: Jacobian Gram on CPU (max {args.max_jacobian_tokens} tokens)...", flush=True)
        G, n_jac = accumulate_jacobian_gram(
            jac_batches, mlp_cpu, max_jacobian_tokens=args.max_jacobian_tokens
        )
    else:
        dim = int(moments["dim"])
        G, n_jac = torch.zeros(dim, dim), 0

    del mlp_cpu
    gc.collect()

    raw = finalize_operator_stats(moments, G, n_jac, ridge=args.ridge)
    report = analyze_operator_artifact(raw)
    report["model_name"] = args.model_name
    report["target_layer"] = args.target_layer
    report["internals_capture_dir"] = str(args.internals_capture_dir)
    report["max_moment_tokens"] = args.max_moment_tokens
    report["max_jacobian_tokens"] = args.max_jacobian_tokens
    report["ridge"] = args.ridge

    save_operator_artifact(args.output_pt, raw, report=report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
