#!/usr/bin/env python3
"""PPL smoke: baseline vs block-factorized MLP on one layer."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch
from attention_compression.block_factorized_ffn import apply_block_factorized_ffn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, default=Path("/mnt/sdb1/dolma-v1_6-sample"))
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--windows-per-bin", type=int, default=6)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def perplexity(model, input_ids, device):
    losses, tokens = [], 0
    for i in range(input_ids.shape[0]):
        batch = input_ids[i : i + 1].to(device)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        n = batch.numel() - batch.shape[0]
        losses.append(float(out.loss.detach()) * n)
        tokens += n
    mean_loss = sum(losses) / tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "windows": int(input_ids.shape[0])}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    csv = args.checkpoints / "selected_windows/selected_train_windows.csv"
    rows = load_selected_rows_by_bin(str(csv), windows_per_bin=args.windows_per_bin)
    input_ids = torch.as_tensor(rows_to_token_batch(rows)[0], dtype=torch.long)

    model_name = "allenai/OLMo-1B-0724-hf"
    out: dict = {
        "model": model_name,
        "layer": args.layer,
        "artifact_dir": str(args.artifact_dir),
    }
    m0 = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True).to(device).eval()
    out["baseline"] = perplexity(m0, input_ids, device)
    del m0
    base = out["baseline"]["perplexity"]

    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True).to(device).eval()
    meta = apply_block_factorized_ffn(
        m, layer_index=args.layer, artifact_dir=args.artifact_dir, device=device, dtype=dtype
    )
    out["apply"] = meta
    p = perplexity(m, input_ids, device)
    out["block_factorized_ffn"] = {**p, "perplexity_ratio_vs_baseline": p["perplexity"] / base}

    suffix = meta.get("merge_mode", "unknown")
    out_path = args.output_json or (
        args.checkpoints / f"layer{args.layer:02d}_block_ffn_{suffix}_benchmark.json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
