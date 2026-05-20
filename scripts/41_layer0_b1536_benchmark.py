#!/usr/bin/env python3
"""PPL smoke: baseline vs coupled b=1536 (o_proj + MLP replaced) on one layer."""
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
from attention_compression.bottleneck_layer import apply_coupled_b1536_layer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, default=Path("/mnt/sdb1/dolma-v1_6-sample"))
    p.add_argument("--artifact-dir", type=Path, default=None)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--windows-per-bin", type=int, default=1)
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
    art = args.artifact_dir or (
        args.checkpoints / f"layer{args.layer:02d}_coupled_b1536_5ep"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    csv = args.checkpoints / "selected_windows/selected_train_windows.csv"
    rows = load_selected_rows_by_bin(str(csv), windows_per_bin=args.windows_per_bin)
    input_ids = torch.as_tensor(rows_to_token_batch(rows)[0], dtype=torch.long)

    out: dict = {"model": "allenai/OLMo-1B-0724-hf", "layer": args.layer, "artifact_dir": str(art)}
    m0 = AutoModelForCausalLM.from_pretrained(out["model"], dtype=dtype, trust_remote_code=True).to(device).eval()
    out["baseline"] = perplexity(m0, input_ids, device)
    del m0
    base = out["baseline"]["perplexity"]

    m = AutoModelForCausalLM.from_pretrained(out["model"], dtype=dtype, trust_remote_code=True).to(device).eval()
    meta = apply_coupled_b1536_layer(
        m, layer_index=args.layer, artifact_dir=art, device=device, dtype=dtype, bottleneck=1536
    )
    out["apply"] = meta
    p = perplexity(m, input_ids, device)
    out["coupled_b1536"] = {**p, "perplexity_ratio_vs_baseline": p["perplexity"] / base}

    out_path = args.output_json or (args.checkpoints / f"layer{args.layer:02d}_b1536_benchmark.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
