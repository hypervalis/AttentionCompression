#!/usr/bin/env python3
"""Layer-0 PPL: baseline vs swapped mimic o_proj vs FFN block (with swap)."""
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
from attention_compression.compress.apply import apply_plan
from attention_compression.compress.targets import CompressionPlan, CompressionTarget, FfnLayerJob, expand_plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, default=Path("/mnt/sdb1/dolma-v1_6-sample"))
    p.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("/mnt/sdb1/dolma-v1_6-sample/selected_windows/selected_train_windows.csv"),
    )
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--windows-per-bin", type=int, default=1)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def perplexity(model: torch.nn.Module, input_ids: torch.Tensor, device: torch.device) -> dict:
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

    rows = load_selected_rows_by_bin(str(args.selected_csv), windows_per_bin=args.windows_per_bin)
    input_ids = torch.as_tensor(rows_to_token_batch(rows)[0], dtype=torch.long)

    out: dict = {"model": args.model_name, "layer": args.layer, "baseline": None}

    m0 = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True).to(device).eval()
    out["baseline"] = perplexity(m0, input_ids, device)
    del m0
    base_ppl = out["baseline"]["perplexity"]

    configs = [
        ("oproj_swap", CompressionTarget.OPROJ, False),
        ("ffn_swap", CompressionTarget.FFN, False),
        ("ffn_hook_only", CompressionTarget.FFN, True),
    ]
    for name, target, keep_teacher in configs:
        plan = expand_plan(
            target=target,
            layer=args.layer,
            head=None,
            layers_spec="all",
            num_layers=16,
            num_heads_per_layer=16,
        )
        m = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True).to(device).eval()
        apply_plan(
            m,
            plan,
            checkpoint_root=args.checkpoints,
            device=device,
            dtype=dtype,
            fallback_joint_root=None,
            fallback_joint_config="q64_k48_v128",
            materialize_dense_qk=False,
            skip_missing=False,
            compression_target=target,
            swap_o_proj=not keep_teacher,
            keep_teacher_oproj_hook_only=keep_teacher,
        )
        p = perplexity(m, input_ids, device)
        out[name] = {**p, "perplexity_ratio_vs_baseline": p["perplexity"] / base_ppl}
        del m

    out_path = args.output_json or (
        args.checkpoints / f"layer{args.layer:02d}_oproj_swap_benchmark.json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
