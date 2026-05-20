#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch
from attention_compression.attention_metrics import (
    attention_kl,
    causal_attention,
    causal_logit_relative_mse,
    cosine_similarity_mean,
    relative_mse,
    topk_overlap,
)
from attention_compression.joint_qkv import JointQKVBranches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one dense attention head to its low-rank replacement.")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--head-index", type=int, default=13)
    parser.add_argument("--q-rank", type=int, default=64)
    parser.add_argument("--k-rank", type=int, default=48)
    parser.add_argument("--v-rank", type=int, default=128)
    parser.add_argument("--keep-dense-v", action="store_true", help="Use the original dense V projection while reducing Q/K.")
    parser.add_argument("--windows-per-bin", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def capture_layer_input(model: torch.nn.Module, input_ids: torch.Tensor, *, layer: int, device: torch.device) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states") if kwargs else None
        if hidden is None and args:
            hidden = args[0]
        if hidden is None:
            raise RuntimeError("Could not capture attention hidden_states")
        captured.append(hidden.detach().float().cpu())

    handle = model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for row in input_ids:
            model(input_ids=row.unsqueeze(0).to(device), use_cache=False)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM
    from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    rows = load_selected_rows_by_bin(args.selected_csv, windows_per_bin=args.windows_per_bin)
    tokens, bins = rows_to_token_batch(rows)
    input_ids = torch.as_tensor(tokens, dtype=torch.long)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.to(device)
    teacher.eval()

    x_cpu = capture_layer_input(teacher, input_ids, layer=args.target_layer, device=device)
    attn = teacher.model.layers[args.target_layer].self_attn
    head_dim = int(attn.head_dim)
    hs = args.head_index * head_dim
    he = hs + head_dim
    projs = {
        "q": attn.q_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous(),
        "k": attn.k_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous(),
        "v": attn.v_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous(),
    }

    branch = JointQKVBranches(
        int(attn.q_proj.in_features),
        head_dim,
        args.q_rank,
        args.k_rank,
        args.v_rank,
    ).to(device)
    branch.load_state_dict(torch.load(args.checkpoint, map_location=device))
    branch.eval()

    seq_len = x_cpu.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    def rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids.expand(q.shape[0], -1)
        cos, sin = teacher.model.rotary_emb(q, pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(1), k.unsqueeze(1), cos, sin, unsqueeze_dim=1)
        return qr.squeeze(1), kr.squeeze(1)

    per_window: list[dict[str, float | str]] = []
    for i in range(x_cpu.shape[0]):
        x = x_cpu[i : i + 1].to(device=device, dtype=torch.float32)
        q = x @ projs["q"]
        k = x @ projs["k"]
        v = x @ projs["v"]
        q_low, k_low, v_branch = branch(x)
        v_low = v if args.keep_dense_v else v_branch
        qr, kr = rope(q, k)
        qlr, klr = rope(q_low, k_low)
        logits, probs, head = causal_attention(qr, kr, v)
        logits_low, probs_low, head_low = causal_attention(qlr, klr, v_low)
        per_window.append(
            {
                "rarity_bin": bins[i],
                "q_relative_mse": relative_mse(q_low, q),
                "k_relative_mse": relative_mse(k_low, k),
                "v_relative_mse": relative_mse(v_low, v),
                "logit_relative_mse": causal_logit_relative_mse(logits_low, logits),
                "attention_kl": attention_kl(probs, probs_low),
                "attention_top5_overlap": topk_overlap(probs, probs_low, 5),
                "head_context_relative_mse": relative_mse(head_low, head),
                "head_context_cosine": cosine_similarity_mean(head_low, head),
            }
        )

    metric_names = [k for k in per_window[0] if k != "rarity_bin"]
    summary = {name: summarize([float(row[name]) for row in per_window]) for name in metric_names}
    result = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "target_layer": args.target_layer,
        "head_index": args.head_index,
        "q_rank": args.q_rank,
        "k_rank": args.k_rank,
        "v_rank": args.v_rank,
        "keep_dense_v": args.keep_dense_v,
        "windows": len(per_window),
        "seq_len": seq_len,
        "rarity_bins": sorted(set(bins)),
        "summary": summary,
        "per_window": per_window,
        "rank_parameter_count": {
            "q": int(2048 * args.q_rank + args.q_rank * head_dim + head_dim),
            "k": int(2048 * args.k_rank + args.k_rank * head_dim + head_dim),
            "v": int(2048 * head_dim + head_dim) if args.keep_dense_v else int(2048 * args.v_rank + args.v_rank * head_dim + head_dim),
            "dense_per_branch": int(2048 * head_dim + head_dim),
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
