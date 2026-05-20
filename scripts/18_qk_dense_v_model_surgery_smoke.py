#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import torch

import attention_compression.qk_surgery as qklib  # noqa: E402

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    if not layers:
        raise ValueError("No layers requested")
    bad = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(f"Layer index out of range for {num_layers} layers: {bad}")
    return sorted(set(layers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch OLMo with Q/K-low-rank + dense-V branches and compare loss.")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing layer_XX_head_YY_q64_k48_densev outputs.")
    parser.add_argument(
        "--fallback-joint-qkv-root",
        default=None,
        help="Optional root containing joint_qkv_model.pt checkpoints; q/k branches are used when QK+dense-V is missing.",
    )
    parser.add_argument("--fallback-joint-config", default="q64_k48_v128")
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--layers", default="all", help="Comma-separated layer indices, or 'all'.")
    parser.add_argument("--windows-per-bin", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-missing-layers",
        action="store_true",
        help="Skip layers that do not have all head checkpoints instead of failing.",
    )
    parser.add_argument(
        "--materialize-dense-qk",
        action="store_true",
        help="Fold low-rank factors into standard nn.Linear per Q/K for inference (dense GEMM path).",
    )
    return parser.parse_args()


def iter_batches(input_ids: torch.Tensor, batch_size: int):
    for start in range(0, input_ids.shape[0], batch_size):
        yield input_ids[start : start + batch_size]


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, input_ids: torch.Tensor, *, batch_size: int, device: torch.device) -> dict[str, float]:
    losses = []
    token_counts = []
    for batch in iter_batches(input_ids, batch_size):
        batch = batch.to(device=device)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        tokens = batch.numel() - batch.shape[0]
        losses.append(float(out.loss.detach().cpu()) * tokens)
        token_counts.append(tokens)
    total_tokens = sum(token_counts)
    loss = sum(losses) / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": total_tokens}


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    rows = load_selected_rows_by_bin(args.selected_csv, windows_per_bin=args.windows_per_bin)
    tokens, bins = rows_to_token_batch(rows)
    input_ids = torch.as_tensor(tokens, dtype=torch.long)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    requested_layers = parse_layers(args.layers, num_layers)
    root = Path(args.checkpoint_root)
    fallback_joint_root = Path(args.fallback_joint_qkv_root) if args.fallback_joint_qkv_root else None

    complete_layers = [
        layer
        for layer in requested_layers
        if qklib.layer_is_complete(
            root,
            layer=layer,
            num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=args.fallback_joint_config,
        )
    ]
    missing_layers = [layer for layer in requested_layers if layer not in complete_layers]
    if missing_layers and not args.allow_missing_layers:
        raise FileNotFoundError(
            "Missing complete QK+dense-V checkpoints for layers "
            f"{missing_layers}. Pass --allow-missing-layers to patch only complete layers."
        )
    if not complete_layers:
        raise FileNotFoundError("No complete QK+dense-V layers found to patch.")

    baseline = evaluate_loss(model, input_ids, batch_size=args.batch_size, device=device)

    reports_by_layer: dict[str, list[dict[str, object]]] = {}
    checkpoint_source_by_layer: dict[str, str] = {}
    dense_qkv_total = 0
    compressed_qk_dense_v_total = 0
    for layer in complete_layers:
        states, reports, source = qklib.load_layer_qk_states(
            root,
            layer=layer,
            num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=args.fallback_joint_config,
        )
        qklib.patch_layer_qk_dense_v(
            model,
            layer_index=layer,
            branch_states=states,
            dtype=dtype,
            device=device,
            materialize_dense=args.materialize_dense_qk,
        )
        reports_by_layer[str(layer)] = reports
        checkpoint_source_by_layer[str(layer)] = source
        head_dim = int(model.model.layers[layer].self_attn.head_dim)
        for state in states:
            counts = qklib.state_parameter_counts(state, head_dim=head_dim)
            dense_qkv_total += counts["dense_qkv"]
            compressed_qk_dense_v_total += counts["qk_low_rank_dense_v"]
    model.eval()
    patched = evaluate_loss(model, input_ids, batch_size=args.batch_size, device=device)

    result = {
        "model_name": args.model_name,
        "checkpoint_root": str(root),
        "requested_layers": requested_layers,
        "patched_layers": complete_layers,
        "missing_layers": missing_layers,
        "checkpoint_source_by_layer": checkpoint_source_by_layer,
        "materialize_dense_qk": args.materialize_dense_qk,
        "patched_heads_per_layer": num_heads,
        "windows": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
        "rarity_bins": sorted(set(bins)),
        "baseline": baseline,
        "patched": patched,
        "loss_delta": patched["loss"] - baseline["loss"],
        "perplexity_ratio": patched["perplexity"] / baseline["perplexity"],
        "parameter_count": {
            "patched_dense_qkv": dense_qkv_total,
            "patched_qk_low_rank_dense_v": compressed_qk_dense_v_total,
            "reduction_fraction_on_patched_qkv": (
                1.0 - compressed_qk_dense_v_total / dense_qkv_total if dense_qkv_total else None
            ),
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
