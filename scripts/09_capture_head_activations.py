#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_compression.activations import (
    CaptureShardMetadata,
    find_transformer_layers,
    load_selected_rows_by_bin,
    rows_to_token_batch,
    write_capture_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture shared attention input and one head target.")
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--head-index", type=int, default=0)
    parser.add_argument("--windows-per-bin", type=int, default=1666)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.samples_per_shard <= 0:
        raise ValueError("samples-per-shard must be positive")

    rows = load_selected_rows_by_bin(args.selected_csv, windows_per_bin=args.windows_per_bin)
    token_batch_np, bins = rows_to_token_batch(rows)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    layer_path, layers = find_transformer_layers(model)
    if args.target_layer < 0 or args.target_layer >= len(layers):
        raise ValueError(f"target_layer={args.target_layer} out of range for {len(layers)} layers")
    layer = layers[args.target_layer]
    attn = getattr(layer, "self_attn")
    head_dim = int(getattr(attn, "head_dim"))
    num_heads = int(getattr(attn, "num_heads", attn.o_proj.in_features // head_dim))
    hidden_size = int(attn.o_proj.in_features)
    if args.head_index < 0 or args.head_index >= num_heads:
        raise ValueError(f"head_index={args.head_index} out of range for {num_heads} heads")
    head_start = args.head_index * head_dim
    head_stop = head_start + head_dim

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[CaptureShardMetadata] = []
    pending_x: list[torch.Tensor] = []
    pending_head: list[torch.Tensor] = []
    pending_rows: list[dict[str, str]] = []
    shard_id = 0
    captured: dict[str, torch.Tensor] = {}

    def attn_pre_hook(_module, inputs, kwargs):
        hidden_states = inputs[0] if inputs else kwargs["hidden_states"]
        captured["x_attn"] = hidden_states.detach().to("cpu")

    def o_proj_pre_hook(_module, inputs):
        pre_o = inputs[0]
        captured["head_context"] = pre_o[..., head_start:head_stop].detach().to("cpu")

    attn_handle = attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
    o_proj_handle = attn.o_proj.register_forward_pre_hook(o_proj_pre_hook)

    def flush() -> None:
        nonlocal shard_id
        if not pending_x:
            return
        x_attn = torch.cat(pending_x, dim=0).contiguous()
        head_context = torch.cat(pending_head, dim=0).contiguous()
        path = out / f"layer_{args.target_layer:02d}_head_{args.head_index:02d}_activations_{shard_id:06d}.pt"
        rarity_counts = Counter(row["rarity_bin"] for row in pending_rows)
        payload = {
            "x_attn": x_attn,
            "head_context": head_context,
            "rows": list(pending_rows),
            "target_layer": args.target_layer,
            "head_index": args.head_index,
            "layer_path": layer_path,
            "target_kind": "pre_o_proj_head_context",
        }
        torch.save(payload, path)
        records.append(
            CaptureShardMetadata(
                shard_id=shard_id,
                path=str(path),
                sample_count=int(x_attn.shape[0]),
                seq_len=int(x_attn.shape[1]),
                hidden_size=int(x_attn.shape[2]),
                dtype=str(x_attn.dtype),
                target_layer=args.target_layer,
                rarity_bin_counts=dict(sorted(rarity_counts.items())),
            )
        )
        print(
            f"wrote shard_id={shard_id} samples={x_attn.shape[0]} "
            f"x_shape={tuple(x_attn.shape)} head_shape={tuple(head_context.shape)} path={path}",
            flush=True,
        )
        shard_id += 1
        pending_x.clear()
        pending_head.clear()
        pending_rows.clear()

    try:
        with torch.inference_mode():
            for start in range(0, token_batch_np.shape[0], args.batch_size):
                stop = min(token_batch_np.shape[0], start + args.batch_size)
                captured.clear()
                input_ids = torch.as_tensor(token_batch_np[start:stop], dtype=torch.long, device=device)
                _ = model(input_ids=input_ids, use_cache=False)
                pending_x.append(captured["x_attn"])
                pending_head.append(captured["head_context"])
                pending_rows.extend(rows[start:stop])
                if len(pending_rows) >= args.samples_per_shard:
                    flush()
    finally:
        attn_handle.remove()
        o_proj_handle.remove()
    flush()

    run_config = {
        "model_name": args.model_name,
        "selected_csv": args.selected_csv,
        "target_layer": args.target_layer,
        "head_index": args.head_index,
        "target_kind": "pre_o_proj_head_context",
        "layer_path": layer_path,
        "num_layers": len(layers),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "windows_per_bin": args.windows_per_bin,
        "batch_size": args.batch_size,
        "samples_per_shard": args.samples_per_shard,
        "device": device,
        "dtype": str(dtype),
        "selected_windows": len(rows),
        "seq_len": int(token_batch_np.shape[1]),
        "rarity_bins": sorted(set(bins)),
        "bin_counts": {name: bins.count(name) for name in sorted(set(bins))},
        "capture_shards": [asdict(record) for record in records],
    }
    write_capture_manifest(output_dir=out, records=records, run_config=run_config)
    print(json.dumps(run_config, indent=2))


if __name__ == "__main__":
    main()
