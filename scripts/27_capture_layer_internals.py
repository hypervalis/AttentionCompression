#!/usr/bin/env python3
"""Capture all attention head contexts and FFN inputs for one layer.

For a selected decoder layer, this records:

- ``head_contexts``: the tensor entering ``attn.o_proj``, reshaped as
  ``[batch, seq, num_heads, head_dim]``. Treats heads as black-box outputs.
- ``ffn_input``: the tensor entering the layer MLP/FFN, i.e. the normalized
  post-attention residual stream.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_compression.activations import (
    CaptureShardMetadata,
    find_transformer_layers,
    load_selected_rows_by_bin,
    rows_to_token_batch,
    write_capture_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture all head contexts and FFN input for a layer.")
    p.add_argument("--selected-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--windows-per-bin", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--samples-per-shard", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    rows = load_selected_rows_by_bin(args.selected_csv, windows_per_bin=args.windows_per_bin)
    token_batch_np, bins = rows_to_token_batch(rows)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.eval()
    model.to(device)
    layer_path, layers = find_transformer_layers(model)
    layer = layers[args.target_layer]
    attn = layer.self_attn
    head_dim = int(attn.head_dim)
    num_heads = int(getattr(attn, "num_heads", attn.o_proj.in_features // head_dim))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[CaptureShardMetadata] = []
    pending_heads: list[torch.Tensor] = []
    pending_ffn: list[torch.Tensor] = []
    pending_rows: list[dict[str, str]] = []
    shard_id = 0
    captured: dict[str, torch.Tensor] = {}

    def o_proj_pre_hook(_module, inputs):
        pre_o = inputs[0].detach().to("cpu")
        captured["head_contexts"] = pre_o.reshape(pre_o.shape[0], pre_o.shape[1], num_heads, head_dim)

    def mlp_pre_hook(_module, inputs):
        captured["ffn_input"] = inputs[0].detach().to("cpu")

    h1 = attn.o_proj.register_forward_pre_hook(o_proj_pre_hook)
    h2 = layer.mlp.register_forward_pre_hook(mlp_pre_hook)

    def flush() -> None:
        nonlocal shard_id
        if not pending_heads:
            return
        head_contexts = torch.cat(pending_heads, dim=0).contiguous()
        ffn_input = torch.cat(pending_ffn, dim=0).contiguous()
        path = out / f"layer_{args.target_layer:02d}_internals_{shard_id:06d}.pt"
        rarity_counts = Counter(row["rarity_bin"] for row in pending_rows)
        payload = {
            "head_contexts": head_contexts,
            "ffn_input": ffn_input,
            "rows": list(pending_rows),
            "target_layer": args.target_layer,
            "layer_path": layer_path,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        torch.save(payload, path)
        records.append(
            CaptureShardMetadata(
                shard_id=shard_id,
                path=str(path),
                sample_count=int(head_contexts.shape[0]),
                seq_len=int(head_contexts.shape[1]),
                hidden_size=int(ffn_input.shape[2]),
                dtype=str(ffn_input.dtype),
                target_layer=args.target_layer,
                rarity_bin_counts=dict(sorted(rarity_counts.items())),
            )
        )
        print(
            f"wrote shard_id={shard_id} samples={ffn_input.shape[0]} "
            f"heads_shape={tuple(head_contexts.shape)} ffn_shape={tuple(ffn_input.shape)} path={path}",
            flush=True,
        )
        shard_id += 1
        pending_heads.clear()
        pending_ffn.clear()
        pending_rows.clear()

    try:
        with torch.inference_mode():
            for start in range(0, token_batch_np.shape[0], args.batch_size):
                stop = min(token_batch_np.shape[0], start + args.batch_size)
                captured.clear()
                input_ids = torch.as_tensor(token_batch_np[start:stop], dtype=torch.long, device=device)
                _ = model(input_ids=input_ids, use_cache=False)
                pending_heads.append(captured["head_contexts"])
                pending_ffn.append(captured["ffn_input"])
                pending_rows.extend(rows[start:stop])
                if len(pending_rows) >= args.samples_per_shard:
                    flush()
    finally:
        h1.remove()
        h2.remove()
    flush()

    run_config = {
        "model_name": args.model_name,
        "selected_csv": args.selected_csv,
        "target_layer": args.target_layer,
        "layer_path": layer_path,
        "num_layers": len(layers),
        "num_heads": num_heads,
        "head_dim": head_dim,
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
    print(json.dumps(run_config, indent=2), flush=True)


if __name__ == "__main__":
    main()
