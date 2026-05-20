#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_compression.activations import (
    find_transformer_layers,
    load_selected_rows_by_bin,
    rows_to_token_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test OLMo layer activation hooks on selected windows.")
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--windows-per-bin", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    rows = load_selected_rows_by_bin(args.selected_csv, windows_per_bin=args.windows_per_bin)
    token_batch_np, bins = rows_to_token_batch(rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    layer_path, layers = find_transformer_layers(model)
    if args.target_layer < 0 or args.target_layer >= len(layers):
        raise ValueError(f"target_layer={args.target_layer} out of range for {len(layers)} layers")
    target_module = layers[args.target_layer]

    captures: list[dict[str, object]] = []

    def hook(_module, inputs, output):
        x = inputs[0]
        y = output[0] if isinstance(output, tuple) else output
        captures.append(
            {
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "output_shape": list(y.shape),
                "output_dtype": str(y.dtype),
                "input_device": str(x.device),
                "output_device": str(y.device),
            }
        )

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0, token_batch_np.shape[0], args.max_batch_size):
                batch_np = token_batch_np[start : start + args.max_batch_size]
                input_ids = torch.as_tensor(batch_np, dtype=torch.long, device=device)
                _ = model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    report = {
        "model_name": args.model_name,
        "device": device,
        "model_dtype": str(dtype),
        "layer_path": layer_path,
        "num_layers": len(layers),
        "target_layer": args.target_layer,
        "selected_windows": len(rows),
        "seq_len": int(token_batch_np.shape[1]),
        "rarity_bins": sorted(set(bins)),
        "bin_counts": {name: bins.count(name) for name in sorted(set(bins))},
        "captures": captures,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
