#!/usr/bin/env python3
"""Side-by-side baseline vs Q/K-low-rank + dense-V model comparison.

For each prompt, runs greedy and sampled continuations with the original OLMo
checkpoint and again after patching every layer's Q and K projections with the
trained low-rank branches (V stays dense). Optionally also reports next-token
loss / perplexity on a window-level eval set, side by side.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_src = str(_REPO_DIR / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch

import attention_compression.qk_surgery as _qk_surgery_lib  # noqa: E402


DEFAULT_PROMPTS: list[str] = [
    "The capital city of France is",
    "In one short sentence, define entropy:",
    "List three reasons solar energy is useful.\n1.",
    "Translate to French: \"The library closes at nine.\"\nFrench:",
    "Q: What is 17 * 24?\nA:",
    "Once upon a time, deep in the forest,",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs Q/K-low-rank+dense-V outputs.")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--fallback-joint-qkv-root", default=None)
    parser.add_argument("--fallback-joint-config", default="q64_k48_v128")
    parser.add_argument("--selected-csv", default=None,
                        help="Optional eval CSV (selected_eval_windows.csv) for loss/perplexity comparison.")
    parser.add_argument("--windows-per-bin", type=int, default=0,
                        help="Per-bin eval windows; 0 disables loss eval.")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--prompts-file", default=None,
                        help="Optional newline-separated prompts file (overrides defaults).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--materialize-dense-qk",
        action="store_true",
        help="Fold low-rank Q/K into nn.Linear at load (same math, dense GEMM inference path).",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        prompts = [p.rstrip("\n") for p in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()]
        return [p for p in prompts if p.strip()]
    return list(DEFAULT_PROMPTS)


@torch.no_grad()
def generate_pair(model, tokenizer, prompts: list[str], *, device: torch.device,
                  max_new_tokens: int, temperature: float, top_p: float, seed: int) -> list[dict]:
    out: list[dict] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        torch.manual_seed(seed)
        greedy = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.manual_seed(seed)
        sampled = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_token_count = int(encoded["input_ids"].shape[1])
        out.append({
            "prompt": prompt,
            "prompt_tokens": prompt_token_count,
            "greedy": tokenizer.decode(greedy[0], skip_special_tokens=True),
            "greedy_continuation": tokenizer.decode(greedy[0][prompt_token_count:], skip_special_tokens=True),
            "sampled": tokenizer.decode(sampled[0], skip_special_tokens=True),
            "sampled_continuation": tokenizer.decode(sampled[0][prompt_token_count:], skip_special_tokens=True),
        })
    return out


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.decode([0])

    prompts = load_prompts(args)
    print(f"Using {len(prompts)} prompts", flush=True)

    eval_input_ids: torch.Tensor | None = None
    eval_bins: list[str] = []
    if args.selected_csv and args.windows_per_bin > 0:
        rows = load_selected_rows_by_bin(args.selected_csv, windows_per_bin=args.windows_per_bin)
        tokens, eval_bins = rows_to_token_batch(rows)
        eval_input_ids = torch.as_tensor(tokens, dtype=torch.long)
        print(f"Eval set: {eval_input_ids.shape[0]} windows, seq_len={eval_input_ids.shape[1]}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    baseline_outputs = generate_pair(
        model, tokenizer, prompts, device=device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
    )

    baseline_eval = None
    if eval_input_ids is not None:
        baseline_eval = _qk_surgery_lib.evaluate_loss(model, eval_input_ids, batch_size=args.eval_batch_size, device=device)

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    root = Path(args.checkpoint_root)
    fallback_joint_root = Path(args.fallback_joint_qkv_root) if args.fallback_joint_qkv_root else None

    requested_layers = list(range(num_layers))
    complete_layers = [
        layer for layer in requested_layers
        if _qk_surgery_lib.layer_is_complete(
            root, layer=layer, num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=args.fallback_joint_config,
        )
    ]
    if len(complete_layers) != num_layers:
        missing = [layer for layer in requested_layers if layer not in complete_layers]
        raise FileNotFoundError(f"Cannot patch full model; missing layers: {missing}")

    checkpoint_source_by_layer: dict[str, str] = {}
    dense_qkv_total = 0
    compressed_qk_dense_v_total = 0
    for layer in complete_layers:
        states, _reports, source = _qk_surgery_lib.load_layer_qk_states(
            root, layer=layer, num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=args.fallback_joint_config,
        )
        _qk_surgery_lib.patch_layer_qk_dense_v(
            model,
            layer_index=layer,
            branch_states=states,
            dtype=dtype,
            device=device,
            materialize_dense=args.materialize_dense_qk,
        )
        checkpoint_source_by_layer[str(layer)] = source
        head_dim = int(model.model.layers[layer].self_attn.head_dim)
        for state in states:
            counts = _qk_surgery_lib.state_parameter_counts(state, head_dim=head_dim)
            dense_qkv_total += counts["dense_qkv"]
            compressed_qk_dense_v_total += counts["qk_low_rank_dense_v"]
    model.eval()

    patched_outputs = generate_pair(
        model, tokenizer, prompts, device=device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
    )

    patched_eval = None
    if eval_input_ids is not None:
        patched_eval = _qk_surgery_lib.evaluate_loss(model, eval_input_ids, batch_size=args.eval_batch_size, device=device)

    pairs = []
    for base, patched in zip(baseline_outputs, patched_outputs, strict=True):
        pairs.append({
            "prompt": base["prompt"],
            "prompt_tokens": base["prompt_tokens"],
            "baseline_greedy_continuation": base["greedy_continuation"],
            "patched_greedy_continuation": patched["greedy_continuation"],
            "baseline_sampled_continuation": base["sampled_continuation"],
            "patched_sampled_continuation": patched["sampled_continuation"],
        })

    result = {
        "model_name": args.model_name,
        "checkpoint_root": str(root),
        "materialize_dense_qk": args.materialize_dense_qk,
        "fallback_joint_qkv_root": str(fallback_joint_root) if fallback_joint_root else None,
        "checkpoint_source_by_layer": checkpoint_source_by_layer,
        "patched_layers": complete_layers,
        "patched_heads_per_layer": num_heads,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "parameter_count": {
            "patched_dense_qkv": dense_qkv_total,
            "patched_qk_low_rank_dense_v": compressed_qk_dense_v_total,
            "reduction_fraction_on_patched_qkv": (
                1.0 - compressed_qk_dense_v_total / dense_qkv_total if dense_qkv_total else None
            ),
        },
        "baseline_eval": baseline_eval,
        "patched_eval": patched_eval,
        "loss_delta": (
            patched_eval["loss"] - baseline_eval["loss"]
            if baseline_eval and patched_eval else None
        ),
        "perplexity_ratio": (
            patched_eval["perplexity"] / baseline_eval["perplexity"]
            if baseline_eval and patched_eval else None
        ),
        "comparisons": pairs,
        "eval_rarity_bins": sorted(set(eval_bins)) if eval_bins else None,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n===== SIDE BY SIDE =====", flush=True)
    for entry in pairs:
        print("\nPROMPT:", entry["prompt"], flush=True)
        print("  BASELINE  (greedy):", entry["baseline_greedy_continuation"], flush=True)
        print("  PATCHED   (greedy):", entry["patched_greedy_continuation"], flush=True)
        print("  BASELINE (sampled):", entry["baseline_sampled_continuation"], flush=True)
        print("  PATCHED  (sampled):", entry["patched_sampled_continuation"], flush=True)
    if baseline_eval and patched_eval:
        print("\n===== EVAL =====", flush=True)
        print(json.dumps({
            "baseline": baseline_eval,
            "patched": patched_eval,
            "loss_delta": result["loss_delta"],
            "perplexity_ratio": result["perplexity_ratio"],
        }, indent=2), flush=True)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
