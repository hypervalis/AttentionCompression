#!/usr/bin/env python3
"""Benchmark inference latency: baseline OLMo vs fully patched Q/K-low-rank + dense-V.

Measures median wall time for:
  - Prefill: one causal forward over ``[batch, seq_len]`` with ``use_cache=False``.
  - Decode: after an untimed prefill with ``use_cache=True``, only the autoregressive
    loop of ``decode_steps`` single-token forwards (greedy argmax), matching typical
    generation body.

Requires torch + transformers; GPU recommended for meaningful numbers.

Example::

  python3 scripts/21_inference_speed_benchmark.py \\
    --checkpoint-root /path/to/qk_dense_v \\
    --fallback-joint-qkv-root /path/to/joint_qkv \\
    --output-json /tmp/bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
for _p in (str(_REPO_DIR), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import _qk_surgery_lib  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs patched inference speed benchmark.")
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--fallback-joint-qkv-root", default=None)
    p.add_argument("--fallback-joint-config", default="q64_k48_v128")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--prefill-batch", type=int, default=1)
    p.add_argument("--prefill-seq-len", type=int, default=1024)
    p.add_argument("--decode-prompt-len", type=int, default=512,
                   help="Tokens used for untimed prefill before timed decode loop.")
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--decode-batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-fused-qk",
        action="store_true",
        help="Use per-head Python matmul loop for patched Q/K (for A/B vs fused path).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model after patching (extra warmup).",
    )
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (PyTorch 2.x).",
    )
    p.add_argument(
        "--compile-static-shapes",
        action="store_true",
        help="Pass dynamic=False to torch.compile (may fail across prefill vs decode).",
    )
    p.add_argument(
        "--compile-enable-cudagraph",
        action="store_true",
        help="Keep inductor CUDAGraphs enabled (often breaks HF causal LM + KV decode).",
    )
    p.add_argument(
        "--compile-warmup",
        type=int,
        default=3,
        help="Extra compile warmup rounds before timed benchmarks (only with --compile).",
    )
    p.add_argument(
        "--materialize-dense-qk",
        action="store_true",
        help="At load time, fold each head's down@up into one nn.Linear per Q/K (VRAM=speed like dense baseline).",
    )
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def resolve_dtype(device: torch.device, spec: str) -> torch.dtype:
    if spec != "auto":
        return getattr(torch, spec)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def median_ms(seconds_list: list[float]) -> float:
    return 1000.0 * float(statistics.median(seconds_list))


def cudagraph_mark_step_begin() -> None:
    """Avoid CUDAGraph overwrite errors when ``torch.compile`` wraps HF decode."""
    fn = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(fn):
        fn()


def patch_all_layers(
    model: torch.nn.Module,
    *,
    checkpoint_root: Path,
    fallback_joint_root: Path | None,
    fallback_joint_config: str,
    dtype: torch.dtype,
    device: torch.device,
    materialize_dense_qk: bool = False,
) -> dict[str, str]:
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    sources: dict[str, str] = {}
    for layer in range(num_layers):
        if not _qk_surgery_lib.layer_is_complete(
            checkpoint_root,
            layer=layer,
            num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=fallback_joint_config,
        ):
            raise FileNotFoundError(f"Incomplete checkpoints for layer {layer}")
        states, _reports, source = _qk_surgery_lib.load_layer_qk_states(
            checkpoint_root,
            layer=layer,
            num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=fallback_joint_config,
        )
        _qk_surgery_lib.patch_layer_qk_dense_v(
            model,
            layer_index=layer,
            branch_states=states,
            dtype=dtype,
            device=device,
            materialize_dense=materialize_dense_qk,
        )
        sources[str(layer)] = source
    return sources


def set_patched_qk_fused(model: torch.nn.Module, *, use_fused: bool) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        for proj in (attn.q_proj, attn.k_proj):
            if isinstance(proj, _qk_surgery_lib.MultiHeadQKLowRankProjection):
                proj._force_naive_forward = not use_fused


@torch.inference_mode()
def compile_warmup_runs(
    model: torch.nn.Module,
    prefill_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    decode_steps: int,
    device: torch.device,
    rounds: int,
) -> None:
    for _ in range(rounds):
        cudagraph_mark_step_begin()
        model(input_ids=prefill_ids, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # Touch decode-shaped forwards (seq_len=1 + cache)
    cudagraph_mark_step_begin()
    out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:, :].argmax(dim=-1)
    for _ in range(min(decode_steps, 32)):
        cudagraph_mark_step_begin()
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:, :].argmax(dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def bench_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        cudagraph_mark_step_begin()
        model(input_ids=input_ids, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        model(input_ids=input_ids, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    total_tokens = input_ids.numel()
    med = median_ms(times)
    return {
        "median_ms_per_forward": med,
        "median_tokens_per_sec": total_tokens / (med / 1000.0),
        "batch": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
    }


@torch.inference_mode()
def bench_decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    decode_steps: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    """Time only the decode loop (single-token forwards with KV cache)."""

    def one_decode_run() -> float:
        cudagraph_mark_step_begin()
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:, :].argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            cudagraph_mark_step_begin()
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:, :].argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        one_decode_run()
    times = [one_decode_run() for _ in range(repeats)]
    med_s = float(statistics.median(times))
    return {
        "median_ms_total_decode": med_s * 1000.0,
        "median_ms_per_decode_step": (med_s / decode_steps) * 1000.0,
        "decode_steps": decode_steps,
        "prompt_len": int(prompt_ids.shape[1]),
        "batch": int(prompt_ids.shape[0]),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(device, args.dtype)

    if args.prefill_batch != args.decode_batch:
        raise ValueError("prefill-batch and decode-batch must match for this benchmark")
    B = args.prefill_batch

    # Random token ids in a safe range (avoid accidental special-token-only rows).
    gen_device = device if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(args.seed)
    prefill_ids = torch.randint(4, 32000, (B, args.prefill_seq_len), generator=gen, dtype=torch.long, device=device)
    prompt_len = min(args.decode_prompt_len, args.prefill_seq_len)
    prompt_ids = prefill_ids[:, :prompt_len].contiguous()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    baseline_prefill = bench_prefill(model, prefill_ids, device, args.warmup, args.repeats)
    baseline_decode = bench_decode(model, prompt_ids, args.decode_steps, device, args.warmup, args.repeats)

    root = Path(args.checkpoint_root)
    fb = Path(args.fallback_joint_qkv_root) if args.fallback_joint_qkv_root else None
    sources = patch_all_layers(
        model,
        checkpoint_root=root,
        fallback_joint_root=fb,
        fallback_joint_config=args.fallback_joint_config,
        dtype=dtype,
        device=device,
        materialize_dense_qk=args.materialize_dense_qk,
    )
    model.eval()
    set_patched_qk_fused(model, use_fused=not args.no_fused_qk and not args.materialize_dense_qk)

    compile_dynamic = not args.compile_static_shapes
    if args.compile:
        if not args.compile_enable_cudagraph:
            try:
                import torch._inductor.config as inductor_config  # type: ignore[attr-defined]

                # ``cudagraphs`` may already be False; trees mode still breaks HF + KV decode.
                inductor_config.triton.cudagraph_trees = False
                inductor_config.triton.cudagraphs = False
            except Exception:
                pass
        compile_kw: dict = {"dynamic": compile_dynamic}
        if args.compile_mode != "default":
            compile_kw["mode"] = args.compile_mode
        model = torch.compile(model, **compile_kw)
        model.eval()
        compile_warmup_runs(
            model, prefill_ids, prompt_ids, args.decode_steps, device, args.compile_warmup
        )

    patched_prefill = bench_prefill(model, prefill_ids, device, args.warmup, args.repeats)
    patched_decode = bench_decode(model, prompt_ids, args.decode_steps, device, args.warmup, args.repeats)

    result = {
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype),
        "checkpoint_root": str(root),
        "fallback_joint_qkv_root": str(fb) if fb else None,
        "checkpoint_source_by_layer": sources,
        "materialize_dense_qk": args.materialize_dense_qk,
        "patched_fused_qk_einsum": not args.no_fused_qk and not args.materialize_dense_qk,
        "torch_compile": args.compile,
        "torch_compile_mode": args.compile_mode if args.compile else None,
        "torch_compile_dynamic": compile_dynamic if args.compile else None,
        "torch_compile_inductor_cudagraphs": bool(args.compile_enable_cudagraph) if args.compile else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "baseline": {"prefill": baseline_prefill, "decode": baseline_decode},
        "patched_qk_dense_v": {"prefill": patched_prefill, "decode": patched_decode},
        "ratios": {
            "prefill_ms_per_forward": patched_prefill["median_ms_per_forward"] / baseline_prefill["median_ms_per_forward"],
            "decode_ms_per_step": patched_decode["median_ms_per_decode_step"]
            / baseline_decode["median_ms_per_decode_step"],
        },
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
