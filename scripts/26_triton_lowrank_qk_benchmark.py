#!/usr/bin/env python3
"""Benchmark a Triton fused per-head low-rank projection kernel.

This is a prototype for keeping the working math

    out_h = x @ down_h @ up_h + bias_h

without materializing ``down_h @ up_h`` as dense Q/K weights. It compares:

- dense materialized linear (speed target, dense VRAM),
- current packed PyTorch low-rank path,
- Triton per-head fused low-rank kernel.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _lowrank_head_kernel(
    x_ptr,
    down_ptr,
    up_ptr,
    bias_ptr,
    out_ptr,
    n_tokens: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_heads: tl.constexpr,
    rank: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)
    offs_o = tl.arange(0, head_dim)

    z = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for d0 in range(0, hidden_dim, BLOCK_D):
        d = d0 + offs_d
        x = tl.load(
            x_ptr + offs_m[:, None] * hidden_dim + d[None, :],
            mask=(offs_m[:, None] < n_tokens) & (d[None, :] < hidden_dim),
            other=0.0,
        )
        down = tl.load(
            down_ptr + pid_h * hidden_dim * rank + d[:, None] * rank + offs_r[None, :],
            mask=(d[:, None] < hidden_dim) & (offs_r[None, :] < rank),
            other=0.0,
        )
        z += tl.dot(x, down)

    up = tl.load(
        up_ptr + pid_h * rank * head_dim + offs_r[:, None] * head_dim + offs_o[None, :],
        mask=offs_r[:, None] < rank,
        other=0.0,
    ).to(tl.float32)
    y = tl.dot(z, up)
    b = tl.load(bias_ptr + pid_h * head_dim + offs_o)
    y += b[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * (num_heads * head_dim) + pid_h * head_dim + offs_o[None, :],
        y,
        mask=offs_m[:, None] < n_tokens,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Triton low-rank Q/K projection benchmark.")
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--block-m", type=int, default=16)
    p.add_argument("--block-d", type=int, default=64)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
    return p.parse_args()


def bench(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000.0 * statistics.median(times)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    H, D, R, HD = args.num_heads, args.hidden_dim, args.rank, args.head_dim

    down = torch.randn(H, D, R, device=device, dtype=dtype) / (D**0.5)
    up = torch.randn(H, R, HD, device=device, dtype=dtype) / (R**0.5)
    bias = torch.randn(H, HD, device=device, dtype=dtype)
    down_cat = down.permute(1, 0, 2).reshape(D, H * R).contiguous()
    up_stack = up.contiguous()
    bias_flat = bias.reshape(H * HD).contiguous()
    w_dense = torch.einsum("hdr,hro->hdo", down.float(), up.float()).to(dtype)
    w_dense = w_dense.permute(1, 0, 2).reshape(D, H * HD).contiguous()

    results = []
    for n_tokens in args.tokens:
        x = torch.randn(n_tokens, D, device=device, dtype=dtype)
        out_tri = torch.empty(n_tokens, H * HD, device=device, dtype=dtype)

        def torch_lowrank():
            z = x @ down_cat
            z = z.reshape(n_tokens, H, R)
            y = torch.einsum("nhr,hro->nho", z, up_stack)
            return y.reshape(n_tokens, H * HD) + bias_flat

        def dense():
            return x @ w_dense + bias_flat

        def triton_lowrank():
            _lowrank_head_kernel[(triton.cdiv(n_tokens, args.block_m), H)](
                x,
                down,
                up,
                bias,
                out_tri,
                n_tokens,
                D,
                H,
                R,
                HD,
                BLOCK_M=args.block_m,
                BLOCK_D=args.block_d,
                BLOCK_R=triton.next_power_of_2(R),
                num_warps=4,
            )
            return out_tri

        y_ref = torch_lowrank()
        y_tri = triton_lowrank()
        max_err = float((y_ref.float() - y_tri.float()).abs().max().item())
        tol = 3e-1 if dtype in (torch.bfloat16, torch.float16) else 1e-3
        if max_err > tol:
            print(f"WARNING n={n_tokens}: Triton max_err={max_err:.4f} > tol={tol}", flush=True)

        dense_ms = bench(dense, warmup=args.warmup, repeats=args.repeats)
        torch_ms = bench(torch_lowrank, warmup=args.warmup, repeats=args.repeats)
        triton_ms = bench(triton_lowrank, warmup=args.warmup, repeats=args.repeats)
        row = {
            "tokens": n_tokens,
            "dense_materialized_ms": dense_ms,
            "torch_packed_lowrank_ms": torch_ms,
            "triton_lowrank_ms": triton_ms,
            "triton_vs_dense": triton_ms / dense_ms,
            "triton_vs_torch_lowrank": triton_ms / torch_ms,
            "max_abs_err_vs_torch_lowrank": max_err,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    print(json.dumps({"config": vars(args), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
