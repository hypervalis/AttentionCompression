#!/usr/bin/env python3
"""Lightly fine-tune the patched Q/K-low-rank + dense-V model on the train windows.

Loads OLMo, patches every layer with the trained Q/K low-rank branches (V stays
dense), freezes everything except those branches, then runs a short language
modeling fine-tune on the selected train windows. The fine-tuned per-head Q/K
state dicts are written in the same layout as ``qk_dense_v`` so the existing
``19_qk_dense_v_compare_outputs.py`` script can be re-pointed at the new root.

Example::

  python3 scripts/20_qk_dense_v_finetune.py \
    --checkpoint-root /mnt/sdb1/dolma-v1_6-sample/qk_dense_v \
    --fallback-joint-qkv-root /mnt/sdb1/dolma-v1_6-sample/joint_qkv \
    --train-csv .../selected_train_windows.csv \
    --eval-csv  .../selected_eval_windows.csv \
    --output-dir /mnt/sdb1/dolma-v1_6-sample/qk_dense_v_finetuned_v1 \
    --train-windows-per-bin 64 --eval-windows-per-bin 32 \
    --steps 400 --batch-size 2 --lr 5e-5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
for _p in (str(_REPO_DIR), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch

import _qk_surgery_lib  # noqa: E402  (script-local shared module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightly fine-tune patched Q/K low-rank + dense-V branches.")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--fallback-joint-qkv-root", default=None)
    parser.add_argument("--fallback-joint-config", default="q64_k48_v128")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--train-windows-per-bin", type=int, default=64)
    parser.add_argument("--eval-windows-per-bin", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200,
                        help="Run an eval pass every N optimizer steps; <=0 only evals at start and end.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory mirroring qk_dense_v layout (per-(layer,head) qk_dense_v_model.pt).")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_trainable_qk(model: torch.nn.Module, *, patched_layers: list[int]) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for layer_index in patched_layers:
        attn = model.model.layers[layer_index].self_attn
        for proj in (attn.q_proj, attn.k_proj):
            for param in proj.parameters():
                # Cast trainable low-rank params to fp32 for optimizer stability.
                param.data = param.data.to(torch.float32)
                param.requires_grad_(True)
                trainable.append(param)
    return trainable


def iter_train_batches(input_ids: torch.Tensor, *, batch_size: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(input_ids.shape[0], generator=gen).tolist()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < batch_size:
                continue
            yield input_ids[idx]


def save_finetuned_layout(
    *,
    model: torch.nn.Module,
    output_dir: Path,
    patched_layers: list[int],
    num_heads: int,
    config_name: str = "q64_k48_densev",
) -> dict[str, list[str]]:
    written: dict[str, list[str]] = {}
    for layer_index in patched_layers:
        attn = model.model.layers[layer_index].self_attn
        q_proj = attn.q_proj
        k_proj = attn.k_proj
        layer_paths: list[str] = []
        for head_index in range(num_heads):
            state = {
                "q.down": q_proj.down[head_index].detach().to(torch.float32).cpu().contiguous(),
                "q.up": q_proj.up[head_index].detach().to(torch.float32).cpu().contiguous(),
                "q.bias": q_proj.bias[head_index].detach().to(torch.float32).cpu().contiguous(),
                "k.down": k_proj.down[head_index].detach().to(torch.float32).cpu().contiguous(),
                "k.up": k_proj.up[head_index].detach().to(torch.float32).cpu().contiguous(),
                "k.bias": k_proj.bias[head_index].detach().to(torch.float32).cpu().contiguous(),
            }
            head_dir = output_dir / f"layer_{layer_index:02d}_head_{head_index:02d}_{config_name}"
            head_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, head_dir / "qk_dense_v_model.pt")
            head_dim = int(state["q.up"].shape[1])
            input_dim = int(state["q.down"].shape[0])
            counts = {
                "dense_qkv": int(3 * (input_dim * head_dim + head_dim)),
                "qk_low_rank_dense_v": int(
                    state["q.down"].numel() + state["q.up"].numel() + state["q.bias"].numel()
                    + state["k.down"].numel() + state["k.up"].numel() + state["k.bias"].numel()
                    + (input_dim * head_dim + head_dim)
                ),
            }
            counts["reduction_fraction"] = 1.0 - counts["qk_low_rank_dense_v"] / counts["dense_qkv"]
            with (head_dir / "qk_dense_v_report.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "target_layer": layer_index,
                    "head_index": head_index,
                    "q_rank": int(state["q.down"].shape[1]),
                    "k_rank": int(state["k.down"].shape[1]),
                    "v": "dense",
                    "parameter_count": counts,
                    "source": "finetuned",
                }, f, indent=2)
            layer_paths.append(str(head_dir))
        written[str(layer_index)] = layer_paths
    return written


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    train_rows = load_selected_rows_by_bin(args.train_csv, windows_per_bin=args.train_windows_per_bin)
    train_tokens, _ = rows_to_token_batch(train_rows)
    train_input_ids = torch.as_tensor(train_tokens, dtype=torch.long)
    print(f"Train: {train_input_ids.shape[0]} windows, seq_len={train_input_ids.shape[1]}", flush=True)

    eval_rows = load_selected_rows_by_bin(args.eval_csv, windows_per_bin=args.eval_windows_per_bin)
    eval_tokens, eval_bins = rows_to_token_batch(eval_rows)
    eval_input_ids = torch.as_tensor(eval_tokens, dtype=torch.long)
    print(f"Eval : {eval_input_ids.shape[0]} windows, seq_len={eval_input_ids.shape[1]}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    root = Path(args.checkpoint_root)
    fallback_joint_root = Path(args.fallback_joint_qkv_root) if args.fallback_joint_qkv_root else None

    patched_layers: list[int] = []
    checkpoint_source_by_layer: dict[str, str] = {}
    for layer in range(num_layers):
        if not _qk_surgery_lib.layer_is_complete(
            root, layer=layer, num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=args.fallback_joint_config,
        ):
            raise FileNotFoundError(f"Cannot patch full model; layer {layer} is missing checkpoints.")
        states, _reports, source = _qk_surgery_lib.load_layer_qk_states(
            root, layer=layer, num_heads=num_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=args.fallback_joint_config,
        )
        _qk_surgery_lib.patch_layer_qk_dense_v(model, layer_index=layer, branch_states=states, dtype=dtype, device=device)
        patched_layers.append(layer)
        checkpoint_source_by_layer[str(layer)] = source

    trainable = make_trainable_qk(model, patched_layers=patched_layers)
    trainable_params = sum(p.numel() for p in trainable)
    print(f"Trainable Q/K parameters: {trainable_params:,}", flush=True)

    pre_eval = _qk_surgery_lib.evaluate_loss(model, eval_input_ids, batch_size=args.eval_batch_size, device=device)
    print(f"pre_finetune_eval {json.dumps(pre_eval)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    train_iter = iter_train_batches(train_input_ids, batch_size=args.batch_size, seed=args.seed)
    history: list[dict[str, float]] = []
    last_eval = pre_eval
    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.autocast(device_type="cpu", enabled=False)

    model.train()
    start_time = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        batch = next(train_iter).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(input_ids=batch, labels=batch, use_cache=False)
            loss = out.loss
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        running_loss += float(loss.detach().cpu())
        if step % args.log_every == 0:
            avg = running_loss / args.log_every
            elapsed = time.time() - start_time
            print(f"step={step:5d} loss={avg:.5f} elapsed={elapsed:.1f}s", flush=True)
            running_loss = 0.0
        if args.eval_every > 0 and step % args.eval_every == 0 and step != args.steps:
            model.eval()
            with torch.no_grad():
                last_eval = _qk_surgery_lib.evaluate_loss(model, eval_input_ids, batch_size=args.eval_batch_size, device=device)
            print(f"eval@step={step} {json.dumps(last_eval)}", flush=True)
            history.append({"step": step, **last_eval})
            model.train()

    model.eval()
    final_eval = _qk_surgery_lib.evaluate_loss(model, eval_input_ids, batch_size=args.eval_batch_size, device=device)
    history.append({"step": args.steps, **final_eval})
    print(f"final_eval {json.dumps(final_eval)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = save_finetuned_layout(
        model=model,
        output_dir=output_dir,
        patched_layers=patched_layers,
        num_heads=num_heads,
    )

    summary = {
        "model_name": args.model_name,
        "checkpoint_root": str(root),
        "fallback_joint_qkv_root": str(fallback_joint_root) if fallback_joint_root else None,
        "checkpoint_source_by_layer": checkpoint_source_by_layer,
        "patched_layers": patched_layers,
        "patched_heads_per_layer": num_heads,
        "trainable_qk_parameters": int(trainable_params),
        "train_windows": int(train_input_ids.shape[0]),
        "eval_windows": int(eval_input_ids.shape[0]),
        "eval_rarity_bins": sorted(set(eval_bins)),
        "seq_len": int(train_input_ids.shape[1]),
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "pre_finetune_eval": pre_eval,
        "post_finetune_eval": final_eval,
        "loss_delta_vs_pre": final_eval["loss"] - pre_eval["loss"],
        "perplexity_ratio_vs_pre": final_eval["perplexity"] / pre_eval["perplexity"],
        "intermediate_eval": history,
        "output_layout_dirs_per_layer": written,
    }
    with (output_dir / "finetune_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
