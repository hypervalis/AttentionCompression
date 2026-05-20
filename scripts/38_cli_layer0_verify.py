#!/usr/bin/env python3
"""Verify ``attention-compress apply --target layer`` against prior experiment metrics.

Compares:
- Q/K: script-18-style CE loss on one layer (baseline vs patched in-memory).
- FFN: best eval from ``bottleneck_ffn_report.json`` vs CE loss on the CLI-saved model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch
from attention_compression.compress.artifacts import ffn_layer_dir, resolve_qk_root
from attention_compression.qk_surgery import layer_is_complete, load_layer_qk_states, patch_layer_qk_dense_v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CLI layer apply vs stored experiment metrics.")
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--checkpoints", type=Path, default=Path("/mnt/sdb1/dolma-v1_6-sample"))
    p.add_argument("--layer", type=int, default=0)
    p.add_argument(
        "--cli-model-dir",
        type=Path,
        required=True,
        help="Output from: attention-compress apply ... --target layer --layer N",
    )
    p.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("/mnt/sdb1/dolma-v1_6-sample/selected_windows/selected_train_windows.csv"),
    )
    p.add_argument("--windows-per-bin", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--projection-rank", type=int, default=768)
    p.add_argument("--ffn-epochs", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


@torch.no_grad()
def ce_loss(model: torch.nn.Module, input_ids: torch.Tensor, *, batch_size: int, device: torch.device) -> dict:
    losses, tokens = [], 0
    for start in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[start : start + batch_size].to(device)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        n = batch.numel() - batch.shape[0]
        losses.append(float(out.loss) * n)
        tokens += n
    loss = sum(losses) / tokens
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": tokens}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    rows = load_selected_rows_by_bin(str(args.selected_csv), windows_per_bin=args.windows_per_bin)
    input_ids = torch.as_tensor(rows_to_token_batch(rows)[0], dtype=torch.long)

    ffn_dir = ffn_layer_dir(
        args.checkpoints,
        layer=args.layer,
        projection_rank=args.projection_rank,
        ffn_epochs=args.ffn_epochs,
    )
    report_path = ffn_dir / "bottleneck_ffn_report.json"
    ffn_report_best: dict = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evals = [e["eval"] for e in report.get("history", []) if "eval" in e]
        ffn_report_best = max(evals, key=lambda x: x.get("cosine", 0.0)) if evals else {}

    qk_root = resolve_qk_root(args.checkpoints)
    num_heads = 16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    baseline = ce_loss(model, input_ids, batch_size=args.batch_size, device=device)

    qk_source = "incomplete"
    qk_only = dict(baseline)
    if layer_is_complete(
        qk_root,
        layer=args.layer,
        num_heads=num_heads,
        fallback_joint_root=None,
        fallback_joint_config="q64_k48_v128",
    ):
        states, _, source = load_layer_qk_states(
            qk_root,
            layer=args.layer,
            num_heads=num_heads,
            fallback_joint_root=None,
            fallback_joint_config="q64_k48_v128",
        )
        patch_layer_qk_dense_v(
            model, layer_index=args.layer, branch_states=states, dtype=dtype, device=device
        )
        model.eval()
        qk_only = ce_loss(model, input_ids, batch_size=args.batch_size, device=device)
        qk_source = source

    cli_model = AutoModelForCausalLM.from_pretrained(str(args.cli_model_dir), dtype=dtype, trust_remote_code=True)
    cli_model.to(device).eval()
    cli_metrics = ce_loss(cli_model, input_ids, batch_size=args.batch_size, device=device)

    # Optional: prior script-18 JSON if present
    prior_18 = args.checkpoints / "qk_dense_v" / "layer00_smoke_script18.json"
    prior_18_data = json.loads(prior_18.read_text()) if prior_18.is_file() else None

    out = {
        "model": args.model_name,
        "layer": args.layer,
        "checkpoints": str(args.checkpoints),
        "qk_root": str(qk_root),
        "ffn_artifact_dir": str(ffn_dir),
        "ffn_report_best_eval": ffn_report_best,
        "eval_windows_per_bin": args.windows_per_bin,
        "baseline": baseline,
        "qk_only_layer_patch": {**qk_only, "checkpoint_source": qk_source},
        "cli_saved_model": cli_metrics,
        "cli_model_dir": str(args.cli_model_dir),
        "delta_ppl_qk_only": qk_only["perplexity"] - baseline["perplexity"],
        "delta_ppl_cli_vs_baseline": cli_metrics["perplexity"] - baseline["perplexity"],
        "prior_script18_json": prior_18_data,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
