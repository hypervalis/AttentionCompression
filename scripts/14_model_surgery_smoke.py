#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from attention_compression.activations import load_selected_rows_by_bin, rows_to_token_batch


class DenseExceptOneHeadLowRankProjection(torch.nn.Module):
    """Projection that stores one head as low-rank factors and the rest as dense rows."""

    def __init__(
        self,
        *,
        original: torch.nn.Linear,
        branch_state: dict[str, torch.Tensor],
        branch_name: str,
        head_index: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        input_dim = int(original.in_features)
        output_dim = int(original.out_features)
        head_start = head_index * head_dim
        head_end = head_start + head_dim
        if head_start < 0 or head_end > output_dim:
            raise ValueError(f"head_index={head_index} is outside projection output dim {output_dim}")

        weight = original.weight.detach().to(device=device, dtype=dtype)
        bias = original.bias.detach().to(device=device, dtype=dtype) if original.bias is not None else None
        self.register_buffer("before_weight", weight[:head_start].contiguous())
        self.register_buffer("after_weight", weight[head_end:].contiguous())
        self.register_buffer("before_bias", None if bias is None else bias[:head_start].contiguous())
        self.register_buffer("after_bias", None if bias is None else bias[head_end:].contiguous())

        prefix = f"{branch_name}."
        down = branch_state[prefix + "down"].to(device=device, dtype=dtype)
        up = branch_state[prefix + "up"].to(device=device, dtype=dtype)
        low_bias = branch_state[prefix + "bias"].to(device=device, dtype=dtype)
        if down.shape[0] != input_dim or up.shape[1] != head_dim or low_bias.shape[0] != head_dim:
            raise ValueError(
                f"Bad {branch_name} branch shape: down={tuple(down.shape)} "
                f"up={tuple(up.shape)} bias={tuple(low_bias.shape)}"
            )
        self.down = torch.nn.Parameter(down, requires_grad=False)
        self.up = torch.nn.Parameter(up, requires_grad=False)
        self.low_bias = torch.nn.Parameter(low_bias, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pieces = []
        if self.before_weight.shape[0] > 0:
            pieces.append(F.linear(hidden_states, self.before_weight, self.before_bias))
        pieces.append(hidden_states @ self.down @ self.up + self.low_bias)
        if self.after_weight.shape[0] > 0:
            pieces.append(F.linear(hidden_states, self.after_weight, self.after_bias))
        return torch.cat(pieces, dim=-1)


class MultiHeadLowRankProjection(torch.nn.Module):
    """Drop-in replacement for an OLMo Q/K/V projection using per-head low-rank factors."""

    def __init__(
        self,
        *,
        branch_states: list[dict[str, torch.Tensor]],
        branch_name: str,
        input_dim: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.branch_name = branch_name
        self.down = torch.nn.ParameterList()
        self.up = torch.nn.ParameterList()
        self.bias = torch.nn.ParameterList()
        for state in branch_states:
            prefix = f"{branch_name}."
            down = state[prefix + "down"].to(device=device, dtype=dtype)
            up = state[prefix + "up"].to(device=device, dtype=dtype)
            bias = state[prefix + "bias"].to(device=device, dtype=dtype)
            if down.shape[0] != input_dim or up.shape[1] != head_dim or bias.shape[0] != head_dim:
                raise ValueError(
                    f"Bad {branch_name} branch shape: down={tuple(down.shape)} "
                    f"up={tuple(up.shape)} bias={tuple(bias.shape)}"
                )
            self.down.append(torch.nn.Parameter(down, requires_grad=False))
            self.up.append(torch.nn.Parameter(up, requires_grad=False))
            self.bias.append(torch.nn.Parameter(bias, requires_grad=False))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pieces = [
            hidden_states @ self.down[i] @ self.up[i] + self.bias[i]
            for i in range(len(self.down))
        ]
        return torch.cat(pieces, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch one OLMo layer with per-head low-rank Q/K/V branches.")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--config-name", default="q64_k48_v128")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--head-index", type=int, default=None, help="Patch only this head; default patches all heads.")
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--windows-per-bin", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_dir(root: Path, *, layer: int, head: int, config_name: str) -> Path:
    candidates = [
        root / f"head_{head:02d}_{config_name}",
        root / f"layer_{layer:02d}_head_{head:02d}_{config_name}_large",
        root / f"layer_{layer:02d}_head_{head:02d}_{config_name}",
    ]
    for path in candidates:
        if (path / "joint_qkv_model.pt").exists():
            return path
    raise FileNotFoundError(
        "No joint_qkv_model.pt found for "
        f"layer={layer} head={head} config={config_name}; tried {[str(p) for p in candidates]}"
    )


def load_branch_states(root: Path, *, layer: int, num_heads: int, config_name: str) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, object]]]:
    states = []
    reports = []
    for head in range(num_heads):
        ckpt_dir = checkpoint_dir(root, layer=layer, head=head, config_name=config_name)
        states.append(torch.load(ckpt_dir / "joint_qkv_model.pt", map_location="cpu"))
        report_path = ckpt_dir / "joint_qkv_report.json"
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
        else:
            report = {"head_index": head, "checkpoint_dir": str(ckpt_dir)}
        report["checkpoint_dir"] = str(ckpt_dir)
        reports.append(report)
    return states, reports


def load_branch_state(root: Path, *, layer: int, head: int, config_name: str) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    ckpt_dir = checkpoint_dir(root, layer=layer, head=head, config_name=config_name)
    state = torch.load(ckpt_dir / "joint_qkv_model.pt", map_location="cpu")
    report_path = ckpt_dir / "joint_qkv_report.json"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {"head_index": head, "checkpoint_dir": str(ckpt_dir)}
    report["checkpoint_dir"] = str(ckpt_dir)
    return state, report


def patch_layer_qkv(
    model: torch.nn.Module,
    *,
    layer_index: int,
    branch_states: list[dict[str, torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    attn = model.model.layers[layer_index].self_attn
    input_dim = int(attn.q_proj.in_features)
    head_dim = int(attn.head_dim)
    attn.q_proj = MultiHeadLowRankProjection(
        branch_states=branch_states,
        branch_name="q",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    attn.k_proj = MultiHeadLowRankProjection(
        branch_states=branch_states,
        branch_name="k",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    attn.v_proj = MultiHeadLowRankProjection(
        branch_states=branch_states,
        branch_name="v",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def patch_one_head_qkv(
    model: torch.nn.Module,
    *,
    layer_index: int,
    head_index: int,
    branch_state: dict[str, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    attn = model.model.layers[layer_index].self_attn
    head_dim = int(attn.head_dim)
    attn.q_proj = DenseExceptOneHeadLowRankProjection(
        original=attn.q_proj,
        branch_state=branch_state,
        branch_name="q",
        head_index=head_index,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    attn.k_proj = DenseExceptOneHeadLowRankProjection(
        original=attn.k_proj,
        branch_state=branch_state,
        branch_name="k",
        head_index=head_index,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    attn.v_proj = DenseExceptOneHeadLowRankProjection(
        original=attn.v_proj,
        branch_state=branch_state,
        branch_name="v",
        head_index=head_index,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


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

    baseline = evaluate_loss(model, input_ids, batch_size=args.batch_size, device=device)

    if args.head_index is None:
        branch_states, reports = load_branch_states(
            Path(args.checkpoint_root),
            layer=args.target_layer,
            num_heads=int(model.config.num_attention_heads),
            config_name=args.config_name,
        )
        patch_layer_qkv(model, layer_index=args.target_layer, branch_states=branch_states, dtype=dtype, device=device)
        patched_heads = "all"
    else:
        branch_state, report = load_branch_state(
            Path(args.checkpoint_root),
            layer=args.target_layer,
            head=args.head_index,
            config_name=args.config_name,
        )
        reports = [report]
        patch_one_head_qkv(
            model,
            layer_index=args.target_layer,
            head_index=args.head_index,
            branch_state=branch_state,
            dtype=dtype,
            device=device,
        )
        patched_heads = [args.head_index]
    model.eval()
    patched = evaluate_loss(model, input_ids, batch_size=args.batch_size, device=device)

    result = {
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "patched_heads": patched_heads,
        "config_name": args.config_name,
        "windows": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
        "rarity_bins": sorted(set(bins)),
        "baseline": baseline,
        "patched": patched,
        "loss_delta": patched["loss"] - baseline["loss"],
        "perplexity_ratio": patched["perplexity"] / baseline["perplexity"],
        "checkpoints": [
            {
                "head_index": int(report.get("head_index", i)),
                "q_rank": report.get("q_rank"),
                "k_rank": report.get("k_rank"),
                "v_rank": report.get("v_rank"),
                "checkpoint_dir": report["checkpoint_dir"],
            }
            for i, report in enumerate(reports)
        ],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
