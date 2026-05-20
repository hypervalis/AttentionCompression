#!/usr/bin/env python3
"""CLI: train or apply validated compression at head, FFN, layer, or model scope."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attention_compression.compress.apply import apply_and_save
from attention_compression.compress.plan import plan_status
from attention_compression.compress.targets import CompressionTarget, expand_plan
from attention_compression.losses import LOSS_KINDS
from attention_compression.compress.train import train_plan
from attention_compression.model_hub import load_causal_lm, num_heads, num_layers

_TARGETS = [t.value for t in CompressionTarget]


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "model",
        type=Path,
        help="Hugging Face model id or local directory (weights + config).",
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        required=True,
        help="Artifact root for trained checkpoints (read/write depending on subcommand).",
    )
    p.add_argument(
        "--target",
        choices=_TARGETS,
        required=True,
        help=(
            "head: Q/K for one attention head; "
            "oproj: swap o_proj for low-rank mimic (one layer); "
            "ffn: mimic o_proj (swapped) + bottleneck MLP bridge (one layer); "
            "layer: all Q/K heads + FFN on one layer; "
            "model: repeat per decoder layer (--layers to subset)."
        ),
    )
    p.add_argument("--layer", type=int, default=None, help="Decoder layer (required for head/ffn/layer).")
    p.add_argument("--head", type=int, default=None, help="Attention head (required for --target head).")
    p.add_argument(
        "--layers",
        default="all",
        help="For --target model: comma-separated indices or ranges (e.g. 0,2,4-7).",
    )
    p.add_argument(
        "--fallback-joint-qkv",
        type=Path,
        default=None,
        help="Optional legacy joint_qkv root if qk_dense_v checkpoints are missing (Q/K apply only).",
    )
    p.add_argument("--fallback-joint-config", default="q64_k48_v128")
    p.add_argument("--projection-rank", type=int, default=768, help="FFN / o_proj mimic rank (script 31–32).")
    p.add_argument("--ffn-epochs", type=int, default=5, help="Bottleneck FFN training epochs (script 32).")


def _plan_from_args(args: argparse.Namespace):
    probe, _dev, _dt = load_causal_lm(args.model, device="cpu", dtype="float32")
    n_layers = num_layers(probe)
    layer_for_heads = args.layer if args.layer is not None else 0
    n_heads = num_heads(probe, layer_for_heads)
    del probe
    return expand_plan(
        target=CompressionTarget(args.target),
        layer=args.layer,
        head=args.head,
        layers_spec=args.layers,
        num_layers=n_layers,
        num_heads_per_layer=n_heads,
    )


def cmd_plan(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    probe, _, _ = load_causal_lm(args.model, device="cpu", dtype="float32")
    n_heads = num_heads(probe, args.layer or 0)
    rows = plan_status(
        plan,
        checkpoint_root=args.checkpoints,
        num_heads_per_layer=n_heads,
        fallback_joint_root=args.fallback_joint_qkv,
        fallback_joint_config=args.fallback_joint_config,
        projection_rank=args.projection_rank,
        ffn_epochs=args.ffn_epochs,
    )
    print(
        json.dumps(
            {
                "model": str(args.model),
                "target": args.target,
                "qk_heads": len(plan.qk_heads),
                "ffn_layers": len(plan.ffn_layers),
                "status": rows,
            },
            indent=2,
        )
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    summary = apply_and_save(
        args.model,
        args.output,
        plan,
        checkpoint_root=args.checkpoints,
        device=args.device,
        dtype=args.dtype,
        fallback_joint_root=args.fallback_joint_qkv,
        fallback_joint_config=args.fallback_joint_config,
        materialize_dense_qk=args.materialize_dense_qk,
        skip_missing=args.skip_missing,
        projection_rank=args.projection_rank,
        ffn_epochs=args.ffn_epochs,
        compression_target=CompressionTarget(args.target),
        swap_o_proj=not args.keep_teacher_oproj,
        keep_teacher_oproj_hook_only=args.keep_teacher_oproj,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    train_plan(
        plan,
        model_path=args.model,
        checkpoint_root=args.checkpoints,
        capture_root=args.capture_root,
        selected_csv=args.selected_csv,
        ae_state=args.ae_state,
        dry_run=args.dry_run,
        q_rank=args.q_rank,
        k_rank=args.k_rank,
        train_windows_per_bin=args.train_windows_per_bin,
        eval_windows_per_bin=args.eval_windows_per_bin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        windows_per_bin=args.windows_per_bin,
        projection_kind=args.projection_kind,
        projection_rank=args.projection_rank,
        oproj_epochs=args.oproj_epochs,
        ffn_epochs=args.ffn_epochs,
        bottleneck_dim=args.bottleneck_dim,
        ffn_hidden_dim=args.ffn_hidden_dim,
        ae_kind=args.ae_kind,
        ae_hidden_dim=args.ae_hidden_dim,
        ffn_loss_kind=args.ffn_loss_kind,
        ffn_loss_relative_weight=args.ffn_loss_relative_weight,
        ffn_cosine_weight=args.ffn_cosine_weight,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attention-compress",
        description=(
            "Train or apply validated compression: "
            "Q/K low-rank + dense V (per head), and/or FFN block "
            "(mimic o_proj + bottleneck MLP on the post-attention residual)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="List checkpoint readiness for the requested target.")
    _add_common_args(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_apply = sub.add_parser("apply", help="Patch checkpoints into a model and save.")
    _add_common_args(p_apply)
    p_apply.add_argument("--output", type=Path, required=True, help="Output directory (save_pretrained).")
    p_apply.add_argument("--device", default="auto")
    p_apply.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p_apply.add_argument(
        "--materialize-dense-qk",
        action="store_true",
        help="Fold low-rank Q/K into dense Linear modules (better decode speed).",
    )
    p_apply.add_argument("--skip-missing", action="store_true", help="Skip layers with incomplete checkpoints.")
    p_apply.add_argument(
        "--keep-teacher-oproj",
        action="store_true",
        help="Do not replace o_proj; keep teacher weights and use mimic only in the MLP bridge hook.",
    )
    p_apply.set_defaults(func=cmd_apply)

    p_train = sub.add_parser("train", help="Capture activations and train into --checkpoints.")
    _add_common_args(p_train)
    p_train.add_argument(
        "--selected-csv",
        type=Path,
        default=None,
        help="Rarity-stratified window manifest (required for Q/K and FFN training).",
    )
    p_train.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help="Activation capture directory (default: same as --checkpoints).",
    )
    p_train.add_argument(
        "--ae-state",
        type=Path,
        default=None,
        help="Frozen head-concat autoencoder .pt (required for ffn / layer / model).",
    )
    p_train.add_argument("--q-rank", type=int, default=64)
    p_train.add_argument("--k-rank", type=int, default=48)
    p_train.add_argument("--train-windows-per-bin", type=int, default=128)
    p_train.add_argument("--eval-windows-per-bin", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=5, help="Q/K training epochs.")
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.add_argument("--seed", type=int, default=13)
    p_train.add_argument("--windows-per-bin", type=int, default=160)
    p_train.add_argument(
        "--projection-kind",
        default="lowrank",
        choices=["dense", "lowrank", "pca_lowrank"],
    )
    p_train.add_argument("--oproj-epochs", type=int, default=10)
    p_train.add_argument("--bottleneck-dim", type=int, default=1024)
    p_train.add_argument("--ffn-hidden-dim", type=int, default=4096)
    p_train.add_argument("--ae-kind", default="decoder_residual_mlp")
    p_train.add_argument("--ae-hidden-dim", type=int, default=4096)
    p_train.add_argument("--ffn-loss-kind", default="both", choices=list(LOSS_KINDS))
    p_train.add_argument("--ffn-loss-relative-weight", type=float, default=0.25)
    p_train.add_argument(
        "--ffn-cosine-weight",
        type=float,
        default=1.0,
        help="Cosine multiplier when --ffn-loss-kind relative_plus_cosine.",
    )
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        plan = _plan_from_args(args)
        if plan.qk_heads and args.selected_csv is None:
            parser.error("--selected-csv is required for Q/K training (head / layer / model)")
        if plan.ffn_layers and args.selected_csv is None:
            parser.error("--selected-csv is required for FFN-block training")
        if plan.ffn_layers and args.ae_state is None:
            parser.error("--ae-state is required for FFN-block training (ffn / layer / model)")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
