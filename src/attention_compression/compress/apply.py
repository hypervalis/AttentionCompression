"""Apply trained compression checkpoints to a loaded causal LM."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_compression.compress.artifacts import ffn_layer_dir, qk_layer_complete, resolve_qk_root
from attention_compression.compress.targets import CompressionPlan
from attention_compression.compress.targets import CompressionTarget
from attention_compression.ffn_bottleneck import apply_ffn_block, apply_mimic_oproj_layer
from attention_compression.model_hub import load_causal_lm, num_heads, save_causal_lm
from attention_compression.qk_surgery import load_layer_qk_states, patch_layer_qk_dense_v


def _head_dim(model: torch.nn.Module, layer_index: int) -> int:
    from attention_compression.model_hub import get_transformer_layers

    attn = get_transformer_layers(model)[layer_index].self_attn
    if hasattr(attn, "head_dim"):
        return int(attn.head_dim)
    hidden = int(getattr(model.config, "hidden_size", 0))
    n = num_heads(model, layer_index)
    return hidden // n


def apply_plan(
    model: torch.nn.Module,
    plan: CompressionPlan,
    *,
    checkpoint_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    fallback_joint_root: Path | None,
    fallback_joint_config: str,
    materialize_dense_qk: bool,
    skip_missing: bool,
    projection_rank: int = 768,
    ffn_epochs: int = 5,
    compression_target: CompressionTarget | None = None,
    swap_o_proj: bool = True,
    keep_teacher_oproj_hook_only: bool = False,
) -> list[dict]:
    """Patch ``model`` in place; returns a log of applied steps."""
    log: list[dict] = []
    qk_layers_done: set[int] = set()
    ffn_layers_done: set[int] = set()

    qk_layers = sorted({j.layer for j in plan.qk_heads})
    for lyr in qk_layers:
        if lyr in qk_layers_done:
            continue
        n_heads = num_heads(model, lyr)
        qk_root = resolve_qk_root(checkpoint_root)
        if not qk_layer_complete(
            checkpoint_root,
            layer=lyr,
            num_heads=n_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=fallback_joint_config,
        ):
            msg = f"incomplete Q/K checkpoints for layer {lyr} under {checkpoint_root}"
            if skip_missing:
                print(f"skip: {msg}", flush=True)
                continue
            raise FileNotFoundError(msg)
        states, reports, source = load_layer_qk_states(
            qk_root,
            layer=lyr,
            num_heads=n_heads,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=fallback_joint_config,
        )
        patch_layer_qk_dense_v(
            model,
            layer_index=lyr,
            branch_states=states,
            dtype=dtype,
            device=device,
            materialize_dense=materialize_dense_qk,
        )
        qk_layers_done.add(lyr)
        log.append(
            {
                "target": "qk",
                "layer": lyr,
                "checkpoint_source": source,
                "materialize_dense_qk": materialize_dense_qk,
                "reports": reports,
            }
        )
        print(f"applied Q/K low-rank + dense V on layer {lyr} ({source})", flush=True)

    for job in plan.ffn_layers:
        if job.layer in ffn_layers_done:
            continue
        art = ffn_layer_dir(
            checkpoint_root,
            layer=job.layer,
            projection_rank=projection_rank,
            ffn_epochs=ffn_epochs,
        )
        if not (art / "bottleneck_ffn.pt").is_file():
            msg = f"missing FFN block at {art}"
            if skip_missing:
                print(f"skip: {msg}", flush=True)
                continue
            raise FileNotFoundError(msg)
        hdim = _head_dim(model, job.layer)
        n_heads = num_heads(model, job.layer)
        if compression_target == CompressionTarget.OPROJ:
            meta = apply_mimic_oproj_layer(
                model,
                layer_index=job.layer,
                artifact_dir=art,
                device=device,
                dtype=dtype,
                num_heads=n_heads,
                head_dim=hdim,
                checkpoint_root=checkpoint_root,
                projection_rank=projection_rank,
            )
            log.append({"target": "oproj", **meta})
            print(f"applied low-rank mimic o_proj on layer {job.layer}", flush=True)
        else:
            do_swap = swap_o_proj and not keep_teacher_oproj_hook_only
            meta = apply_ffn_block(
                model,
                layer_index=job.layer,
                artifact_dir=art,
                device=device,
                dtype=dtype,
                num_heads=n_heads,
                head_dim=hdim,
                swap_o_proj=do_swap,
                install_mlp_bridge=True,
                checkpoint_root=checkpoint_root,
                projection_rank=projection_rank,
            )
            log.append({"target": "ffn", **meta})
            kind = "swapped mimic o_proj + MLP bridge" if do_swap else "hook mimic + MLP bridge"
            print(f"applied FFN block ({kind}) on layer {job.layer}", flush=True)
        ffn_layers_done.add(job.layer)

    if plan.qk_heads and not qk_layers_done:
        heads_requested = sorted({(j.layer, j.head) for j in plan.qk_heads})
        print(
            "note: --target head trains one Q/K branch; apply still patches the whole layer "
            f"once all {num_heads(model, heads_requested[0][0])} head checkpoints exist. "
            f"Requested heads: {heads_requested}",
            flush=True,
        )

    return log


def apply_and_save(
    model_path: str | Path,
    output_path: str | Path,
    plan: CompressionPlan,
    *,
    checkpoint_root: Path,
    device: str = "auto",
    dtype: str = "auto",
    fallback_joint_root: Path | None = None,
    fallback_joint_config: str = "q64_k48_v128",
    materialize_dense_qk: bool = False,
    skip_missing: bool = False,
    projection_rank: int = 768,
    ffn_epochs: int = 5,
    compression_target: CompressionTarget | None = None,
    swap_o_proj: bool = True,
    keep_teacher_oproj_hook_only: bool = False,
) -> dict:
    model, dev, dt = load_causal_lm(model_path, device=device, dtype=dtype)
    model.eval()
    log = apply_plan(
        model,
        plan,
        checkpoint_root=checkpoint_root,
        device=dev,
        dtype=dt,
        fallback_joint_root=fallback_joint_root,
        fallback_joint_config=fallback_joint_config,
        materialize_dense_qk=materialize_dense_qk,
        skip_missing=skip_missing,
        projection_rank=projection_rank,
        ffn_epochs=ffn_epochs,
        compression_target=compression_target,
        swap_o_proj=swap_o_proj,
        keep_teacher_oproj_hook_only=keep_teacher_oproj_hook_only,
    )
    save_causal_lm(model, output_path)
    summary = {"model_path": str(model_path), "output_path": str(output_path), "applied": log}
    summary_path = Path(output_path) / "compression_apply_log.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
