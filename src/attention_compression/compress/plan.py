"""Summarize which checkpoints exist for a compression plan."""
from __future__ import annotations

from pathlib import Path

from attention_compression.compress.artifacts import ffn_layer_dir, qk_head_dir, qk_layer_complete
from attention_compression.compress.targets import CompressionPlan


def plan_status(
    plan: CompressionPlan,
    *,
    checkpoint_root: Path,
    num_heads_per_layer: int,
    fallback_joint_root: Path | None,
    fallback_joint_config: str,
    projection_rank: int = 768,
    ffn_epochs: int = 5,
) -> list[dict]:
    rows: list[dict] = []
    for job in plan.qk_heads:
        d = qk_head_dir(checkpoint_root, layer=job.layer, head=job.head)
        rows.append(
            {
                "kind": "qk_head",
                "layer": job.layer,
                "head": job.head,
                "checkpoint_dir": str(d),
                "ready": (d / "qk_dense_v_model.pt").is_file(),
            }
        )

    for job in plan.ffn_layers:
        d = ffn_layer_dir(
            checkpoint_root,
            layer=job.layer,
            projection_rank=projection_rank,
            ffn_epochs=ffn_epochs,
        )
        rows.append(
            {
                "kind": "ffn_block",
                "layer": job.layer,
                "checkpoint_dir": str(d),
                "ready": (d / "bottleneck_ffn.pt").is_file(),
                "note": "mimic o_proj + bottleneck MLP (scripts 27→31→32)",
            }
        )

    for lyr in sorted({j.layer for j in plan.qk_heads}):
        complete = qk_layer_complete(
            checkpoint_root,
            layer=lyr,
            num_heads=num_heads_per_layer,
            fallback_joint_root=fallback_joint_root,
            fallback_joint_config=fallback_joint_config,
        )
        rows.append(
            {
                "kind": "qk_layer_apply",
                "layer": lyr,
                "ready": complete,
                "note": "all heads trained (required before Q/K can be patched on this layer)",
            }
        )
    return rows
