"""Conventional checkpoint directory names under an artifact root."""
from __future__ import annotations

from pathlib import Path


def resolve_qk_root(root: Path) -> Path:
    """Q/K checkpoints usually live under ``<artifact_root>/qk_dense_v/``."""
    sub = root / "qk_dense_v"
    if sub.is_dir():
        return sub
    return root


def qk_head_dir(root: Path, *, layer: int, head: int, q_rank: int = 64, k_rank: int = 48) -> Path:
    """Match ``qk_surgery.qk_checkpoint_dir`` layout (validated default: Q64 K48)."""
    base = resolve_qk_root(root)
    if q_rank == 64 and k_rank == 48:
        return base / f"layer_{layer:02d}_head_{head:02d}_q64_k48_densev"
    return base / f"layer_{layer:02d}_head_{head:02d}_q{q_rank}_k{k_rank}_densev"


def qk_layer_complete(
    root: Path,
    *,
    layer: int,
    num_heads: int,
    fallback_joint_root: Path | None = None,
    fallback_joint_config: str = "q64_k48_v128",
) -> bool:
    from attention_compression.qk_surgery import layer_is_complete

    return layer_is_complete(
        resolve_qk_root(root),
        layer=layer,
        num_heads=num_heads,
        fallback_joint_root=fallback_joint_root,
        fallback_joint_config=fallback_joint_config,
    )


def oproj_layer_dir(root: Path, *, layer: int, rank: int = 768, windows_tag: str = "160pb") -> Path:
    """Script 31 output: low-rank ``o_proj`` trained to match teacher (use for weight swap)."""
    return root / f"layer{layer:02d}_oproj_mimic_ae_lowrank{rank}_{windows_tag}"


def resolve_oproj_swap_dir(
    root: Path,
    *,
    layer: int,
    projection_rank: int = 768,
    windows_tag: str = "160pb",
    ffn_epochs: int = 5,
) -> Path:
    """Prefer script-31 ``compressed_oproj.pt`` over script-32 co-trained ``mimic_oproj``."""
    d31 = oproj_layer_dir(root, layer=layer, rank=projection_rank, windows_tag=windows_tag)
    if (d31 / "compressed_oproj.pt").is_file():
        return d31
    return ffn_layer_dir(root, layer=layer, projection_rank=projection_rank, ffn_epochs=ffn_epochs)


def internals_capture_dir(root: Path, *, layer: int, windows_tag: str = "160pb") -> Path:
    return root / f"layer{layer:02d}_internals_{windows_tag}"


def ffn_layer_dir(
    root: Path,
    *,
    layer: int,
    projection_rank: int = 768,
    ffn_epochs: int = 5,
) -> Path:
    """Script 32 output (mimic o_proj + bottleneck MLP), matching script 33 naming."""
    return root / f"layer{layer:02d}_bottleneck_ffn_lowrank{projection_rank}_cotrain_{ffn_epochs}ep"
