"""Compression targets: one head (Q/K), FFN block, full layer, or whole model."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompressionTarget(str, Enum):
    """What to compress (validated tracks in research/FINDINGS.md)."""

    HEAD = "head"  # low-rank Q/K (+ dense V) for one attention head
    OPROJ = "oproj"  # swap o_proj for script-32 mimic only (same layer artifact dir)
    FFN = "ffn"  # mimic o_proj + bottleneck MLP on residual (scripts 27→31→32)
    LAYER = "layer"  # all Q/K heads on the layer + FFN block
    MODEL = "model"  # repeat layer work for each decoder layer


@dataclass(frozen=True)
class QkHeadJob:
    layer: int
    head: int


@dataclass(frozen=True)
class FfnLayerJob:
    layer: int


@dataclass(frozen=True)
class CompressionPlan:
    """Expanded train/apply units for a single ``--target`` choice."""

    qk_heads: tuple[QkHeadJob, ...]
    ffn_layers: tuple[FfnLayerJob, ...]


def parse_layers(spec: str, num_layers: int) -> list[int]:
    spec = spec.strip()
    if spec in ("all", "model"):
        return list(range(num_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    bad = [x for x in layers if x < 0 or x >= num_layers]
    if bad:
        raise ValueError(f"Layer index out of range 0..{num_layers - 1}: {bad}")
    return layers


def expand_plan(
    *,
    target: CompressionTarget,
    layer: int | None,
    head: int | None,
    layers_spec: str,
    num_layers: int,
    num_heads_per_layer: int,
) -> CompressionPlan:
    """Map CLI ``--target`` to Q/K head jobs and FFN layer jobs."""

    if target == CompressionTarget.HEAD:
        if layer is None or head is None:
            raise ValueError("--target head requires --layer and --head")
        return CompressionPlan(qk_heads=(QkHeadJob(layer, head),), ffn_layers=())

    if target in (CompressionTarget.FFN, CompressionTarget.OPROJ):
        if layer is None:
            raise ValueError(f"--target {target.value} requires --layer")
        return CompressionPlan(qk_heads=(), ffn_layers=(FfnLayerJob(layer),))

    if target == CompressionTarget.LAYER:
        if layer is None:
            raise ValueError("--target layer requires --layer")
        qk = tuple(QkHeadJob(layer, h) for h in range(num_heads_per_layer))
        return CompressionPlan(qk_heads=qk, ffn_layers=(FfnLayerJob(layer),))

    # MODEL
    layers = parse_layers(layers_spec, num_layers)
    qk: list[QkHeadJob] = []
    ffn: list[FfnLayerJob] = []
    for lyr in layers:
        heads = [head] if head is not None else list(range(num_heads_per_layer))
        qk.extend(QkHeadJob(lyr, h) for h in heads)
        ffn.append(FfnLayerJob(lyr))
    return CompressionPlan(qk_heads=tuple(qk), ffn_layers=tuple(ffn))
