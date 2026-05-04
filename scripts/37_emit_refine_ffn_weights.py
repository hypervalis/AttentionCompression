#!/usr/bin/env python3
"""Emit a dense ``--both-weights`` list for a second-stage FFN sweep around 36's picks.

Reads JSON written by ``scripts/36_select_ffn_loss_tradeoff.py`` (``--write-json``). For each
layer's best run with a parsed ``loss_relative_weight``, adds ``n`` weights spaced between
``w - half_width`` and ``w + half_width`` (clamped to ``[0, 1]``), unions across layers, sorts,
and prints a comma-separated string suitable for ``scripts/34_sweep_bottleneck_ffn_loss.py``.

Example::

    python scripts/37_emit_refine_ffn_weights.py \\
      --selection-json artifacts/ffn_loss_selections_layers1_4_main_sweep.json \\
      --half-width 0.05 --n 9

    # then paste printed line into:
    #   ... 34 ... --both-only --both-weights '<paste>' --run-tag refine_w_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit refined both-weights from 36 selection JSON.")
    p.add_argument("--selection-json", type=Path, required=True)
    p.add_argument(
        "--half-width",
        type=float,
        default=0.05,
        help="Each best w contributes [w-half, w+half] (clamped to [0,1]).",
    )
    p.add_argument("--n", type=int, default=9, help="Points per layer (including endpoints of local span).")
    p.add_argument("--decimals", type=int, default=4, help="Rounding for dedupe / printing.")
    return p.parse_args()


def linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def main() -> None:
    args = parse_args()
    if not args.selection_json.is_file():
        print(f"missing: {args.selection_json}", file=sys.stderr)
        sys.exit(1)
    doc = json.loads(args.selection_json.read_text(encoding="utf-8"))
    weights: set[float] = set()
    for block in doc.get("layers", []):
        best = block.get("best") or {}
        w = best.get("loss_relative_weight")
        if w is None:
            continue
        lo = max(0.0, float(w) - args.half_width)
        hi = min(1.0, float(w) + args.half_width)
        for x in linspace(lo, hi, max(2, args.n)):
            weights.add(round(x, args.decimals))
    merged = sorted(weights)
    if not merged:
        print("no loss_relative_weight in selection JSON", file=sys.stderr)
        sys.exit(2)
    s = ",".join(format(w, ".6g") for w in merged)
    print(s)


if __name__ == "__main__":
    main()
