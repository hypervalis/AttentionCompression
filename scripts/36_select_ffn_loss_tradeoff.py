#!/usr/bin/env python3
"""Pick bottleneck-FFN sweep runs closest to ``relative_mse -> 0`` and ``cosine -> 1``.

Reads ``sweep_summary.json`` under each ``layerNN_ffn_loss_sweep[_TAG]/`` (same layout as
``scripts/34_sweep_bottleneck_ffn_loss.py``). For every requested layer, ranks candidate runs
by a scalar score (lower is better)::

    score = relative_mse + gamma * (1 - cosine)

so you can emphasize alignment (large ``gamma``) or MSE (small ``gamma``). Optional
``--cosine-floor`` / ``--mse-ceiling`` drop infeasible runs before ranking.

Typical workflow for layers where ``loss_relative`` gave low cosine:

1. Re-run or extend a **both(w)** grid (dense ``w`` near 0 helps), e.g.::

     python scripts/34_sweep_bottleneck_ffn_loss.py \\
       --artifact-base-dir /path/to/artifacts \\
       --first-layer 1 --last-layer 4 \\
       --skip-missing-layer \\
       --oproj-projection-kind lowrank --oproj-rank 768 \\
       --ae-state .../layer00_head_concat_ae_half_residual_mlp/head_context_concat_autoencoder.pt \\
       --both-only --both-weights \"0.05,0.08,0.1,0.12,0.15,0.2\" \\
       --run-tag midlayer_wgrid_v1

2. Select per layer::

     python scripts/36_select_ffn_loss_tradeoff.py \\
       --artifact-base-dir /path/to/artifacts \\
       --layers 1,2,3,4 --run-tag midlayer_wgrid_v1 \\
       --candidates both --gamma 80 --write-json selections.json

3. Optional finer ``w`` grid around those picks (emit ``--both-weights`` for script 34)::

     python scripts/37_emit_refine_ffn_weights.py --selection-json selections.json \\
       --half-width 0.05 --n 9
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select best FFN sweep run by MSE vs cosine tradeoff.")
    p.add_argument("--artifact-base-dir", required=True, type=Path)
    p.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer indices (e.g. 1,2,3,4). Empty: all layers that have a sweep dir.",
    )
    p.add_argument(
        "--run-tag",
        default="",
        help="Same as script 34: read layerNN_ffn_loss_sweep_TAG/ when set.",
    )
    p.add_argument(
        "--candidates",
        choices=["both", "both_and_cosine", "all"],
        default="both",
        help="both: only loss_both_w* runs; both_and_cosine: include loss_cosine; all: every run with eval metrics.",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=80.0,
        help="Score = relative_mse + gamma * (1 - cosine). Increase to prefer cosine nearer 1.",
    )
    p.add_argument(
        "--cosine-floor",
        type=float,
        default=None,
        help="If set, exclude runs with eval cosine below this before ranking.",
    )
    p.add_argument(
        "--mse-ceiling",
        type=float,
        default=None,
        help="If set, exclude runs with eval relative_mse above this before ranking.",
    )
    p.add_argument("--write-json", type=Path, default=None, help="Write selections + ranked lists as JSON.")
    p.add_argument("--top-k", type=int, default=5, help="How many runners-up to print per layer.")
    return p.parse_args()


def sweep_parent(base: Path, layer: int, run_tag: str) -> Path:
    tag = f"layer{layer:02d}"
    rt = (run_tag or "").strip()
    if not rt:
        return base / f"{tag}_ffn_loss_sweep"
    return base / f"{tag}_ffn_loss_sweep_{rt}"


def include_run(name: str, candidates: str) -> bool:
    if candidates == "all":
        return True
    if candidates == "both":
        return name.startswith("loss_both_w")
    if candidates == "both_and_cosine":
        return name.startswith("loss_both_w") or name == "loss_cosine"
    raise ValueError(candidates)


def parse_both_weight(name: str) -> float | None:
    m = re.match(r"loss_both_w(\d+p\d+)$", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def main() -> None:
    args = parse_args()
    base: Path = args.artifact_base_dir
    if not base.is_dir():
        print(f"not a directory: {base}", file=sys.stderr)
        sys.exit(1)

    rt = (args.run_tag or "").strip()
    if args.layers.strip():
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers = []
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            if rt:
                m = re.match(rf"^layer(\d{{2}})_ffn_loss_sweep_{re.escape(rt)}$", p.name)
            else:
                m = re.match(r"^layer(\d{2})_ffn_loss_sweep$", p.name)
            if not m:
                continue
            layers.append(int(m.group(1)))
        layers = sorted(set(layers))

    out_doc: dict = {"artifact_base_dir": str(base), "run_tag": (args.run_tag or "").strip() or None, "layers": []}

    for layer in layers:
        parent = sweep_parent(base, layer, args.run_tag)
        summary_path = parent / "sweep_summary.json"
        if not summary_path.is_file():
            print(f"layer {layer}: missing {summary_path}", file=sys.stderr)
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        ranked: list[dict] = []
        for r in data.get("runs", []):
            name = r.get("name") or ""
            if not include_run(name, args.candidates):
                continue
            ev = r.get("eval") or {}
            mse = ev.get("relative_mse")
            cos = ev.get("cosine")
            if mse is None or cos is None or (isinstance(mse, float) and math.isnan(mse)):
                continue
            if args.cosine_floor is not None and cos < args.cosine_floor:
                continue
            if args.mse_ceiling is not None and mse > args.mse_ceiling:
                continue
            score = float(mse) + args.gamma * (1.0 - float(cos))
            w = parse_both_weight(name)
            ranked.append(
                {
                    "name": name,
                    "loss_relative_weight": w,
                    "relative_mse": mse,
                    "cosine": cos,
                    "score": score,
                    "output_dir": r.get("output_dir"),
                }
            )
        ranked.sort(key=lambda x: x["score"])
        if not ranked:
            print(f"layer {layer}: no candidates after filters (see {summary_path})")
            continue
        best = ranked[0]
        print(f"\n=== layer {layer} (gamma={args.gamma}) best: {best['name']} ===")
        print(f"  relative_mse={best['relative_mse']:.6g}  cosine={best['cosine']:.6f}  score={best['score']:.6g}")
        if best.get("loss_relative_weight") is not None:
            print(f"  implied --loss-kind both --loss-relative-weight {best['loss_relative_weight']}")
        print(f"  top-{args.top_k}:")
        for row in ranked[: args.top_k]:
            print(
                f"    {row['name']:22s}  mse={row['relative_mse']:.6g}  cos={row['cosine']:.6f}  score={row['score']:.6g}"
            )
        out_doc["layers"].append({"layer": layer, "best": best, "ranked": ranked})

    if args.write_json:
        args.write_json.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
        print(f"\nWrote {args.write_json}")


if __name__ == "__main__":
    main()
