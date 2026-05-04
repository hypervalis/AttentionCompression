#!/usr/bin/env python3
"""Sweep script-32 training losses: relative, cosine, and both(w) for several w.

Single layer: pass ``--output-parent``, ``--internals-capture-dir``,
``--compressed-oproj-pt``, and ``--target-layer``.

All layers (same layout as ``scripts/33_...``): pass ``--artifact-base-dir`` and
``--first-layer`` / ``--last-layer``. Each layer ``L`` uses::

    {base}/layer{L:02d}_internals_160pb
    {base}/layer{L:02d}_oproj_mimic_ae_lowrank{R}_160pb/compressed_oproj.pt

and writes under ``{base}/layer{L:02d}_ffn_loss_sweep[_TAG]/`` when ``--run-tag`` is set. A combined
``all_layers_ffn_loss_sweep_summary[_TAG].json`` is written under ``--artifact-base-dir``.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep bottleneck FFN loss (script 32).")
    p.add_argument(
        "--artifact-base-dir",
        default=None,
        help="If set, sweep every layer in [--first-layer, --last-layer] using standard capture/oproj paths.",
    )
    p.add_argument("--first-layer", type=int, default=0)
    p.add_argument("--last-layer", type=int, default=15)
    p.add_argument(
        "--skip-missing-layer",
        action="store_true",
        help="With --artifact-base-dir: skip a layer if capture shards or compressed_oproj.pt is missing.",
    )
    p.add_argument(
        "--expected-capture-shards",
        type=int,
        default=60,
        help="Minimum .pt shard count to treat capture as ready (with --skip-missing-layer).",
    )
    p.add_argument(
        "--output-parent",
        default=None,
        help="Single-layer mode: directory holding loss_relative, loss_cosine, ... subdirs.",
    )
    p.add_argument("--internals-capture-dir", default=None)
    p.add_argument("--compressed-oproj-pt", default=None)
    p.add_argument("--oproj-projection-kind", required=True, choices=["dense", "lowrank", "pca_lowrank"])
    p.add_argument("--oproj-rank", type=int, default=768)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0, help="Single-layer mode only.")
    p.add_argument("--bottleneck-dim", type=int, default=1024)
    p.add_argument("--ffn-hidden-dim", type=int, default=4096)
    p.add_argument("--ae-state", required=True)
    p.add_argument("--ae-kind", default="decoder_residual_mlp", choices=["linear", "decoder_residual_mlp"])
    p.add_argument("--ae-hidden-dim", type=int, default=1536)
    p.add_argument("--train-windows-per-bin", type=int, default=128)
    p.add_argument("--eval-windows-per-bin", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--both-weights",
        default="0.1,0.25,0.5,0.75",
        help="Comma-separated ``--loss-relative-weight`` values for ``--loss-kind both`` runs.",
    )
    p.add_argument(
        "--run-tag",
        default="",
        help="With --artifact-base-dir: write under layerNN_ffn_loss_sweep_TAG/ and a distinct master JSON "
        "so reruns (e.g. denser w grid) do not overwrite prior sweeps.",
    )
    p.add_argument(
        "--both-only",
        action="store_true",
        help="Skip relative and cosine baselines; only run the both(w) grid (for weight-focused reruns).",
    )
    p.add_argument("--skip-relative", action="store_true")
    p.add_argument("--skip-cosine", action="store_true")
    p.add_argument("--skip-both", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def base_cmd(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(repo_root() / "scripts" / "32_train_bottleneck_ffn_after_mimic_oproj.py"),
        "--internals-capture-dir",
        args.internals_capture_dir,
        "--compressed-oproj-pt",
        args.compressed_oproj_pt,
        "--oproj-projection-kind",
        args.oproj_projection_kind,
        "--oproj-rank",
        str(args.oproj_rank),
        "--output-dir",
        str(out_dir),
        "--model-name",
        args.model_name,
        "--target-layer",
        str(args.target_layer),
        "--bottleneck-dim",
        str(args.bottleneck_dim),
        "--ffn-hidden-dim",
        str(args.ffn_hidden_dim),
        "--ae-state",
        args.ae_state,
        "--ae-kind",
        args.ae_kind,
        "--ae-hidden-dim",
        str(args.ae_hidden_dim),
        "--train-windows-per-bin",
        str(args.train_windows_per_bin),
        "--eval-windows-per-bin",
        str(args.eval_windows_per_bin),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]


def run_one(args: argparse.Namespace, name: str, extra: list[str], results: list[dict]) -> None:
    parent = Path(args.output_parent)
    out = parent / name
    cmd = base_cmd(args, out) + extra
    print("RUN", " ".join(cmd), flush=True)
    if args.dry_run:
        results.append({"name": name, "dry_run": True, "cmd": cmd})
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root()))
    report_path = out / "bottleneck_ffn_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    last = report["history"][-1]
    results.append(
        {
            "name": name,
            "output_dir": str(out),
            "loss_kind": report.get("loss_kind"),
            "loss_relative_weight": report.get("loss_relative_weight"),
            "last_epoch": last.get("epoch"),
            "train_loss": last.get("train_loss"),
            "eval": last.get("eval"),
        }
    )


def count_capture_shards(capture_dir: Path) -> int:
    return len(list(capture_dir.glob("layer_*_internals_*.pt")))


def run_sweep_for_namespace(args: argparse.Namespace) -> dict:
    weights = [float(x.strip()) for x in args.both_weights.split(",") if x.strip()]
    for w in weights:
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"Invalid both weight: {w}")

    parent = Path(args.output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    if not args.skip_relative:
        run_one(args, "loss_relative", ["--loss-kind", "relative"], results)
    if not args.skip_cosine:
        run_one(args, "loss_cosine", ["--loss-kind", "cosine"], results)
    if not args.skip_both:
        for w in weights:
            wtag = f"{w:g}".replace(".", "p")
            run_one(
                args,
                f"loss_both_w{wtag}",
                ["--loss-kind", "both", "--loss-relative-weight", str(w)],
                results,
            )

    summary_path = parent / "sweep_summary.json"
    payload = {
        "output_parent": str(parent),
        "internals_capture_dir": args.internals_capture_dir,
        "compressed_oproj_pt": args.compressed_oproj_pt,
        "target_layer": args.target_layer,
        "epochs": args.epochs,
        "both_weights": weights,
        "run_tag": (args.run_tag or "").strip() or None,
        "runs": results,
    }
    if not args.dry_run:
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if args.both_only:
        args.skip_relative = True
        args.skip_cosine = True
    base = args.artifact_base_dir

    def sweep_dir_name(tag: str) -> str:
        rt = (args.run_tag or "").strip()
        if not rt:
            return f"{tag}_ffn_loss_sweep"
        return f"{tag}_ffn_loss_sweep_{rt}"

    def master_summary_name() -> str:
        rt = (args.run_tag or "").strip()
        if not rt:
            return "all_layers_ffn_loss_sweep_summary.json"
        return f"all_layers_ffn_loss_sweep_summary_{rt}.json"

    if base:
        base_path = Path(base)
        if not base_path.is_dir():
            raise FileNotFoundError(f"artifact base not a directory: {base_path}")
        master: dict = {
            "artifact_base_dir": str(base_path),
            "run_tag": (args.run_tag or "").strip() or None,
            "master_summary": master_summary_name(),
            "layers": [],
        }
        for layer in range(args.first_layer, args.last_layer + 1):
            tag = f"layer{layer:02d}"
            capture_dir = base_path / f"{tag}_internals_160pb"
            oproj_pt = base_path / f"{tag}_oproj_mimic_ae_lowrank{args.oproj_rank}_160pb" / "compressed_oproj.pt"
            if args.skip_missing_layer:
                n = count_capture_shards(capture_dir)
                if n < args.expected_capture_shards or not oproj_pt.is_file():
                    print(
                        f"skip {tag}: capture_shards={n} (need>={args.expected_capture_shards}) "
                        f"compressed_oproj_exists={oproj_pt.is_file()}",
                        flush=True,
                    )
                    master["layers"].append(
                        {
                            "layer": layer,
                            "skipped": True,
                            "capture_shards": n,
                            "compressed_oproj_pt": str(oproj_pt),
                        }
                    )
                    continue
            layer_ns = copy.copy(args)
            layer_ns.target_layer = layer
            layer_ns.output_parent = str(base_path / sweep_dir_name(tag))
            layer_ns.internals_capture_dir = str(capture_dir)
            layer_ns.compressed_oproj_pt = str(oproj_pt)
            print(f"\n========== sweep {tag} -> {layer_ns.output_parent} ==========\n", flush=True)
            payload = run_sweep_for_namespace(layer_ns)
            master["layers"].append({"layer": layer, **payload})
        out_master = base_path / master_summary_name()
        if not args.dry_run:
            out_master.write_text(json.dumps(master, indent=2), encoding="utf-8")
        print(json.dumps(master, indent=2), flush=True)
        return

    if not args.output_parent or not args.internals_capture_dir or not args.compressed_oproj_pt:
        raise SystemExit(
            "Single-layer mode: require --output-parent, --internals-capture-dir, and --compressed-oproj-pt "
            "(or use --artifact-base-dir for all layers)."
        )
    payload = run_sweep_for_namespace(args)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
