"""Train Q/K (09+16) or FFN block (27+31+32) from a :class:`CompressionPlan`."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from attention_compression.compress.artifacts import (
    ffn_layer_dir,
    internals_capture_dir,
    oproj_layer_dir,
    qk_head_dir,
)
from attention_compression.compress.targets import CompressionPlan, FfnLayerJob, QkHeadJob


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root() / "src")
    old = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src if not old else f"{src}{os.pathsep}{old}"
    return env


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("CMD:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root()), env=_subprocess_env())


def _train_qk_head(
    job: QkHeadJob,
    *,
    py: str,
    scripts: Path,
    model_path: str | Path,
    checkpoint_root: Path,
    cap_root: Path,
    selected_csv: str | Path,
    dry_run: bool,
    q_rank: int,
    k_rank: int,
    train_windows_per_bin: int,
    eval_windows_per_bin: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    windows_per_bin: int,
    capture_batch_size: int,
    samples_per_shard: int,
) -> None:
    cap_dir = cap_root / f"layer{job.layer:02d}_head{job.head:02d}_capture"
    out_dir = qk_head_dir(checkpoint_root, layer=job.layer, head=job.head, q_rank=q_rank, k_rank=k_rank)
    if not any(cap_dir.glob("*.pt")):
        _run(
            [
                py,
                str(scripts / "09_capture_head_activations.py"),
                "--selected-csv",
                str(selected_csv),
                "--output-dir",
                str(cap_dir),
                "--model-name",
                str(model_path),
                "--target-layer",
                str(job.layer),
                "--head-index",
                str(job.head),
                "--windows-per-bin",
                str(windows_per_bin),
                "--batch-size",
                str(capture_batch_size),
                "--samples-per-shard",
                str(samples_per_shard),
            ],
            dry_run=dry_run,
        )
    _run(
        [
            py,
            str(scripts / "16_train_qk_dense_v.py"),
            "--capture-dir",
            str(cap_dir),
            "--output-dir",
            str(out_dir),
            "--model-name",
            str(model_path),
            "--target-layer",
            str(job.layer),
            "--head-index",
            str(job.head),
            "--q-rank",
            str(q_rank),
            "--k-rank",
            str(k_rank),
            "--train-windows-per-bin",
            str(train_windows_per_bin),
            "--eval-windows-per-bin",
            str(eval_windows_per_bin),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--seed",
            str(seed),
        ],
        dry_run=dry_run,
    )


def _train_ffn_layer(
    job: FfnLayerJob,
    *,
    py: str,
    scripts: Path,
    model_path: str | Path,
    checkpoint_root: Path,
    cap_root: Path,
    selected_csv: str | Path,
    ae_state: str | Path,
    dry_run: bool,
    train_windows_per_bin: int,
    eval_windows_per_bin: int,
    batch_size: int,
    lr: float,
    seed: int,
    windows_per_bin: int,
    capture_batch_size: int,
    samples_per_shard: int,
    projection_kind: str,
    projection_rank: int,
    oproj_epochs: int,
    ffn_epochs: int,
    bottleneck_dim: int,
    ffn_hidden_dim: int,
    ae_kind: str,
    ae_hidden_dim: int,
    ffn_loss_kind: str,
    ffn_loss_relative_weight: float,
    ffn_cosine_weight: float,
) -> None:
    capture_dir = internals_capture_dir(cap_root, layer=job.layer)
    oproj_dir = oproj_layer_dir(checkpoint_root, layer=job.layer, rank=projection_rank)
    oproj_pt = oproj_dir / "compressed_oproj.pt"
    out_dir = ffn_layer_dir(checkpoint_root, layer=job.layer, projection_rank=projection_rank, ffn_epochs=ffn_epochs)

    if not any(capture_dir.glob("layer_*_internals_*.pt")) and not any(capture_dir.glob("*.pt")):
        _run(
            [
                py,
                str(scripts / "27_capture_layer_internals.py"),
                "--selected-csv",
                str(selected_csv),
                "--output-dir",
                str(capture_dir),
                "--model-name",
                str(model_path),
                "--target-layer",
                str(job.layer),
                "--windows-per-bin",
                str(windows_per_bin),
                "--batch-size",
                str(capture_batch_size),
                "--samples-per-shard",
                str(samples_per_shard),
            ],
            dry_run=dry_run,
        )

    if not oproj_pt.is_file():
        _run(
            [
                py,
                str(scripts / "31_train_compressed_oproj_from_bottleneck.py"),
                "--capture-dir",
                str(capture_dir),
                "--output-dir",
                str(oproj_dir),
                "--ae-state",
                str(ae_state),
                "--model-name",
                str(model_path),
                "--target-layer",
                str(job.layer),
                "--train-windows-per-bin",
                str(train_windows_per_bin),
                "--eval-windows-per-bin",
                str(eval_windows_per_bin),
                "--projection-kind",
                projection_kind,
                "--projection-rank",
                str(projection_rank),
                "--epochs",
                str(oproj_epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--seed",
                str(seed),
            ],
            dry_run=dry_run,
        )

    if not (out_dir / "bottleneck_ffn.pt").is_file():
        _run(
            [
                py,
                str(scripts / "32_train_bottleneck_ffn_after_mimic_oproj.py"),
                "--internals-capture-dir",
                str(capture_dir),
                "--compressed-oproj-pt",
                str(oproj_pt),
                "--oproj-projection-kind",
                projection_kind,
                "--oproj-rank",
                str(projection_rank),
                "--output-dir",
                str(out_dir),
                "--model-name",
                str(model_path),
                "--target-layer",
                str(job.layer),
                "--bottleneck-dim",
                str(bottleneck_dim),
                "--ffn-hidden-dim",
                str(ffn_hidden_dim),
                "--ae-state",
                str(ae_state),
                "--ae-kind",
                ae_kind,
                "--ae-hidden-dim",
                str(ae_hidden_dim),
                "--train-windows-per-bin",
                str(train_windows_per_bin),
                "--eval-windows-per-bin",
                str(eval_windows_per_bin),
                "--epochs",
                str(ffn_epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--seed",
                str(seed),
                "--loss-kind",
                ffn_loss_kind,
                "--loss-relative-weight",
                str(ffn_loss_relative_weight),
                "--cosine-weight",
                str(ffn_cosine_weight),
            ],
            dry_run=dry_run,
        )


def train_plan(
    plan: CompressionPlan,
    *,
    model_path: str | Path,
    checkpoint_root: Path,
    capture_root: Path | None,
    selected_csv: str | Path | None,
    ae_state: Path | None,
    dry_run: bool,
    q_rank: int = 64,
    k_rank: int = 48,
    train_windows_per_bin: int = 128,
    eval_windows_per_bin: int = 32,
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 5e-5,
    seed: int = 13,
    windows_per_bin: int = 160,
    capture_batch_size: int = 2,
    samples_per_shard: int = 16,
    projection_kind: str = "lowrank",
    projection_rank: int = 768,
    oproj_epochs: int = 10,
    ffn_epochs: int = 5,
    bottleneck_dim: int = 1024,
    ffn_hidden_dim: int = 4096,
    ae_kind: str = "decoder_residual_mlp",
    ae_hidden_dim: int = 4096,
    ffn_loss_kind: str = "both",
    ffn_loss_relative_weight: float = 0.25,
    ffn_cosine_weight: float = 1.0,
) -> None:
    if plan.qk_heads and selected_csv is None:
        raise ValueError("Q/K training requires --selected-csv")
    if plan.ffn_layers and (selected_csv is None or ae_state is None):
        raise ValueError("FFN-block training requires --selected-csv and --ae-state")

    py = sys.executable
    scripts = repo_root() / "scripts"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    cap_root = capture_root or checkpoint_root

    for job in plan.qk_heads:
        _train_qk_head(
            job,
            py=py,
            scripts=scripts,
            model_path=model_path,
            checkpoint_root=checkpoint_root,
            cap_root=cap_root,
            selected_csv=selected_csv,  # type: ignore[arg-type]
            dry_run=dry_run,
            q_rank=q_rank,
            k_rank=k_rank,
            train_windows_per_bin=train_windows_per_bin,
            eval_windows_per_bin=eval_windows_per_bin,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            windows_per_bin=windows_per_bin,
            capture_batch_size=capture_batch_size,
            samples_per_shard=samples_per_shard,
        )

    for job in plan.ffn_layers:
        _train_ffn_layer(
            job,
            py=py,
            scripts=scripts,
            model_path=model_path,
            checkpoint_root=checkpoint_root,
            cap_root=cap_root,
            selected_csv=selected_csv,  # type: ignore[arg-type]
            ae_state=ae_state,  # type: ignore[arg-type]
            dry_run=dry_run,
            train_windows_per_bin=train_windows_per_bin,
            eval_windows_per_bin=eval_windows_per_bin,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            windows_per_bin=windows_per_bin,
            capture_batch_size=capture_batch_size,
            samples_per_shard=samples_per_shard,
            projection_kind=projection_kind,
            projection_rank=projection_rank,
            oproj_epochs=oproj_epochs,
            ffn_epochs=ffn_epochs,
            bottleneck_dim=bottleneck_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            ae_kind=ae_kind,
            ae_hidden_dim=ae_hidden_dim,
            ffn_loss_kind=ffn_loss_kind,
            ffn_loss_relative_weight=ffn_loss_relative_weight,
            ffn_cosine_weight=ffn_cosine_weight,
        )
