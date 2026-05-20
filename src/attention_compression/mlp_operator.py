"""Estimate MLP as an operator: Cov(x), E[J^T J], and ridge-linear B."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from attention_compression.supervised_pca import centered_stats_from_raw, fit_ridge_rrr

BASIS_KINDS = ("input_pca", "operator_jacobian", "linear_rrr")


def rank_for_variance(ratios: torch.Tensor, threshold: float) -> int:
    cum = torch.cumsum(ratios, dim=0)
    idx = int(torch.searchsorted(cum, threshold).item())
    return min(idx + 1, ratios.numel())


def spectrum_report(evals: torch.Tensor, *, label: str) -> dict:
    total = evals.sum().clamp_min(1e-12)
    ratios = (evals / total).tolist()
    ranks = {f"var_{int(t * 100)}pct": rank_for_variance(evals / total, t) for t in (0.9, 0.95, 0.99)}
    return {
        "label": label,
        "top_eigenvalues": evals[:16].tolist(),
        "ranks_for_variance": ranks,
    }


def block_offdiag_frobenius(gram: torch.Tensor, num_blocks: int) -> dict:
    """Frobenius norm of off-diagonal blocks of ``gram`` (equal index bands)."""
    dim = int(gram.shape[0])
    block = dim // num_blocks
    if block * num_blocks != dim:
        raise ValueError(f"dim {dim} not divisible by num_blocks {num_blocks}")
    off = 0.0
    on = 0.0
    for i in range(num_blocks):
        lo_i = i * block
        hi_i = lo_i + block
        for j in range(num_blocks):
            lo_j = j * block
            hi_j = lo_j + block
            sl = gram[lo_i:hi_i, lo_j:hi_j]
            nrm = float(torch.linalg.norm(sl, ord="fro").item())
            if i == j:
                on += nrm**2
            else:
                off += nrm**2
    return {
        "num_blocks": num_blocks,
        "offdiag_fro_norm": off**0.5,
        "diag_fro_norm": on**0.5,
        "offdiag_over_diag": off**0.5 / max(on**0.5, 1e-12),
    }


def _add_jacobian_gram_cpu(
    mlp_cpu: torch.nn.Module,
    x_tokens: torch.Tensor,
    G: torch.Tensor,
    *,
    start_count: int,
    target: int,
) -> int:
    """Accumulate ``G += J^T J`` with per-token ``jacrev`` (MLP-only module on CPU)."""
    mlp_cpu.eval().float()

    def f_vec(x1d: torch.Tensor) -> torch.Tensor:
        return mlp_cpu(x1d.unsqueeze(0)).squeeze(0)

    n = start_count
    for t in range(x_tokens.shape[0]):
        xt = x_tokens[t].detach().requires_grad_(True)
        J = torch.func.jacrev(f_vec)(xt)
        G.add_(J.T @ J)
        del J
        n += 1
        if n % 16 == 0 or n == target:
            print(f"  jacobian tokens {n}/{target}", flush=True)
    return n


def accumulate_moment_stats(
    capture_paths: list[Path],
    mlp: torch.nn.Module,
    device: torch.device,
    *,
    max_moment_tokens: int = 200_000,
    max_jacobian_tokens: int = 1024,
    moment_batch_size: int = 256,
    seed: int = 13,
) -> tuple[dict[str, torch.Tensor | int], list[torch.Tensor]]:
    """First pass: batched moments for Cov(x), ridge B; return Jacobian token batches."""
    mlp.eval()
    dim: int | None = None
    n_mom = 0
    sum_x: torch.Tensor | None = None
    sum_y: torch.Tensor | None = None
    xtx: torch.Tensor | None = None
    xty: torch.Tensor | None = None
    yty: torch.Tensor | None = None
    jac_remaining = max_jacobian_tokens
    gen = torch.Generator().manual_seed(seed)
    jac_batches: list[torch.Tensor] = []

    for path in capture_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        x_all = shard["ffn_input"].reshape(-1, shard["ffn_input"].shape[-1]).float()
        if dim is None:
            dim = int(x_all.shape[-1])
            sum_x = torch.zeros(dim, device=device)
            sum_y = torch.zeros(dim, device=device)
            xtx = torch.zeros(dim, dim, device=device)
            xty = torch.zeros(dim, dim, device=device)
            yty = torch.zeros(dim, dim, device=device)

        take_m = min(x_all.shape[0], max_moment_tokens - n_mom)
        if take_m <= 0:
            continue
        x_shard = x_all[:take_m]
        for start in range(0, take_m, moment_batch_size):
            x = x_shard[start : start + moment_batch_size].to(device=device)
            with torch.no_grad():
                y = mlp(x)
            assert sum_x is not None and xtx is not None and sum_y is not None
            sum_x = sum_x + x.sum(dim=0)
            sum_y = sum_y + y.sum(dim=0)
            xtx = xtx + x.T @ x
            xty = xty + x.T @ y
            yty = yty + y.T @ y
            n_mom += int(x.shape[0])

        if jac_remaining > 0:
            x_j = x_shard
            n_pick = min(jac_remaining, x_j.shape[0])
            if n_pick < x_j.shape[0]:
                idx = torch.randperm(x_j.shape[0], generator=gen)[:n_pick]
                x_j = x_j[idx]
            jac_remaining -= n_pick
            jac_batches.append(x_j.cpu())

        if n_mom >= max_moment_tokens:
            break

    if dim is None or n_mom < 2 or sum_x is None:
        raise RuntimeError("need captures with at least 2 tokens")
    assert xtx is not None and xty is not None and yty is not None

    mean_x = sum_x / n_mom
    mean_y = sum_y / n_mom
    cov_x = (xtx - n_mom * torch.outer(mean_x, mean_x)) / (n_mom - 1)
    moments = {
        "dim": dim,
        "n_moment_tokens": n_mom,
        "mean_x": mean_x.cpu(),
        "mean_y": mean_y.cpu(),
        "cov_x": cov_x.cpu(),
        "sum_x": sum_x.cpu(),
        "sum_y": sum_y.cpu(),
        "xtx": xtx.cpu(),
        "xty": xty.cpu(),
        "yty": yty.cpu(),
    }
    return moments, jac_batches


def accumulate_jacobian_gram(
    jac_batches: list[torch.Tensor],
    mlp_cpu: torch.nn.Module,
    *,
    max_jacobian_tokens: int,
) -> tuple[torch.Tensor, int]:
    dim = int(next(iter(jac_batches)).shape[-1]) if jac_batches else 0
    G = torch.zeros(dim, dim)
    n_jac = 0
    for x_j in jac_batches:
        n_jac = _add_jacobian_gram_cpu(mlp_cpu, x_j, G, start_count=n_jac, target=max_jacobian_tokens)
        if n_jac >= max_jacobian_tokens:
            break
    return G / max(n_jac, 1), n_jac


def finalize_operator_stats(
    moments: dict[str, torch.Tensor | int],
    operator_gram: torch.Tensor,
    n_jacobian_tokens: int,
    *,
    ridge: float,
) -> dict[str, torch.Tensor | int | float]:
    n_mom = int(moments["n_moment_tokens"])
    stats = centered_stats_from_raw(
        n=n_mom,
        sum_x=moments["sum_x"].numpy(),  # type: ignore[union-attr]
        sum_y=moments["sum_y"].numpy(),  # type: ignore[union-attr]
        xtx=moments["xtx"].numpy(),  # type: ignore[union-attr]
        xty=moments["xty"].numpy(),  # type: ignore[union-attr]
        yty=moments["yty"].numpy(),  # type: ignore[union-attr]
    )
    b_full, _rrr_eigvecs, rrr_eigvals = fit_ridge_rrr(stats, ridge=ridge)
    _b_svd_u, b_svd_s, b_svd_vt = np.linalg.svd(b_full, full_matrices=False)
    return {
        "dim": moments["dim"],
        "n_moment_tokens": n_mom,
        "n_jacobian_tokens": n_jacobian_tokens,
        "mean_x": moments["mean_x"],
        "mean_y": moments["mean_y"],
        "cov_x": moments["cov_x"],
        "operator_gram": operator_gram,
        "b_full": torch.from_numpy(b_full),
        "rrr_eigvals": torch.from_numpy(rrr_eigvals),
        "b_singular_values": torch.from_numpy(b_svd_s),
        "b_svd_vt": torch.from_numpy(b_svd_vt),
    }


def accumulate_mlp_operator_stats(
    capture_paths: list[Path],
    mlp: torch.nn.Module,
    device: torch.device,
    *,
    max_moment_tokens: int = 200_000,
    max_jacobian_tokens: int = 1024,
    ridge: float = 1e-4,
    seed: int = 13,
    moment_batch_size: int = 256,
    mlp_cpu: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor | int | float]:
    """Moments on ``device``; Jacobian Gram on CPU ``mlp_cpu`` (defaults to ``mlp.cpu()``)."""
    moments, jac_batches = accumulate_moment_stats(
        capture_paths,
        mlp,
        device,
        max_moment_tokens=max_moment_tokens,
        max_jacobian_tokens=max_jacobian_tokens,
        moment_batch_size=moment_batch_size,
        seed=seed,
    )
    if mlp_cpu is None:
        mlp_cpu = mlp.float().cpu()
    if max_jacobian_tokens > 0 and jac_batches:
        G, n_jac = accumulate_jacobian_gram(jac_batches, mlp_cpu, max_jacobian_tokens=max_jacobian_tokens)
    else:
        dim = int(moments["dim"])
        G, n_jac = torch.zeros(dim, dim), 0
    return finalize_operator_stats(moments, G, n_jac, ridge=ridge)


def decompose_gram(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    basis = evecs[:, order]
    return basis, evals


def analyze_operator_artifact(raw: dict) -> dict:
    cov_x = raw["cov_x"].float()
    G = raw["operator_gram"].float()
    basis_x, evals_x = decompose_gram(cov_x)
    basis_g, evals_g = decompose_gram(G)
    b_s = raw["b_singular_values"].float()

    num_blocks = 4
    dim = int(raw["dim"])
    block = dim // num_blocks
    g_op = basis_g.T @ G @ basis_g
    coupling = {
        "G_in_input_pca_blocks": block_offdiag_frobenius(basis_x.T @ G @ basis_x, num_blocks),
        "G_in_operator_eigen_blocks": block_offdiag_frobenius(g_op, num_blocks),
        "G_in_coordinate_blocks": block_offdiag_frobenius(G, num_blocks),
    }

    return {
        "dim": dim,
        "n_moment_tokens": int(raw["n_moment_tokens"]),
        "n_jacobian_tokens": int(raw["n_jacobian_tokens"]),
        "input_pca": spectrum_report(evals_x, label="Cov(x)"),
        "operator_jacobian": spectrum_report(evals_g, label="E[J^T J]"),
        "linear_rrr": spectrum_report(b_s**2, label="s^2 from SVD(B_ridge)"),
        "block_coupling": coupling,
        "mean_x_norm": float(raw["mean_x"].norm()),
    }


def save_operator_artifact(
    path: str | Path,
    raw: dict,
    *,
    report: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    basis_g, evals_g = decompose_gram(raw["operator_gram"].float())
    basis_x, evals_x = decompose_gram(raw["cov_x"].float())
    payload = {
        "mean_x": raw["mean_x"],
        "mean_y": raw["mean_y"],
        "cov_x": raw["cov_x"],
        "operator_gram": raw["operator_gram"],
        "basis_input_pca": basis_x,
        "evals_input_pca": evals_x,
        "basis_operator": basis_g,
        "evals_operator": evals_g,
        "b_full": raw["b_full"],
        "b_singular_values": raw["b_singular_values"],
        "n_moment_tokens": raw["n_moment_tokens"],
        "n_jacobian_tokens": raw["n_jacobian_tokens"],
        "dim": raw["dim"],
    }
    torch.save(payload, path)
    if report is not None:
        report_path = path.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_operator_basis(
    artifact_path: str | Path,
    *,
    kind: str = "operator_jacobian",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean, basis, eigenvalues)`` for block-FFN training."""
    kind = kind.strip().lower()
    if kind not in BASIS_KINDS:
        raise ValueError(f"kind must be one of {BASIS_KINDS}")
    ckpt = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
    mean = ckpt["mean_x"].float()
    if kind == "input_pca":
        return mean, ckpt["basis_input_pca"].float(), ckpt["evals_input_pca"].float()
    if kind == "operator_jacobian":
        return mean, ckpt["basis_operator"].float(), ckpt["evals_operator"].float()
    # linear_rrr: right singular vectors of ridge B (input directions)
    b_full = ckpt["b_full"].float()
    _u, s, vt = torch.linalg.svd(b_full, full_matrices=False)
    return mean, vt.T.float(), s**2
