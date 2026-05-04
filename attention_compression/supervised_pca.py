from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CenteredStats:
    n: int
    x_mean: np.ndarray
    y_mean: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray


def centered_stats_from_raw(
    *,
    n: int,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: np.ndarray,
) -> CenteredStats:
    if n <= 1:
        raise ValueError("Need at least two samples")
    x_mean = sum_x / n
    y_mean = sum_y / n
    sxx = xtx - n * np.outer(x_mean, x_mean)
    sxy = xty - n * np.outer(x_mean, y_mean)
    syy = yty - n * np.outer(y_mean, y_mean)
    return CenteredStats(n=n, x_mean=x_mean, y_mean=y_mean, sxx=sxx, sxy=sxy, syy=syy)


def fit_ridge_rrr(
    stats: CenteredStats,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit full ridge regression and return rank directions.

    The rank-r supervised PCA / RRR predictor is:

        y = (x - x_mean) @ B_full @ V_r @ V_r.T + y_mean

    where `V_r` are the top eigenvectors of the fitted-response covariance in
    target space.
    """
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    x_dim = stats.sxx.shape[0]
    regularized = stats.sxx + ridge * np.eye(x_dim, dtype=stats.sxx.dtype)
    b_full = np.linalg.solve(regularized, stats.sxy)
    fitted_cov = b_full.T @ stats.sxx @ b_full
    eigvals, eigvecs = np.linalg.eigh(fitted_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return b_full, eigvecs, eigvals


def rrr_coefficients(
    b_full: np.ndarray,
    eigvecs: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank > eigvecs.shape[1]:
        raise ValueError(f"rank={rank} exceeds target dimension {eigvecs.shape[1]}")
    directions = eigvecs[:, :rank]
    return b_full @ directions @ directions.T


def predict_centered(x: np.ndarray, b_rank: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    return (x - x_mean) @ b_rank + y_mean
