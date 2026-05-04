from __future__ import annotations

import numpy as np


def covariance_from_raw(*, n: int, sum_x: np.ndarray, xtx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        raise ValueError("Need at least two samples")
    mean = sum_x / n
    centered = xtx - n * np.outer(mean, mean)
    cov = centered / (n - 1)
    return mean, cov


def pca_spectrum(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals[::-1], 0.0)
    total = float(eigvals.sum())
    if total <= 0:
        return eigvals, np.zeros_like(eigvals)
    return eigvals, np.cumsum(eigvals) / total


def dims_for_thresholds(cumulative: np.ndarray, thresholds: list[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        out[f"{threshold:.4f}"] = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    return out
