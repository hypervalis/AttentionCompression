from __future__ import annotations

import numpy as np

from attention_compression.supervised_pca import centered_stats_from_raw, fit_ridge_rrr, predict_centered, rrr_coefficients


def test_rrr_rank_recovers_low_rank_linear_map() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(512, 6))
    u = rng.normal(size=(6, 1))
    v = rng.normal(size=(1, 3))
    b_true = u @ v
    y = x @ b_true + 0.01 * rng.normal(size=(512, 3))

    stats = centered_stats_from_raw(
        n=x.shape[0],
        sum_x=x.sum(axis=0),
        sum_y=y.sum(axis=0),
        xtx=x.T @ x,
        xty=x.T @ y,
        yty=y.T @ y,
    )
    b_full, eigvecs, eigvals = fit_ridge_rrr(stats, ridge=1e-6)
    b_rank1 = rrr_coefficients(b_full, eigvecs, rank=1)
    pred = predict_centered(x, b_rank1, stats.x_mean, stats.y_mean)

    assert eigvals[0] > eigvals[1] * 100
    assert np.mean((pred - y) ** 2) < 1e-3
