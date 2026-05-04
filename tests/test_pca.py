from __future__ import annotations

import numpy as np

from attention_compression.pca import covariance_from_raw, dims_for_thresholds, pca_spectrum


def test_pca_spectrum_detects_dominant_dimension() -> None:
    rng = np.random.default_rng(9)
    x = np.column_stack(
        [
            5.0 * rng.normal(size=512),
            0.5 * rng.normal(size=512),
            0.1 * rng.normal(size=512),
        ]
    )
    _, cov = covariance_from_raw(n=x.shape[0], sum_x=x.sum(axis=0), xtx=x.T @ x)
    eigvals, cumulative = pca_spectrum(cov)
    dims = dims_for_thresholds(cumulative, [0.9, 0.99])

    assert eigvals[0] > eigvals[1] * 50
    assert dims["0.9000"] == 1
    assert dims["0.9900"] <= 2
