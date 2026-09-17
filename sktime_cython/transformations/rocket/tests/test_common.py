"""Tests for the scaffolding shared by MiniRocket and MultiRocket.

Self-contained (numpy + pytest only) so they run in cibuildwheel's isolated
wheel-test env.
"""

import numpy as np
import pytest

from sktime_cython.transformations.rocket._common import (
    _biases_from_C,
    _fit_dilations,
    _quantiles,
)


def _biases_from_C_loop(C, quantiles, num_features_per_dilation, num_kernels):
    """Reference: the per-combination loop _biases_from_C replaced."""
    biases = np.zeros(
        num_kernels * int(np.sum(num_features_per_dilation)), dtype=np.float32
    )
    feature_index_start = 0
    combination_index = 0
    for dilation_index in range(len(num_features_per_dilation)):
        nfd = num_features_per_dilation[dilation_index]
        for _kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + nfd
            biases[feature_index_start:feature_index_end] = np.quantile(
                C[combination_index],
                quantiles[feature_index_start:feature_index_end],
            ).astype(np.float32)
            feature_index_start = feature_index_end
            combination_index += 1
    return biases


@pytest.mark.parametrize("n_timepoints", [9, 60, 137])
@pytest.mark.parametrize(
    "num_features_per_dilation", [[1], [4, 2, 1], [32, 16, 8, 4, 2, 1, 1]]
)
def test_biases_from_C_matches_loop(num_features_per_dilation, n_timepoints):
    """Vectorized quantiles must match the loop they replaced."""
    num_kernels = 84
    nfpd = np.array(num_features_per_dilation, dtype=np.int32)
    rng = np.random.RandomState(0)
    C = rng.normal(size=(num_kernels * len(nfpd), n_timepoints)).astype(np.float32)
    quantiles = _quantiles(num_kernels * int(nfpd.sum()))

    np.testing.assert_allclose(
        _biases_from_C(C, quantiles, nfpd, num_kernels),
        _biases_from_C_loop(C, quantiles, nfpd, num_kernels),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("num_features", [84, 168, 10_000])
def test_fit_dilations_features_add_up(num_features):
    """Features per dilation sum to the per-kernel budget, one per dilation."""
    dilations, nfpd = _fit_dilations(137, num_features, 32)
    assert len(dilations) == len(nfpd)
    assert nfpd.sum() == max(num_features, 84) // 84
