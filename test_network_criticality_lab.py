import numpy as np

from network_criticality_lab import (
    driven_lyapunov,
    make_reservoir,
    normalize_spectral_radius,
)


def test_spectral_radius_normalization():
    matrix = normalize_spectral_radius(np.asarray([[2.0, 0.0], [0.0, 0.5]]))
    assert np.isclose(np.max(np.abs(np.linalg.eigvals(matrix))), 1.0)


def test_reservoir_shapes_are_stable():
    weights, inputs = make_reservoir(2, units=20)
    assert weights.shape == (20, 20)
    assert inputs.shape == (20, 2)


def test_lyapunov_measurement_is_finite():
    weights, inputs = make_reservoir(3, units=20)
    exponent = driven_lyapunov(weights, inputs, 1.0, seed=4, steps=20, warmup=5)
    assert np.isfinite(exponent)
