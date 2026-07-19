import numpy as np

from spatiotemporal_field_workspace_lab import (
    CLASSES,
    GRID_SIZE,
    MODULES,
    TIME_BINS,
    encode,
    make_codebook,
    make_spatial_basis,
    normalize_rows,
    temporal_waves,
)


def test_spatial_basis_has_one_normalized_field_per_module():
    basis = make_spatial_basis()
    assert basis.shape == (MODULES, GRID_SIZE * GRID_SIZE)
    assert np.allclose(np.linalg.norm(basis, axis=1), 1.0)


def test_codebook_spans_all_joint_classes():
    codes, phases = make_codebook(7)
    assert codes.shape[0] * phases.shape[0] == CLASSES


def test_temporal_waves_are_normalized():
    waves = temporal_waves(np.zeros((3, MODULES), dtype=np.float32))
    assert waves.shape == (3, MODULES, TIME_BINS)
    assert np.allclose(np.linalg.norm(waves, axis=2), 1.0)


def test_spatiotemporal_encoding_has_expected_dimension():
    payloads = np.ones((2, MODULES), dtype=np.float32)
    phases = np.zeros((2, MODULES), dtype=np.float32)
    active = np.ones((2, MODULES), dtype=np.float32)
    basis = make_spatial_basis()
    projection = np.ones((MODULES * 3, TIME_BINS * GRID_SIZE * GRID_SIZE), dtype=np.float32)
    encoded = encode("spatiotemporal_field", payloads, phases, active, basis, projection)
    assert encoded.shape == (2, TIME_BINS * GRID_SIZE * GRID_SIZE)
    assert np.allclose(np.linalg.norm(encoded, axis=1), 1.0)


def test_normalize_rows_handles_zero_vector():
    assert np.all(np.isfinite(normalize_rows(np.zeros((2, 4), dtype=np.float32))))
