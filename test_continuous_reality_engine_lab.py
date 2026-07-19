import numpy as np

from continuous_reality_engine_lab import (
    CONDITIONS,
    correct_latent,
    paired_contrast,
    run_episode,
    sanitize_observation,
)


def test_observation_sanitizer_preserves_channel_ranges():
    obs = np.array([2, -1, 3, -4, 2] + [4] * 8 + [-2], dtype=np.float32)
    clean = sanitize_observation(obs)
    assert np.all((clean[0:2] >= 0) & (clean[0:2] <= 1))
    assert np.all((clean[2:4] >= -1) & (clean[2:4] <= 1))
    assert np.all((clean[4:] >= 0) & (clean[4:] <= 1))


def test_full_grounding_replaces_prediction_with_observation():
    predicted = np.zeros(14, dtype=np.float32)
    observed = np.ones(14, dtype=np.float32)
    corrected = correct_latent(predicted, observed, gain=1.0)
    assert np.allclose(corrected, observed)


def test_reactive_condition_never_calls_forward_model():
    result = run_episode([], "reactive_controller", "circles", seed=42000, max_steps=8)
    assert result["model_calls"] == 0
    assert result["planning_steps"] == 0


def test_condition_set_contains_requested_ablation_families():
    assert set(CONDITIONS) == {
        "reactive_controller",
        "triggered_mpc",
        "continuous_generative",
        "continuous_grounded",
        "ungrounded_generative",
    }


def test_paired_contrast_counts_matched_outcomes():
    rows = [
        {"geometry": "x", "seed": 1, "condition": "a", "success": 1.0},
        {"geometry": "x", "seed": 1, "condition": "b", "success": 0.0},
        {"geometry": "x", "seed": 2, "condition": "a", "success": 1.0},
        {"geometry": "x", "seed": 2, "condition": "b", "success": 1.0},
    ]
    result = paired_contrast(rows, "a", "b")
    assert result["wins"] == 1
    assert result["losses"] == 0
    assert result["ties"] == 1
