import numpy as np

from embodied_world_model_lab import Transition
from episodic_playback_lab import EpisodicMemory, RetrievalTelemetry, paired_contrast


def sample_rows():
    obs = np.ones(14, dtype=np.float32) * 0.5
    safe = obs.copy()
    safe[4] = 0.4
    collision = obs.copy()
    collision[-1] = 1.0
    return [
        Transition(obs, 0, safe, 0.0),
        Transition(obs, 1, collision, 1.0),
    ]


def test_memory_binds_positive_progress_and_negative_collision():
    memory = EpisodicMemory.from_transitions(sample_rows(), k=2)
    assert memory.valences[0] > 0
    assert memory.valences[1] < 0


def test_bound_valence_prefers_successful_action():
    memory = EpisodicMemory.from_transitions(sample_rows(), k=2)
    telemetry = RetrievalTelemetry()
    action = memory.choose_action(
        sample_rows()[0].obs,
        np.random.default_rng(3),
        telemetry,
        use_valence=True,
    )
    assert action == 0
    assert telemetry.queries == 1


def test_shuffle_preserves_valence_marginal():
    original = EpisodicMemory.from_transitions(sample_rows(), k=2)
    shuffled = EpisodicMemory.from_transitions(sample_rows(), k=2, shuffle_valence_seed=4)
    assert np.allclose(np.sort(original.valences), np.sort(shuffled.valences))


def test_paired_contrast_counts_wins_and_ties():
    rows = [
        {"geometry": "x", "seed": 1, "condition": "a", "success": 1.0},
        {"geometry": "x", "seed": 1, "condition": "b", "success": 0.0},
        {"geometry": "x", "seed": 2, "condition": "a", "success": 1.0},
        {"geometry": "x", "seed": 2, "condition": "b", "success": 1.0},
    ]
    result = paired_contrast(rows, "a", "b")
    assert result["wins"] == 1
    assert result["ties"] == 1
