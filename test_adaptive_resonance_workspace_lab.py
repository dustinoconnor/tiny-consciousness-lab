import numpy as np

from adaptive_resonance_workspace_lab import (
    ACTIONS,
    AdaptiveResonanceWorkspace,
    complement_code,
)


def test_complement_code_preserves_constant_activity():
    pattern = complement_code(np.array([0.1, 0.4, 0.9]))
    assert pattern.shape == (6,)
    assert np.isclose(pattern.sum(), 3.0)


def test_mismatch_reset_creates_distinct_category():
    agent = AdaptiveResonanceWorkspace(vigilance=0.9)
    agent.learn(np.zeros(3), "approach", "dark", 1.0)
    agent.learn(np.ones(3), "avoid", "bright", -1.0)
    assert len(agent.templates) == 2
    assert agent.reset_count >= 1


def test_reset_lesion_conflates_mismatched_input():
    agent = AdaptiveResonanceWorkspace(vigilance=0.9, use_reset=False)
    agent.learn(np.zeros(3), "approach", "dark", 1.0)
    agent.learn(np.ones(3), "avoid", "bright", -1.0)
    assert len(agent.templates) == 1


def test_one_workspace_packet_routes_report_and_control():
    agent = AdaptiveResonanceWorkspace(vigilance=0.8)
    observation = np.array([0.8, 0.2, 0.7])
    learned = agent.learn(observation, "approach", "food", 1.0)
    inferred = agent.infer(observation)
    assert learned.category == inferred.category
    assert inferred.action == "approach"
    assert inferred.report == "food"
    assert inferred.resonant


def test_forced_category_intervention_switches_packet_outputs():
    agent = AdaptiveResonanceWorkspace(vigilance=0.9)
    food = np.array([0.9, 0.1, 0.8])
    hazard = np.array([0.1, 0.9, 0.2])
    agent.learn(food, "approach", "food", 1.0)
    agent.learn(hazard, "avoid", "hazard", -1.0)
    baseline = agent.infer(food)
    forced_category = next(
        index for index in range(len(agent.templates))
        if ACTIONS[int(np.argmax(agent.action_counts[index]))] == "avoid"
    )
    forced = agent.infer(food, forced_category=forced_category)
    assert baseline.action == "approach"
    assert baseline.report == "food"
    assert forced.action == "avoid"
    assert forced.report == "hazard"
