from three_level_memory_lab import CONDITIONS, run_episode


def test_condition_matrix_contains_bound_and_shuffled_hierarchies():
    assert set(CONDITIONS) == {
        "reactive_controller",
        "triggered_mpc",
        "three_level_bound",
        "three_level_shuffled",
    }


def test_reactive_path_requires_no_models_or_memory():
    result = run_episode([], None, "reactive_controller", "circles", 64000, max_steps=5)
    assert result["model_calls"] == 0
    assert result["memory_queries"] == 0
    assert result["reflex_fraction"] == 1.0
