import unittest

from computational_invariance_lab import (
    MessagePassingEgo,
    Observation,
    ObservationalReplay,
    SymbolicEgo,
    VectorEgo,
    generate_trace,
    output_pair_equal,
    outputs_equal,
    run_trace,
)


class ComputationalInvarianceTests(unittest.TestCase):
    def test_three_realizations_match_on_ordinary_trace(self):
        trace = generate_trace(9, 30)
        symbolic = run_trace(SymbolicEgo, trace)
        vector = run_trace(VectorEgo, trace)
        messages = run_trace(MessagePassingEgo, trace)
        self.assertTrue(all(outputs_equal(a, b) for a, b in zip(symbolic, vector)))
        self.assertTrue(all(outputs_equal(a, b) for a, b in zip(symbolic, messages)))

    def test_forced_workspace_changes_report_and_action(self):
        observation = Observation(1.0, 0.0, 0.0, 0.8, 0.0)
        machine = SymbolicEgo()
        baseline = machine.step(observation)
        forced_machine = SymbolicEgo()
        forced = forced_machine.step(observation, {"force_workspace": "danger"})
        self.assertNotEqual(baseline.action, forced.action)
        self.assertNotEqual(baseline.report, forced.report)

    def test_replay_matches_observed_output_but_ignores_intervention(self):
        observation = Observation(1.0, 0.0, 0.0, 0.8, 0.0)
        baseline = SymbolicEgo().step(observation)
        replay = ObservationalReplay([baseline])
        intervened = SymbolicEgo().step(observation, {"force_workspace": "danger"})
        self.assertTrue(output_pair_equal(baseline, replay.output_at(0)))
        self.assertFalse(output_pair_equal(intervened, replay.output_at(0, {"force_workspace": "danger"})))

    def test_internal_intervention_profile_is_representation_invariant(self):
        observation = Observation(0.2, 0.7, 0.4, -0.5, 0.3)
        outputs = [
            machine().step(observation, {"force_workspace": "obstacle"})
            for machine in (SymbolicEgo, VectorEgo, MessagePassingEgo)
        ]
        self.assertTrue(outputs_equal(outputs[0], outputs[1]))
        self.assertTrue(outputs_equal(outputs[0], outputs[2]))


if __name__ == "__main__":
    unittest.main()
