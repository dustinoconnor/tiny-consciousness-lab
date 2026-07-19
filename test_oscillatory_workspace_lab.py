import unittest

from oscillatory_workspace_lab import run_experiment


class OscillatoryWorkspaceLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, _ = run_experiment(trials=240, seed=41)

    def metric(self, condition, name="binding_correct"):
        return self.summary["conditions"][condition][name]

    def test_phase_lock_beats_same_frequency_without_locking(self):
        self.assertGreater(
            self.metric("coherent_40hz") - self.metric("same_40hz_unlocked"),
            0.50,
        )

    def test_frequency_mismatch_degrades_sustained_binding(self):
        self.assertGreater(
            self.metric("coherent_40hz") - self.metric("mixed_frequency"),
            0.50,
        )

    def test_targeted_phase_shift_is_causally_damaging(self):
        self.assertGreater(
            self.metric("coherent_40hz") - self.metric("phase_shift_intervention"),
            0.80,
        )
        self.assertGreater(self.metric("phase_shift_intervention", "false_binding"), 0.80)

    def test_40hz_is_not_privileged_in_abstract_software(self):
        difference = abs(
            self.metric("coherent_40hz") - self.metric("coherent_20hz_control")
        )
        self.assertLess(difference, 0.03)


if __name__ == "__main__":
    unittest.main()
