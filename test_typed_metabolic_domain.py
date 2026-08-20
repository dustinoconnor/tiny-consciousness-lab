import unittest
import tempfile
from pathlib import Path

from typed_metabolic_domain import HeldOutMetabolicRoleLearner, TypedMetabolicRolePool


class TypedMetabolicRoleTests(unittest.TestCase):
    def test_admission_and_held_out_verification_are_separate(self):
        learner = HeldOutMetabolicRoleLearner()
        for feature, relieved in (
            ("blue", True),
            ("yellow", False),
            ("red", False),
            ("blue", True),
        ):
            learner.observe(feature, relieved)

        audit = learner.audit()
        self.assertEqual(audit["status"], "admitted_unverified")
        self.assertEqual(audit["admitted_nutrient"], "blue")
        self.assertEqual(audit["admission_cutoff"], 4)
        self.assertEqual(audit["held_out_confirmations"], 0)

        learner.observe("blue", True)
        self.assertEqual(learner.status, "admitted_unverified")
        learner.observe("yellow", False)
        self.assertEqual(learner.status, "verified_held_out")

    def test_color_swap_swaps_learned_nutrient(self):
        learner = HeldOutMetabolicRoleLearner()
        for feature, relieved in (
            ("yellow", True),
            ("blue", False),
            ("red", False),
            ("yellow", True),
        ):
            learner.observe(feature, relieved)
        self.assertEqual(learner.admitted_nutrient, "yellow")

    def test_unknown_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown_metabolic_feature"):
            TypedMetabolicRolePool().update("purple", True)

    def test_verified_discovery_memory_loads_read_only_into_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "output.json"
            learner = HeldOutMetabolicRoleLearner(memory_path=source)
            for feature, relieved in (
                ("red", False), ("yellow", False), ("blue", True),
                ("blue", True), ("yellow", False), ("blue", True),
            ):
                learner.observe(feature, relieved)
            loaded = HeldOutMetabolicRoleLearner(
                memory_path=output,
                discovery_memory_path=source,
            )
            self.assertTrue(loaded.memory_loaded)
            self.assertEqual(loaded.status, "verified_held_out")
            self.assertEqual(loaded.admitted_nutrient, "blue")
            self.assertEqual(loaded.pool.updates, 6)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
