import unittest

from formulate_ordered_interaction import (
    compile_proposal,
    evaluate_held_out,
    select_rate_masked_choice,
)


class OrderedFormulationTests(unittest.TestCase):
    def test_compiler_checks_grounding_without_checking_correct_answer(self):
        summary = {
            "red_then_blue": {"episodes": 2},
            "red_then_yellow": {"episodes": 1},
        }
        proposal = {
            "initiator": "red",
            "second_action": "blue",
            "relation": "after",
            "observed_effect": "suppresses_probe",
            "confidence": 0.7,
        }
        compiled = compile_proposal(proposal, summary, 3)
        self.assertEqual(compiled["second_action"], "blue")

    def test_held_out_evaluation_happens_after_admission(self):
        compiled = {
            "initiator": "red",
            "second_action": "yellow",
            "observed_effect": "suppresses_probe",
        }
        held_out = [
            {"ordered_action": "red_then_yellow", "probe_suppressed": True},
            {"ordered_action": "red_then_yellow", "probe_suppressed": True},
        ]
        self.assertEqual(evaluate_held_out(compiled, held_out)["accuracy"], 1.0)

    def test_rate_mask_blocks_strong_wrong_role_bias(self):
        summary = {
            "red_then_blue": {"episodes": 2, "suppressed": 0},
            "red_then_yellow": {"episodes": 1, "suppressed": 1},
        }
        selected, rates, eligible = select_rate_masked_choice(
            summary, {"blue": 100.0, "yellow": -100.0}
        )
        self.assertEqual(selected, "yellow")
        self.assertEqual(rates, {"blue": 0.0, "yellow": 1.0})
        self.assertEqual(eligible, ["yellow"])

    def test_rate_mask_uses_calibrated_gain_only_for_rate_tie(self):
        summary = {
            "red_then_blue": {"episodes": 2, "suppressed": 1},
            "red_then_yellow": {"episodes": 4, "suppressed": 2},
        }
        selected, _rates, eligible = select_rate_masked_choice(
            summary, {"blue": -0.2, "yellow": 0.3}
        )
        self.assertEqual(selected, "yellow")
        self.assertEqual(eligible, ["blue", "yellow"])


if __name__ == "__main__":
    unittest.main()
