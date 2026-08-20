import unittest

from counterbalance_ordered_formulation import question_for


class CounterbalancedOrderedFormulationTests(unittest.TestCase):
    def test_color_assignment_and_presentation_order_are_independent(self):
        blue_first = question_for("yellow", ("blue", "yellow"), "then")
        yellow_first = question_for("yellow", ("yellow", "blue"), "then")
        self.assertIn("red then yellow: 1 of 1 suppressed", blue_first)
        self.assertIn("red then yellow: 1 of 1 suppressed", yellow_first)
        self.assertLess(blue_first.index("red then blue"), blue_first.index("red then yellow"))
        self.assertLess(yellow_first.index("red then yellow"), yellow_first.index("red then blue"))

    def test_after_phrasing_preserves_event_semantics(self):
        prompt = question_for("blue", ("yellow", "blue"), "after")
        self.assertIn("blue after red: 1 of 1 suppressed", prompt)
        self.assertIn("yellow after red: 0 of 2 suppressed", prompt)

    def test_neutral_calibration_does_not_encode_suppressor(self):
        blue = question_for("blue", ("blue", "yellow"), "then", neutral=True)
        yellow = question_for("yellow", ("blue", "yellow"), "then", neutral=True)
        self.assertEqual(blue, yellow)


if __name__ == "__main__":
    unittest.main()
