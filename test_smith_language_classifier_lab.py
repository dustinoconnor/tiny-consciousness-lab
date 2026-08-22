import unittest

from smith_language_classifier_lab import frozen_cases, parse_choice, prompt_for


class SmithLanguageClassifierLabTests(unittest.TestCase):
    def test_frozen_matrix_counterbalances_identity_order_need_and_mode(self):
        cases = frozen_cases()
        self.assertEqual(len(cases), 24)
        for key, values in {
            "protective_feature": {"yellow": 12, "blue": 12},
            "need": {"pending_hazard": 12, "critical_hunger": 12},
            "classification_mode": {
                "meaningful": 8, "anonymous": 8, "shuffled": 8,
            },
        }.items():
            counts = {value: sum(case[key] == value for case in cases) for value in values}
            self.assertEqual(counts, values)
        self.assertEqual(
            sum(case["candidate_order"] == ["yellow", "blue"] for case in cases),
            12,
        )

    def test_prompt_does_not_contain_scoring_answer_fields(self):
        for case in frozen_cases():
            prompt = prompt_for(case)
            self.assertNotIn("expected_id", prompt)
            self.assertNotIn("expected_feature", prompt)
            self.assertNotIn("correct", prompt.lower())

    def test_choice_parser_is_strict(self):
        self.assertEqual(parse_choice("choice=A"), "A")
        self.assertEqual(parse_choice("Choice = b"), "B")
        self.assertEqual(parse_choice("yellow"), "invalid")


if __name__ == "__main__":
    unittest.main()
