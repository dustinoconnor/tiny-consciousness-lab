import unittest

from causal_dsl_lora import (
    HELDOUT_PAIRS,
    TRAIN_FEATURES,
    build_curriculum,
    exact_semantic_match,
    parse_prediction,
    prompt_for,
    target_for,
)


class CausalDslLoraTests(unittest.TestCase):
    def test_curriculum_holds_out_unity_features_and_balances_orders(self):
        train, heldout = build_curriculum(train_size=64, heldout_size=16)
        train_text = " ".join(target_for(example, "c1") for example in train)
        self.assertNotIn("red", train_text)
        self.assertNotIn("blue", train_text)
        self.assertEqual({example.evidence_order for example in train}, {"canonical", "reversed"})
        heldout_names = {name for pair in HELDOUT_PAIRS for name in pair}
        self.assertTrue(all(example.target in heldout_names for example in heldout))
        self.assertTrue(set(TRAIN_FEATURES).isdisjoint(heldout_names))
        target_counts = {
            feature: sum(example.target == feature for example in train)
            for feature in TRAIN_FEATURES
        }
        comparison_counts = {
            feature: sum(example.comparison == feature for example in train)
            for feature in TRAIN_FEATURES
        }
        self.assertTrue(all(target_counts.values()))
        self.assertTrue(all(comparison_counts.values()))
        self.assertLessEqual(max(target_counts.values()) - min(target_counts.values()), 5)

    def test_json_and_c1_targets_have_identical_semantics(self):
        train, _heldout = build_curriculum(train_size=1, heldout_size=1)
        example = train[0]
        json_prediction = parse_prediction(target_for(example, "json"), "json")
        c1_prediction = parse_prediction(target_for(example, "c1"), "c1")
        self.assertEqual(json_prediction, c1_prediction)
        self.assertTrue(exact_semantic_match(json_prediction, example))
        l1_prediction = parse_prediction(target_for(example, "l1"), "l1")
        self.assertEqual(json_prediction, l1_prediction)
        h1_prediction = parse_prediction(target_for(example, "h1"), "h1")
        self.assertEqual(json_prediction, h1_prediction)

    def test_prompts_share_evidence_but_have_distinct_output_grammars(self):
        train, _heldout = build_curriculum(train_size=1, heldout_size=1)
        example = train[0]
        json_prompt = prompt_for(example, "json")
        c1_prompt = prompt_for(example, "c1")
        for feature in (example.target, example.comparison):
            self.assertIn(feature, json_prompt)
            self.assertIn(feature, c1_prompt)
        self.assertIn("JSON object", json_prompt)
        self.assertIn("C1 cause comparison", c1_prompt)
        l1_prompt = prompt_for(example, "l1")
        self.assertIn("L1 c cause k comparison", l1_prompt)
        h1_prompt = prompt_for(example, "h1")
        self.assertIn("H1 cause feature control feature", h1_prompt)


if __name__ == "__main__":
    unittest.main()
