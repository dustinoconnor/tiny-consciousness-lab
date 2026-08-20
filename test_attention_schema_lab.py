import unittest

import numpy as np

from attention_schema_lab import (
    LinearAttentionSchema,
    evaluate,
    run_benchmark,
    schema_features,
    train_attention_schema,
)
from adaptive_gnw_ignition_lab import GovernorConfig
from four_context_conductor_lab import train_conductor
from gnw_ignition_lab import make_gnw_blocks


class AttentionSchemaTests(unittest.TestCase):
    def test_feature_contract_contains_only_router_state(self):
        features = schema_features(
            [0.70, 0.10, 0.10, 0.10],
            [0.60, 0.15, 0.15, 0.10],
            0,
        )
        self.assertEqual(features.shape, (15,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_linear_schema_learns_stable_next_focus(self):
        model = LinearAttentionSchema(seed=4)
        base = schema_features(
            [0.82, 0.06, 0.06, 0.06],
            [0.78, 0.08, 0.07, 0.07],
            0,
        )
        features = np.stack([base] * 24)
        targets = np.zeros(24, dtype=np.int64)
        before = model.probabilities(base)[0]
        model.fit(features, targets, epochs=80)
        after = model.probabilities(base)[0]
        self.assertGreater(after, before)
        self.assertEqual(int(np.argmax(model.probabilities(base))), 0)

    def test_lesion_matches_no_schema_behavior(self):
        conductor = train_conductor(31, steps=900)
        config = GovernorConfig(0.60, 0.05, 2, 0.0)
        training = make_gnw_blocks(32, 12, 6)
        schema = train_attention_schema(
            conductor, training, config, seed=33
        )
        evaluation = make_gnw_blocks(34, 14, 6)
        baseline = evaluate(
            "adaptive_gnw", conductor, evaluation, 35, config, schema
        )
        lesion = evaluate(
            "ast_schema_lesion", conductor, evaluation, 35, config, schema
        )
        self.assertEqual(
            [row["action_specialist"] for row in baseline],
            [row["action_specialist"] for row in lesion],
        )
        self.assertEqual(
            [row["utility"] for row in baseline],
            [row["utility"] for row in lesion],
        )

    def test_quick_benchmark_reports_registered_conditions(self):
        payload = run_benchmark(
            seed_count=1,
            train_steps=900,
            governor_train_blocks=8,
            schema_train_blocks=10,
            eval_blocks=12,
            steps_per_block=6,
        )
        self.assertEqual(payload["schema_parameter_count"], 60)
        self.assertEqual(set(payload["summary"]), {
            "adaptive_gnw",
            "ast_schema",
            "ast_schema_shuffled",
            "ast_schema_lesion",
        })


if __name__ == "__main__":
    unittest.main()
