import json
import unittest

from tiny_scientist import (
    MetabolicEpisode,
    PassiveScientificGNW,
    analyze,
    bound_candidate_payloads,
    bound_schema_prompt,
    causal_ir_prompt,
    causal_ir_candidate_texts,
    causal_ir_v2_candidate_texts,
    labeled_causal_ir_candidate_texts,
    labeled_causal_ir_syntax_candidate_texts,
    extract_causal_ir_v2,
    compile_formal_falsifier,
    extract_causal_ir,
    extract_json_object,
    evidence_summary,
    hypothesis_prompt,
    metabolic_episodes,
    validate_bound_hypothesis,
    validate_hypothesis,
)
from tiny_scientist_production_memory import (
    VerifiedProductionMemory,
    commit_verified_rule,
)


class TinyScientistTests(unittest.TestCase):
    def test_bound_verifier_accepts_grounded_negative_effect(self):
        summary = {
            "feature_outcomes": {
                "silver": {
                    "mean_pressure_delta": -0.34,
                    "mean_positive_delay_seconds": 3.0,
                },
                "gold": {
                    "mean_pressure_delta": 0.0,
                    "mean_positive_delay_seconds": None,
                },
            }
        }
        result = validate_bound_hypothesis(
            {
                "target_cause": "silver",
                "comparison_feature": "gold",
                "observed_effect": "pressure_decrease",
                "latency_seconds": 3.0,
                "confidence": 0.5,
            },
            summary,
        )
        self.assertEqual(
            result["hypothesis"]["observed_effect"], "pressure_decrease"
        )

    def episode(self, feature="red", delta=0.34, intervening=False):
        return MetabolicEpisode(
            feature=feature,
            pickup_time=0.0,
            pressure_before=0.0,
            peak_pressure=delta,
            pressure_delta=delta,
            peak_delay_seconds=10.0,
            intervening_pickup=intervening,
        )

    def test_passive_gnw_ignites_for_isolated_surprise_without_authority(self):
        gate = PassiveScientificGNW(refractory_episodes=0)
        broadcast = gate.observe(self.episode())
        self.assertIsNotNone(broadcast)
        self.assertEqual(gate.action_influence, 0)

    def test_passive_gnw_holds_confounded_low_salience_episode(self):
        gate = PassiveScientificGNW(refractory_episodes=0)
        broadcast = gate.observe(
            self.episode(feature="blue", delta=0.04, intervening=True)
        )
        self.assertIsNone(broadcast)

    def test_hypothesis_parser_enforces_zero_authority(self):
        raw = "prefix " + json.dumps(
            {
                "condition": "consume red object",
                "proposed_cause": "red objects may trigger internal pressure",
                "predicted_effect": "pressure rises",
                "delay_seconds": 10,
                "comparison": "blue objects",
                "falsifier": "matched red objects do not precede pressure",
                "confidence": 0.7,
            }
        )
        hypothesis = validate_hypothesis(extract_json_object(raw))
        self.assertEqual(hypothesis["status"], "proposed_unverified")
        self.assertEqual(hypothesis["authority"], 0.0)

    def test_semantic_verifier_rejects_aggregate_as_cause(self):
        payload = {
            "condition": "episode_count",
            "proposed_cause": "increased positive pressure",
            "predicted_effect": "blue pressure increases",
            "delay_seconds": 10,
            "comparison": "red episodes",
            "falsifier": "blue episodes",
            "confidence": 0.9,
        }
        summary = {"feature_outcomes": {"blue": {}, "red": {}}}
        with self.assertRaisesRegex(ValueError, "cause_has_no_observed_feature"):
            validate_hypothesis(payload, summary)

    def test_semantic_verifier_rejects_reversed_feature_direction(self):
        payload = {
            "condition": "consume a blue object",
            "proposed_cause": "blue objects trigger internal pressure",
            "predicted_effect": "internal pressure rises",
            "delay_seconds": 10,
            "comparison": "red objects",
            "falsifier": "blue pickups do not precede a pressure rise",
            "confidence": 0.7,
        }
        summary = {
            "feature_outcomes": {
                "blue": {
                    "mean_pressure_delta": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.0,
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "reverses_observed_direction"):
            validate_hypothesis(payload, summary)

    def test_semantic_verifier_accepts_grounded_direction_and_delay(self):
        payload = {
            "condition": "consume a red object",
            "proposed_cause": "red objects may trigger internal pressure",
            "predicted_effect": "internal pressure rises",
            "delay_seconds": 10,
            "comparison": "blue objects",
            "falsifier": "matched red pickups do not precede a pressure rise",
            "confidence": 0.7,
        }
        summary = {
            "feature_outcomes": {
                "blue": {
                    "mean_pressure_delta": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.0,
                },
            }
        }
        result = validate_hypothesis(payload, summary)
        self.assertEqual(result["status"], "proposed_unverified")
        self.assertEqual(result["authority"], 0.0)

    def test_semantic_verifier_rejects_falsifier_about_control_feature(self):
        payload = {
            "condition": "pickup",
            "proposed_cause": "red",
            "predicted_effect": "increased internal pressure",
            "delay_seconds": 10.464,
            "comparison": "blue",
            "falsifier": (
                "Observe a decrease in internal pressure after a blue pickup. "
                "This would contradict the prediction that red increases pressure."
            ),
            "confidence": 0.95,
        }
        summary = {
            "feature_outcomes": {
                "blue": {
                    "mean_pressure_delta": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.464,
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "falsifier_targets_wrong_feature"):
            validate_hypothesis(payload, summary)

    def test_offline_pipeline_separates_red_and_blue_outcomes(self):
        rows = []
        total = 0
        for second in range(45):
            feature = "none"
            if second == 1:
                total += 1
                feature = "blue"
            if second == 21:
                total += 1
                feature = "red"
            pressure = 0.34 if 31 <= second <= 34 else 0.0
            rows.append(
                {
                    "time": float(second),
                    "mushroom_pickups_total": total,
                    "mushroom_feature": feature,
                    "metabolic_pressure": pressure,
                }
            )
        result = analyze(rows)
        features = result["summary"]["feature_outcomes"]
        self.assertLess(features["blue"]["mean_pressure_delta"], 0.1)
        self.assertGreater(features["red"]["mean_pressure_delta"], 0.3)
        self.assertEqual(result["gnw"]["action_influence"], 0)

    def test_durable_red_counter_repairs_missing_transient_feature(self):
        rows = []
        for second in range(18):
            rows.append(
                {
                    "time": float(second),
                    "mushroom_pickups_total": 1 if second >= 1 else 0,
                    "red_mushroom_pickups_total": 1 if second >= 1 else 0,
                    "mushroom_feature": "none",
                    "metabolic_pressure": 0.34 if 11 <= second <= 14 else 0.0,
                }
            )
        result = analyze(rows)
        self.assertEqual(result["summary"]["feature_outcomes"]["red"]["episodes"], 1)
        self.assertNotIn("none", result["summary"]["feature_outcomes"])

    def test_isolation_rejects_control_immediately_after_causal_pickup(self):
        rows = []
        for second in range(24):
            total = 0
            red_total = 0
            if second >= 1:
                total = 1
                red_total = 1
            if second >= 2:
                total = 2
            rows.append(
                {
                    "time": float(second),
                    "mushroom_pickups_total": total,
                    "red_mushroom_pickups_total": red_total,
                    "mushroom_feature": "none",
                    "metabolic_pressure": 0.34 if 11 <= second <= 15 else 0.0,
                }
            )
        episodes = metabolic_episodes(rows)
        self.assertEqual([episode.feature for episode in episodes], ["red", "blue"])
        self.assertTrue(all(episode.intervening_pickup for episode in episodes))
        summary = evidence_summary(episodes, [])
        self.assertEqual(summary["feature_outcomes"]["blue"]["isolated_episodes"], 0)

    def test_homeostatic_filter_retains_both_features_without_gnw_metadata(self):
        summary = {
            "feature_outcomes": {
                "blue": {
                    "isolated_episodes": 11,
                    "mean_pressure_delta": 0.0,
                    "positive_pressure_fraction": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "isolated_episodes": 2,
                    "mean_pressure_delta": 0.34,
                    "positive_pressure_fraction": 1.0,
                    "mean_positive_delay_seconds": 10.464,
                },
            },
            "gnw_broadcast_features": {"red": 2},
        }
        original = hypothesis_prompt(summary, "original")
        filtered = hypothesis_prompt(summary, "homeostatic_filtered")
        self.assertIn("Passive GNW", original)
        self.assertNotIn("GNW", filtered)
        self.assertIn("observed feature blue", filtered)
        self.assertIn("observed feature red", filtered)
        self.assertIn("without selecting its answer", filtered)

    def test_unknown_evidence_interface_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported_evidence_interface"):
            hypothesis_prompt({}, "answer_leaking")

    def test_bound_schema_compiles_falsifier_from_single_cause_binding(self):
        summary = {
            "feature_outcomes": {
                "blue": {
                    "mean_pressure_delta": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.464,
                },
            }
        }
        result = validate_bound_hypothesis(
            {
                "target_cause": "red",
                "comparison_feature": "blue",
                "observed_effect": "pressure_increase",
                "latency_seconds": 10.464,
                "confidence": 0.8,
            },
            summary,
        )
        self.assertEqual(
            result["falsifier_schema"]["action_to_retest"],
            result["hypothesis"]["target_cause"],
        )
        self.assertEqual(result["authority"], 0.0)

    def test_bound_schema_rejects_reversed_cause(self):
        summary = {
            "feature_outcomes": {
                "blue": {"mean_pressure_delta": 0.0},
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.464,
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "bound_cause_reverses_evidence"):
            validate_bound_hypothesis(
                {
                    "target_cause": "blue",
                    "comparison_feature": "red",
                    "observed_effect": "pressure_increase",
                    "latency_seconds": 10.464,
                    "confidence": 0.8,
                },
                summary,
            )

    def test_bound_prompt_has_one_cause_field_and_no_falsifier_fields(self):
        summary = {"feature_outcomes": {"blue": {}, "red": {}}}
        prompt = bound_schema_prompt(summary)
        self.assertIn("target_cause", prompt)
        self.assertNotIn("action_to_retest", prompt)
        self.assertNotIn("expected_contradictory_outcome", prompt)

    def test_causal_ir_round_trip_uses_canonical_verifier(self):
        summary = {
            "feature_outcomes": {
                "blue": {"mean_pressure_delta": 0.0},
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.464,
                },
            }
        }
        payload = extract_causal_ir("C1 red blue + 10.464 0.95")
        result = validate_bound_hypothesis(payload, summary)
        self.assertEqual(result["hypothesis"]["target_cause"], "red")
        self.assertEqual(
            result["hypothesis"]["observed_effect"], "pressure_increase"
        )
        self.assertEqual(result["falsifier_schema"]["action_to_retest"], "red")

    def test_causal_ir_rejects_extra_or_malformed_records(self):
        with self.assertRaisesRegex(ValueError, "no_single_causal_ir"):
            extract_causal_ir("C1 red blue + ten .9")
        with self.assertRaisesRegex(ValueError, "no_single_causal_ir"):
            extract_causal_ir("C1 red blue + 10 .9\nC1 blue red 0 0 .5")

    def test_causal_ir_prompt_is_symmetric_and_non_answer_leaking(self):
        summary = {
            "feature_outcomes": {
                "blue": {
                    "isolated_episodes": 3,
                    "mean_pressure_delta": 0.0,
                    "positive_pressure_fraction": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "isolated_episodes": 3,
                    "mean_pressure_delta": 0.34,
                    "positive_pressure_fraction": 1.0,
                    "mean_positive_delay_seconds": 10.464,
                },
            }
        }
        canonical = causal_ir_prompt(summary, evidence_order="canonical")
        reversed_prompt = causal_ir_prompt(summary, evidence_order="reversed")
        self.assertIn("C1 cause comparison effect delay confidence", canonical)
        self.assertNotIn("target_cause", canonical)
        self.assertLess(canonical.index("blue 3"), canonical.index("red 3"))
        self.assertLess(reversed_prompt.index("red 3"), reversed_prompt.index("blue 3"))

    def test_causal_ir_candidates_match_json_candidate_semantics(self):
        summary = {
            "feature_outcomes": {
                "blue": {"mean_positive_delay_seconds": None},
                "red": {"mean_positive_delay_seconds": 10.464},
            }
        }
        ir_candidates = causal_ir_candidate_texts(summary)
        self.assertEqual(len(ir_candidates), len(bound_candidate_payloads(summary)))
        self.assertIn("C1 red blue + 10.464 0.5", ir_candidates)
        self.assertIn("C1 blue red - 0.0 0.5", ir_candidates)
        labeled = labeled_causal_ir_candidate_texts(summary)
        self.assertEqual(len(labeled), len(bound_candidate_payloads(summary)))
        self.assertIn("L1 c red k blue e + t 10.464 q 0.5", labeled)
        self.assertIn("L1 c blue k red e - t 0.0 q 0.5", labeled)

    def test_syntax_only_l1_candidates_allow_repeated_roles(self):
        summary = {
            "feature_outcomes": {
                "blue": {"mean_positive_delay_seconds": None},
                "red": {"mean_positive_delay_seconds": 10.464},
            }
        }
        role_bound = labeled_causal_ir_candidate_texts(summary)
        syntax_only = labeled_causal_ir_syntax_candidate_texts(summary)
        self.assertEqual(len(syntax_only), 2 * len(role_bound))
        self.assertIn("L1 c red k red e + t 10.464 q 0.5", syntax_only)
        self.assertNotIn("L1 c red k red e + t 10.464 q 0.5", role_bound)
        self.assertIn("L1 c red k blue e + t 10.464 q 0.5", syntax_only)

    def test_abstract_scaffolds_have_same_non_current_binding(self):
        summary = {"feature_outcomes": {"blue": {}, "red": {}}}
        json_prompt = bound_schema_prompt(
            summary, include_abstract_example=True
        )
        ir_prompt = causal_ir_prompt(summary, include_abstract_example=True)
        self.assertIn("feature alpha", json_prompt)
        self.assertIn("feature alpha", ir_prompt)
        self.assertIn('"target_cause":"alpha"', json_prompt)
        self.assertIn("C1 alpha beta - 5 0.8", ir_prompt)
        self.assertIn("C1 gamma delta + 7 0.8", ir_prompt)
        self.assertIn("not current evidence", json_prompt)
        self.assertIn("not current evidence", ir_prompt)

    def test_causal_ir_v2_dependency_order_round_trip(self):
        payload = extract_causal_ir_v2("C2 + 10.464 red blue 0.9")
        self.assertEqual(payload["target_cause"], "red")
        self.assertEqual(payload["observed_effect"], "pressure_increase")
        summary = {
            "feature_outcomes": {
                "blue": {"mean_positive_delay_seconds": None},
                "red": {"mean_positive_delay_seconds": 10.464},
            }
        }
        self.assertIn(
            "C2 + 10.464 red blue 0.5",
            causal_ir_v2_candidate_texts(summary),
        )

    def test_constrained_candidates_are_symmetric_and_non_leaking(self):
        summary = {
            "feature_outcomes": {
                "blue": {"mean_positive_delay_seconds": None},
                "red": {"mean_positive_delay_seconds": 10.464},
            }
        }
        candidates = bound_candidate_payloads(summary)
        by_target = {
            target: {
                (item["observed_effect"], item["latency_seconds"])
                for item in candidates
                if item["target_cause"] == target
            }
            for target in ("blue", "red")
        }
        self.assertEqual(by_target["blue"], by_target["red"])
        self.assertTrue(
            all(
                item["target_cause"] != item["comparison_feature"]
                for item in candidates
            )
        )

    def test_reversed_evidence_order_changes_lines_not_candidate_space(self):
        summary = {
            "feature_outcomes": {
                "blue": {"mean_positive_delay_seconds": None},
                "red": {"mean_positive_delay_seconds": 10.464},
            }
        }
        canonical = bound_schema_prompt(summary, evidence_order="canonical")
        reversed_prompt = bound_schema_prompt(summary, evidence_order="reversed")
        self.assertLess(
            canonical.index("observed feature blue"),
            canonical.index("observed feature red"),
        )
        self.assertLess(
            reversed_prompt.index("observed feature red"),
            reversed_prompt.index("observed feature blue"),
        )
        reordered_summary = {
            "feature_outcomes": dict(
                reversed(list(summary["feature_outcomes"].items()))
            )
        }
        self.assertEqual(
            bound_candidate_payloads(summary),
            bound_candidate_payloads(reordered_summary),
        )

    def test_formal_compiler_rebinds_falsifier_to_model_cause(self):
        summary = {
            "feature_outcomes": {
                "blue": {
                    "mean_pressure_delta": 0.0,
                    "mean_positive_delay_seconds": None,
                },
                "red": {
                    "mean_pressure_delta": 0.34,
                    "mean_positive_delay_seconds": 10.464,
                },
            }
        }
        result = compile_formal_falsifier(
            {
                "condition": "pickup",
                "proposed_cause": "red",
                "predicted_effect": "increased internal pressure",
                "delay_seconds": 10.464,
                "comparison": "blue",
                "falsifier": "a decrease after blue would contradict red",
                "confidence": 0.95,
            },
            summary,
        )
        self.assertEqual(
            result["falsifier_schema"]["action_to_retest"],
            "red",
        )
        self.assertIn("Matched red pickups do not", result["falsifier"])
        self.assertEqual(result["authority"], 0.0)

    def test_formal_compiler_rejects_unbound_cause(self):
        summary = {"feature_outcomes": {"blue": {}, "red": {}}}
        with self.assertRaisesRegex(ValueError, "requires_single_bound_cause"):
            compile_formal_falsifier(
                {
                    "condition": "pickup",
                    "proposed_cause": "positive-pressure fraction",
                    "predicted_effect": "increased internal pressure",
                },
                summary,
            )

    def test_verified_rule_commit_is_read_only_and_idempotent(self):
        import tempfile
        from pathlib import Path

        summary = {
            "date": "2026-08-01",
            "verdict": "preregistered_verification_pass",
            "rule_status": "eligible_for_verified_shadow_production_memory",
            "hypothesis_under_test": {
                "cause": "red pickup",
                "effect": "increased internal pressure",
                "predicted_delay_seconds": 10.464,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.json"
            memory = Path(directory) / "memory.json"
            source.write_text(json.dumps(summary), encoding="utf-8")
            first = commit_verified_rule(source, memory)
            second = commit_verified_rule(source, memory)
            audit = VerifiedProductionMemory(memory).audit()
        self.assertEqual(first["committed_rule"], "red_pickup_delayed_pressure_increase_v1")
        self.assertEqual(second["rule_count"], 1)
        self.assertEqual(audit["authority"], 0.0)
        self.assertEqual(audit["control_permissions"], [])
        self.assertEqual(audit["action_influence"], 0)


if __name__ == "__main__":
    unittest.main()
