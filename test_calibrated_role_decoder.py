import unittest

from calibrated_role_decoder import (
    CALIBRATION_LABELS,
    CALIBRATION_REGIMES,
    RESERVED_LABELS,
    RESERVED_REGIMES,
    ROLES,
    apply_role_bias,
    apply_calibrated_decoder,
    canonicalize_records,
    estimate_role_bias,
    fit_rate_weight,
    make_records,
    matrix_spec,
    payload_hash,
)


class CalibratedRoleDecoderTests(unittest.TestCase):
    def test_canonicalization_is_order_invariant_and_outcome_blind(self):
        records = make_records(
            CALIBRATION_LABELS, CALIBRATION_LABELS[0], CALIBRATION_REGIMES[0]
        )
        reversed_records = list(reversed(records))
        text, mapping = canonicalize_records(records)
        reversed_text, reversed_mapping = canonicalize_records(reversed_records)
        self.assertEqual(text, reversed_text)
        self.assertEqual(mapping, reversed_mapping)
        changed = make_records(
            CALIBRATION_LABELS, CALIBRATION_LABELS[2], CALIBRATION_REGIMES[1]
        )
        self.assertEqual(mapping, canonicalize_records(changed)[1])

    def test_calibration_and_reserve_use_disjoint_labels_and_ratios(self):
        self.assertTrue(set(CALIBRATION_LABELS).isdisjoint(RESERVED_LABELS))
        calibration_counts = {
            (tuple(item["positive"]), tuple(item["control"]))
            for item in CALIBRATION_REGIMES
        }
        reserved_counts = {
            (tuple(item["positive"]), tuple(item["control"]))
            for item in RESERVED_REGIMES
        }
        self.assertTrue(calibration_counts.isdisjoint(reserved_counts))

    def test_every_role_is_equally_assigned_in_each_matrix(self):
        for labels, regimes in (
            (CALIBRATION_LABELS, CALIBRATION_REGIMES),
            (RESERVED_LABELS, RESERVED_REGIMES),
        ):
            rows = matrix_spec(labels, regimes)
            assignments = {role: 0 for role in ROLES}
            for row in rows:
                _, mapping = canonicalize_records(row["records"])
                assignments[mapping[row["suppressor"]]] += 1
            self.assertEqual(set(assignments.values()), {len(regimes)})

    def test_bias_estimator_uses_balanced_means_not_answers(self):
        rows = []
        for index, role in enumerate(ROLES):
            rows.append(
                {
                    "correct_role": role,
                    "neutral_adjusted_gains": {
                        "role_0": 1.0,
                        "role_1": 3.0,
                        "role_2": -2.0,
                    },
                }
            )
        bias = estimate_role_bias(rows)
        self.assertEqual(bias, {"role_0": 1.0, "role_1": 3.0, "role_2": -2.0})
        self.assertEqual(
            apply_role_bias(rows[0]["neutral_adjusted_gains"], bias),
            {role: 0.0 for role in ROLES},
        )

    def test_unbalanced_calibration_is_rejected(self):
        rows = [
            {
                "correct_role": "role_0",
                "neutral_adjusted_gains": {role: 0.0 for role in ROLES},
            }
        ]
        with self.assertRaisesRegex(ValueError, "not_balanced"):
            estimate_role_bias(rows)

    def test_one_symmetric_rate_weight_repairs_calibration_margin(self):
        rows = []
        for index, correct_role in enumerate(ROLES):
            rates = {role: 0.2 for role in ROLES}
            rates[correct_role] = 0.7
            rows.append(
                {
                    "correct_role": correct_role,
                    "observed_rates": rates,
                    "neutral_adjusted_gains": {
                        "role_0": 0.0,
                        "role_1": 0.4,
                        "role_2": -0.2,
                    },
                }
            )
        bias = estimate_role_bias(rows)
        weight = fit_rate_weight(rows, bias, required_margin=0.01)
        for row in rows:
            scores = apply_calibrated_decoder(
                row["neutral_adjusted_gains"], bias, row["observed_rates"], weight
            )
            self.assertEqual(max(ROLES, key=scores.get), row["correct_role"])
            alternatives = [scores[r] for r in ROLES if r != row["correct_role"]]
            self.assertGreaterEqual(
                scores[row["correct_role"]] - max(alternatives), 0.01 - 1e-12
            )

    def test_calibration_hash_detects_changes(self):
        frozen = {"role_bias": {role: 0.0 for role in ROLES}}
        changed = {"role_bias": {role: (1.0 if role == "role_2" else 0.0) for role in ROLES}}
        self.assertNotEqual(payload_hash(frozen), payload_hash(changed))


if __name__ == "__main__":
    unittest.main()
