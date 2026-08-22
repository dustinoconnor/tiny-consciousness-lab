import unittest

from normalized_ordered_formulation import canonicalize_records, raw_records, COUNT_REGIMES


class NormalizedOrderedFormulationTests(unittest.TestCase):
    def test_record_order_and_surface_form_collapse_to_same_canonical_text(self):
        regime = COUNT_REGIMES[0]
        variants = []
        for order in ("forward", "reverse"):
            for surface in ("first_before_second", "second_follows_first"):
                records = raw_records("yellow", regime, order, surface)
                variants.append(canonicalize_records(records)[0])
        self.assertEqual(len(set(variants)), 1)

    def test_role_mapping_is_outcome_blind_and_lexically_stable(self):
        first = raw_records("yellow", COUNT_REGIMES[0], "forward", "first_before_second")
        second = raw_records("blue", COUNT_REGIMES[1], "reverse", "second_follows_first")
        self.assertEqual(canonicalize_records(first)[1], canonicalize_records(second)[1])
        self.assertEqual(canonicalize_records(first)[1], {"blue": "role_0", "yellow": "role_1"})

    def test_invalid_counts_are_rejected(self):
        records = raw_records("yellow", COUNT_REGIMES[0], "forward", "first_before_second")
        records[0]["suppressed"] = records[0]["episodes"] + 1
        with self.assertRaisesRegex(ValueError, "invalid_counts"):
            canonicalize_records(records)


if __name__ == "__main__":
    unittest.main()
