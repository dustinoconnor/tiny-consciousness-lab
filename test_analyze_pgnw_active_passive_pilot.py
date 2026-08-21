import unittest

from analyze_pgnw_active_passive_pilot import fisher_two_sided


class ActivePassivePilotAnalysisTests(unittest.TestCase):
    def test_perfect_four_by_four_split_has_registered_p_value(self):
        self.assertAlmostEqual(
            fisher_two_sided(4, 4, 0, 4),
            0.02857142857142857,
        )

    def test_equal_rates_have_unit_p_value(self):
        self.assertEqual(fisher_two_sided(2, 4, 2, 4), 1.0)
