import unittest

from pisa_xai.pisa import (
    brr_standard_error,
    combine_plausible_value_estimates,
    low_performer_flag,
    math_pv_columns,
    replicate_weight_columns,
)
from pisa_xai.evaluation import calibration_bins, calibration_summary, threshold_sensitivity


class PisaUtilityTests(unittest.TestCase):
    def test_math_pv_columns(self):
        self.assertEqual(math_pv_columns(3), ["PV1MATH", "PV2MATH", "PV3MATH"])

    def test_replicate_weight_columns(self):
        self.assertEqual(replicate_weight_columns(count=2), ["W_FSTURWT1", "W_FSTURWT2"])

    def test_brr_standard_error(self):
        se = brr_standard_error(10.0, [9.0, 11.0, 10.5])
        self.assertAlmostEqual(se, (0.05 * (1.0 + 1.0 + 0.25)) ** 0.5)

    def test_combine_plausible_value_estimates(self):
        pooled = combine_plausible_value_estimates([100.0, 102.0, 98.0], [4.0, 4.5, 3.5])
        self.assertAlmostEqual(pooled.estimate, 100.0)
        self.assertGreater(pooled.standard_error, 0)

    def test_low_performer_flag(self):
        self.assertEqual(low_performer_flag(410.0), 1)
        self.assertEqual(low_performer_flag(430.0), 0)

    def test_threshold_sensitivity(self):
        table = threshold_sensitivity([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(set(table["threshold_rule"]), {"default_0.50", "youden_j", "max_f1"})
        self.assertTrue((table["auc"] == 1.0).all())

    def test_calibration_summary(self):
        summary = calibration_summary([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertIn("calibration_slope", summary)
        self.assertGreater(summary["calibration_slope"], 0)
        self.assertAlmostEqual(summary["mean_predicted_probability"], 0.5)

    def test_calibration_bins(self):
        table = calibration_bins([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)
        self.assertEqual(len(table), 2)
        self.assertEqual(table["n"].tolist(), [2, 2])
        self.assertAlmostEqual(table.attrs["expected_calibration_error"], 0.15)


if __name__ == "__main__":
    unittest.main()
