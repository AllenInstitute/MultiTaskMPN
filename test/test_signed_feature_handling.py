"""Signed modulation features (W * Var(M)) are handled symmetrically."""

import unittest

import _bootstrap  # noqa: F401
import numpy as np
import clustering
import lesion_plot
import modulation_variants as mv


class UnresponsiveDetectionTests(unittest.TestCase):
    def test_strongly_negative_rows_are_responsive_and_near_zero_rows_are_not(self):
        data = np.array([[1.0, 2.0, 3.0],        # strong positive
                         [-1.0, -2.0, -3.0],     # strong negative: must stay
                         [1e-5, 0.0, -1e-5],     # near zero: unresponsive
                         [0.0, 0.0, 0.0]])       # silent: unresponsive
        mask = clustering.unresponsive_row_mask(data, 1e-3)
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_non_negative_data_matches_the_former_signed_mean_rule(self):
        rng = np.random.default_rng(0)
        data = rng.random((50, 12)) * (rng.random((50, 1)) < 0.9)   # some silent rows
        data[:5] *= 1e-5
        former = data.mean(axis=1) < 1e-3 * data.mean(axis=1).max()
        np.testing.assert_array_equal(clustering.unresponsive_row_mask(data, 1e-3), former)

    def test_all_zero_input_flags_everything_without_dividing_by_zero(self):
        mask = clustering.unresponsive_row_mask(np.zeros((3, 4)), 1e-3)
        np.testing.assert_array_equal(mask, [True, True, True])


class SignedLog1pTests(unittest.TestCase):
    def test_matches_log1p_for_non_negative_and_is_odd_symmetric(self):
        positive = np.array([0.0, 0.5, 3.0])
        np.testing.assert_allclose(mv.signed_log1p(positive), np.log1p(positive))
        np.testing.assert_allclose(mv.signed_log1p(-positive), -np.log1p(positive))
        self.assertTrue(np.isfinite(mv.signed_log1p(np.array([-5.0, -1.0, -0.999]))).all())


class TuningProfileTests(unittest.TestCase):
    def test_signed_variant_compares_profiles_by_magnitude(self):
        means = np.array([[0.2, -0.2], [0.5, -0.5]])
        out, note = lesion_plot._tuning_profiles_for_variant(
            means, "modulation_all_var_weighted_unnormalized")
        np.testing.assert_allclose(out, np.abs(means))
        self.assertIn("absolute", note)
        # same shape, opposite sign now correlates at +1 rather than -1
        self.assertAlmostEqual(np.corrcoef(out.T)[0, 1], 1.0)

    def test_non_negative_variants_pass_through(self):
        means = np.array([[0.2, 0.1], [0.5, 0.4]])
        for key in ("modulation_all_unnormalized", "modulation_all_normalized",
                    "modulation_all_weighted_unnormalized",
                    "modulation_all_abs_weighted_unnormalized"):
            out, note = lesion_plot._tuning_profiles_for_variant(means, key)
            np.testing.assert_array_equal(out, means)
            self.assertEqual(note, "cluster mean")


if __name__ == "__main__":
    unittest.main()
