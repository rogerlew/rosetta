"""
Rosetta version 3-alpha (3a)
Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.
"""

import os
import sys
import unittest

import numpy as np

_TESTS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rosetta import Rosetta2, Rosetta3  # noqa: E402

EXPECTED_ROSETTA3 = {
    "theta_r": np.float64(0.08483115264178565),
    "theta_s": np.float64(0.3937930854917945),
    "alpha": np.float64(0.008722688765033827),
    "npar": np.float64(1.395122369483862),
    "ks": np.float64(14.975089304545355),
    "wp": np.float64(0.12985017473983151),
    "fc": np.float64(0.24865853187087367),
}

EXPECTED_ROSETTA2 = {
    "theta_r": np.float64(0.08813819523701401),
    "theta_s": np.float64(0.39822412595693213),
    "alpha": np.float64(0.008485314744924408),
    "npar": np.float64(1.3937248893831298),
    "ks": np.float64(12.244374325715864),
    "wp": np.float64(0.13412603960701866),
    "fc": np.float64(0.25576995240017614),
}

DEFAULT_ABS_TOL = 1e-9
DEFAULT_REL_TOL = 1e-6


class RosettaRegressionTests(unittest.TestCase):
    sand = 45.0
    silt = 35.0
    clay = 20.0
    bd = 1.45

    def assert_matches_expected(self, observed, expected, model_name):
        missing = {k: v for k, v in expected.items() if v is None}
        if missing:
            self.fail(
                f"Missing expected values for {model_name}. Populate EXPECTED_{model_name.upper()} with "
                f"the following results:\n{observed}"
            )

        for key, expected_value in expected.items():
            self.assertIn(key, observed, msg=f"{model_name} result missing '{key}'")
            observed_value = observed[key]
            if isinstance(expected_value, (int, float)):
                self.assertTrue(
                    np.isclose(
                        observed_value,
                        expected_value,
                        rtol=DEFAULT_REL_TOL,
                        atol=DEFAULT_ABS_TOL,
                    ),
                    msg=(
                        f"{model_name} mismatch for '{key}': expected {expected_value}, "
                        f"got {observed_value}"
                    ),
                )
            else:
                self.assertEqual(
                    observed_value,
                    expected_value,
                    msg=f"{model_name} mismatch for '{key}'",
                )

    def test_rosetta3_predict_baseline(self):
        model = Rosetta3()
        observed = model.predict_kwargs(
            sand=self.sand,
            silt=self.silt,
            clay=self.clay,
            bd=self.bd,
            calc_wilting_point=True,
            calc_field_capacity=True,
        )
        self.assert_matches_expected(observed, EXPECTED_ROSETTA3, "rosetta3")

    def test_rosetta2_predict_baseline(self):
        model = Rosetta2()
        observed = model.predict_kwargs(
            sand=self.sand,
            silt=self.silt,
            clay=self.clay,
            calc_wilting_point=True,
            calc_field_capacity=True,
        )
        self.assert_matches_expected(observed, EXPECTED_ROSETTA2, "rosetta2")


if __name__ == "__main__":
    unittest.main()
