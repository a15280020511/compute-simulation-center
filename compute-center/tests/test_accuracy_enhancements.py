import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_assurance import (
    conditional_coverage_check,
    coverage_evaluation,
    parameter_identifiability_check,
    predictive_check,
)
from cvxpy_assurance import constraint_residual_audit
from feedback_executor import evaluate
from state_estimation import kalman_filter, scalar_nonlinear_filter
from surrogate_experiment import latin_hypercube_design, surrogate_error_validation


class AccuracyEnhancementTests(unittest.TestCase):
    def test_coverage_and_conditional(self):
        result = coverage_evaluation([1, 2, 3], [0, 1, 2], [2, 3, 4], 0.9)
        self.assertEqual(result["empirical_coverage"], 1.0)
        conditional = conditional_coverage_check(
            [1] * 40,
            [0] * 40,
            [2] * 40,
            ["a"] * 20 + ["b"] * 20,
            0.9,
        )
        self.assertEqual(len(conditional["groups"]), 2)

    def test_identifiability_and_predictive(self):
        self.assertTrue(
            parameter_identifiability_check([[1, 0], [0, 1], [1, 1]])["identifiable"]
        )
        self.assertIn(
            "observed_90_coverage",
            predictive_check([[1, 2], [1.1, 1.9], [0.9, 2.1]], [1, 2]),
        )

    def test_residual_and_feedback(self):
        self.assertTrue(
            constraint_residual_audit([1, 2], [0, 0], [2, 3], 3)[
                "feasible_within_tolerance"
            ]
        )
        result = evaluate(
            [
                {
                    "record_type": "realized_outcome",
                    "prediction": 0.8,
                    "realized": 1,
                    "kind": "probability",
                }
            ]
        )
        self.assertAlmostEqual(result["brier"], 0.04)

    def test_state_and_design(self):
        result = kalman_filter(
            [[1.0], [1.2]], [[1]], [[1]], [[0.1]], [[0.2]], [0], [[1]]
        )
        self.assertEqual(len(result["states"]), 2)
        self.assertEqual(
            len(
                scalar_nonlinear_filter(
                    [1, 1.1],
                    "particle_filter",
                    0.1,
                    0.2,
                    1,
                    1,
                    particles=100,
                )["states"]
            ),
            2,
        )
        self.assertEqual(len(latin_hypercube_design([[0, 1], [0, 2]], 10)["design"]), 10)
        self.assertTrue(surrogate_error_validation([1, 2], [1, 2])["publish_allowed"])

    @unittest.skipUnless(
        os.environ.get("RUN_CVXPY_SMOKE") == "1"
        and importlib.util.find_spec("cvxpy") is not None,
        "CVXPY smoke is isolated to the constraints dependency matrix",
    )
    def test_cvxpy_smoke(self):
        from cvxpy_assurance import convex_resource_allocation

        self.assertLessEqual(
            convex_resource_allocation([1, 2], [0, 0], [10, 10], 5)[
                "maximum_constraint_violation"
            ],
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
