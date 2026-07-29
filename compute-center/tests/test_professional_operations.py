#!/usr/bin/env python3
"""Acceptance tests for professional GIS, Bayesian, and econometric operations."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_dispatch  # noqa: E402
from compute_runner import ComputeError  # noqa: E402
from professional_operations import (  # noqa: E402
    bayesian_inference,
    econometric_analysis,
    gis_spatial_analysis,
)


class ProfessionalOperationTests(unittest.TestCase):
    def test_registered_in_production_dispatcher(self) -> None:
        self.assertIn("gis_spatial_analysis", compute_dispatch.OPERATIONS)
        self.assertIn("bayesian_inference", compute_dispatch.OPERATIONS)
        self.assertIn("econometric_analysis", compute_dispatch.OPERATIONS)

    def test_geodesic_distance_and_overlay(self) -> None:
        distances = gis_spatial_analysis(
            {
                "mode": "geodesic_distance_matrix",
                "points": [
                    {"id": "a", "longitude": 0, "latitude": 0},
                    {"id": "b", "longitude": 1, "latitude": 0},
                ],
            }
        )
        self.assertAlmostEqual(
            distances["distance_matrix"][0][1], 111_319.49, places=1
        )

        overlay = gis_spatial_analysis(
            {
                "mode": "geometry_overlay",
                "crs": "EPSG:3857",
                "action": "intersection",
                "left": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
                    ],
                },
                "right": {
                    "type": "Polygon",
                    "coordinates": [
                        [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]
                    ],
                },
            }
        )
        self.assertEqual(overlay["result_area"], 1.0)
        self.assertEqual(overlay["result_geometry"]["type"], "Polygon")

    def test_geographic_nearest_rejected(self) -> None:
        with self.assertRaisesRegex(ComputeError, "projected CRS"):
            gis_spatial_analysis(
                {
                    "mode": "nearest_features",
                    "crs": "EPSG:4326",
                    "source_geometries": [
                        {"type": "Point", "coordinates": [0, 0]}
                    ],
                    "target_geometries": [
                        {"type": "Point", "coordinates": [1, 1]}
                    ],
                }
            )

    def test_beta_binomial_and_bayesian_regression(self) -> None:
        posterior = bayesian_inference(
            {
                "mode": "beta_binomial",
                "prior_alpha": 1,
                "prior_beta": 1,
                "successes": 8,
                "trials": 10,
            }
        )
        self.assertEqual(posterior["posterior"]["alpha"], 9.0)
        self.assertEqual(posterior["posterior"]["beta"], 3.0)
        self.assertAlmostEqual(posterior["posterior"]["mean"], 0.75)

        regression = bayesian_inference(
            {
                "mode": "bayesian_linear_regression",
                "x": [[0], [1], [2], [3]],
                "y": [1, 3, 5, 7],
            }
        )
        coefficients = {
            row["name"]: row["posterior_mean"]
            for row in regression["coefficients"]
        }
        self.assertAlmostEqual(coefficients["intercept"], 1.0, places=5)
        self.assertAlmostEqual(coefficients["x1"], 2.0, places=5)

    def test_ols_and_difference_in_differences(self) -> None:
        ols = econometric_analysis(
            {
                "mode": "ols",
                "x": [[0], [1], [2], [3], [4]],
                "y": [1, 3, 5, 7, 9],
                "covariance_type": "HC1",
            }
        )
        coefficients = {
            row["name"]: row["estimate"] for row in ols["coefficients"]
        }
        self.assertAlmostEqual(coefficients["intercept"], 1.0, places=10)
        self.assertAlmostEqual(coefficients["x1"], 2.0, places=10)
        self.assertAlmostEqual(ols["r_squared"], 1.0, places=10)

        did = econometric_analysis(
            {
                "mode": "difference_in_differences",
                "outcome": [1, 2, 1, 4, 2, 3, 2, 7],
                "treatment": [0, 0, 1, 1, 0, 0, 1, 1],
                "post": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        self.assertAlmostEqual(
            did["difference_in_differences_estimate"]["estimate"],
            3.0,
            places=10,
        )
        self.assertIn("parallel trends", did["identification_warning"])

    def test_iv_2sls(self) -> None:
        instruments = [[float(index)] for index in range(-10, 10)]
        endogenous = []
        outcome = []
        for index, row in enumerate(instruments):
            z = row[0]
            disturbance = -0.5 if index % 2 == 0 else 0.5
            x = 0.8 * z + disturbance
            y = 1.0 + 2.0 * x + 0.2 * disturbance
            endogenous.append([x])
            outcome.append(y)
        result = econometric_analysis(
            {
                "mode": "iv_2sls",
                "y": outcome,
                "endogenous": endogenous,
                "instruments": instruments,
                "covariance_type": "HC1",
            }
        )
        estimate = next(
            row["estimate"]
            for row in result["coefficients"]
            if row["name"] == "endogenous_1"
        )
        self.assertTrue(math.isfinite(estimate))
        self.assertGreater(estimate, 1.5)
        self.assertLess(estimate, 2.5)
        self.assertFalse(result["first_stage"][0]["rule_of_thumb_weak"])


if __name__ == "__main__":
    unittest.main()
