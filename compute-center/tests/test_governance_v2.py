#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from calibration_engine import calibrate
from constraint_engine import enforce_constraints, evaluate_constraints
from model_comparison import compare_models
from model_ensemble import ensemble_predictions
from model_governance import lifecycle_status, load_model_registry, registered_model, validate_registry_operation_coverage, validate_ticket_governance
from residual_diagnostics import diagnose_residuals


class GovernanceV2Tests(unittest.TestCase):
    def test_schemas_are_valid(self) -> None:
        for name in (
            "assumption-register.schema.json",
            "assumption-library.schema.json",
            "mechanism-register.schema.json",
            "experiment-profile.schema.json",
            "credibility-profile.schema.json",
            "calibration-profile.schema.json",
            "constraint-profile.schema.json",
            "validation-profile.schema.json",
        ):
            schema = json.loads((HERE / name).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)

    def test_model_registry_covers_current_operations(self) -> None:
        registry = load_model_registry()
        operations = {row["operation"] for row in registry["models"]}
        report = validate_registry_operation_coverage(operations)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["covered_operation_count"], 29)
        self.assertTrue(registered_model("monte_carlo")["calibration_supported"])
        for name in ("crisis_early_warning", "information_diffusion_analysis", "causal_policy_evaluation", "bayesian_network_inference", "strategic_policy_analysis", "transport_forecast_analysis"):
            self.assertEqual(registered_model(name)["maturity"], "controlled-preview")

    def test_governance_profile_validation(self) -> None:
        ticket = {
            "operation": "monte_carlo",
            "inputs": {},
            "assumption_register": [{
                "assumption_id": "demand-growth-001",
                "type": "parameter",
                "statement": "Demand growth is uncertain.",
                "linked_parameter": "growth",
                "source_type": "historical",
                "basis": "Frozen historical series.",
                "confidence": "low",
                "minimum": 0.0,
                "maximum": 0.2,
                "distribution": "uniform",
                "calibration_status": "uncalibrated"
            }],
            "calibration_profile": {
                "backend": "least_squares",
                "objective": "rmse",
                "observations": [1.0],
                "result_paths": ["mean"],
                "parameters": [{"name": "growth", "input_path": "model.intercept", "initial": 0.1, "minimum": 0, "maximum": 1}]
            },
            "constraint_profile": {
                "hard_constraints": [{"id": "growth-bounds", "type": "bounds", "field": "growth", "minimum": 0, "maximum": 1}],
                "independent_post_check": True
            },
            "validation_profile": {
                "strategy": "holdout",
                "baseline_required": True,
                "baseline_model_id": "baseline",
                "baseline_predictions": [1.0],
                "structure_comparison_required": True,
                "metrics": ["rmse", "mae"]
            }
        }
        report = validate_ticket_governance(ticket)
        self.assertEqual(report["model_id"], "monte_carlo-registered-v1")
        for name in ("assumption_register", "calibration_profile", "constraint_profile", "validation_profile"):
            self.assertTrue(report["documents"][name])
        for name in ("mechanism_register", "experiment_profile", "credibility_profile"):
            self.assertFalse(report["documents"][name])

    def test_lifecycle_recalibration_trigger(self) -> None:
        self.assertEqual(lifecycle_status(registered_model("time_series_forecast"), triggered_events={"data_drift"})["status"], "MODEL_RECALIBRATION_REQUIRED")

    def test_least_squares_recovers_parameters(self) -> None:
        x = np.linspace(0, 5, 50)
        actual = 2.5 * x + 1.2
        def model(parameters):
            return parameters["slope"] * x + parameters["intercept"]
        profile = {
            "backend": "least_squares",
            "objective": "rmse",
            "loss": "linear",
            "parameters": [
                {"name": "slope", "initial": 1.0, "minimum": 0, "maximum": 10},
                {"name": "intercept", "initial": 0.0, "minimum": -5, "maximum": 5}
            ]
        }
        result = calibrate(model, actual, profile)
        self.assertAlmostEqual(result["parameters"]["slope"], 2.5, places=6)
        self.assertAlmostEqual(result["parameters"]["intercept"], 1.2, places=6)
        self.assertLess(result["metrics"]["rmse"], 1e-8)

    def test_hard_constraint_violation(self) -> None:
        profile = {"hard_constraints": [{"id": "probability", "type": "probability", "field": "p"}, {"id": "stock", "type": "nonnegative", "field": "stock"}, {"id": "mass", "type": "sum_equals", "fields": ["a", "b"], "target": 1.0}], "independent_post_check": True}
        report = evaluate_constraints({"p": 1.2, "stock": 3, "a": 0.4, "b": 0.6}, profile)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["violation_count"], 1)
        with self.assertRaises(ValueError):
            enforce_constraints({"p": 1.2, "stock": 3, "a": 0.4, "b": 0.6}, profile)

    def test_model_comparison_rejects_complex_underperformance(self) -> None:
        report = compare_models([1, 2, 3, 4], {"baseline": [1, 2, 3, 4.1], "complex": [1, 2, 3, 5]}, baseline_model_id="baseline", complexity={"baseline": 1, "complex": 10}, minimum_improvement_over_baseline=0.01)
        self.assertEqual(report["selected_model_id"], "baseline")
        self.assertIn("complex", report["complex_models_not_better_than_baseline"])

    def test_ensemble_weight_cap(self) -> None:
        report = ensemble_predictions({"a": [1, 2], "b": [2, 3], "c": [3, 4]}, method="inverse_error_capped", validation_errors={"a": 0.1, "b": 1.0, "c": 2.0}, maximum_weight=0.6)
        self.assertLessEqual(max(report["weights"].values()), 0.600000001)
        self.assertEqual(len(report["prediction"]), 2)

    def test_residual_diagnostics(self) -> None:
        report = diagnose_residuals([1, 2, 3, 4, 5], [1.1, 1.9, 3.1, 3.9, 5.1])
        self.assertIn(report["status"], {"PASS", "WARN"})
        self.assertAlmostEqual(report["diagnostics"]["mae"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
