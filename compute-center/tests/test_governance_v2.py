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
        self.assertEqual(report["covered_operation_count"], 31)
        self.assertTrue(registered_model("monte_carlo")["calibration_supported"])
        self.assertFalse(registered_model("symbolic_mathematics")["calibration_supported"])
        self.assertFalse(registered_model("large_scale_data_intelligence")["calibration_supported"])
        for name in ("crisis_early_warning", "information_diffusion_analysis", "causal_policy_evaluation", "bayesian_network_inference", "strategic_policy_analysis", "transport_forecast_analysis", "symbolic_mathematics", "large_scale_data_intelligence"):
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

    def test_least_squares_recovers_parameters(self) -> None:
        profile = {
            "backend": "least_squares",
            "objective": "rmse",
            "observations": [3.0, 5.0, 7.0],
            "result_paths": ["prediction[0]", "prediction[1]", "prediction[2]"],
            "parameters": [{"name": "slope", "initial": 0.5, "minimum": 0.0, "maximum": 5.0}],
        }
        result = calibrate(
            profile,
            lambda parameters: {"prediction": [parameters["slope"] * x + 1.0 for x in [1.0, 2.0, 3.0]]},
        )
        self.assertAlmostEqual(result["parameters"]["slope"], 2.0, places=4)
        self.assertLess(result["objective_value"], 1e-8)

    def test_hard_constraint_violation(self) -> None:
        profile = {
            "hard_constraints": [{"id": "positive", "type": "bounds", "field": "value", "minimum": 0.0}],
            "soft_constraints": [],
            "independent_post_check": True,
        }
        report = enforce_constraints(profile, {"value": -1.0})
        self.assertEqual(report["status"], "BLOCK")
        self.assertFalse(evaluate_constraints(profile, {"value": -1.0})["passed"])

    def test_residual_diagnostics(self) -> None:
        report = diagnose_residuals([1, 2, 3, 4], [1.1, 1.9, 3.1, 3.9])
        self.assertEqual(report["status"], "PASS")
        self.assertIn("mean_error", report["metrics"])

    def test_model_comparison_rejects_complex_underperformance(self) -> None:
        report = compare_models(
            [
                {"model_id": "baseline", "predictions": [1, 2, 3], "parameter_count": 1},
                {"model_id": "complex", "predictions": [0, 0, 0], "parameter_count": 10},
            ],
            [1, 2, 3],
            metric="rmse",
        )
        self.assertEqual(report["selected_model_id"], "baseline")

    def test_ensemble_weight_cap(self) -> None:
        report = ensemble_predictions(
            [
                {"model_id": "a", "predictions": [1.0, 2.0], "weight": 0.8},
                {"model_id": "b", "predictions": [1.1, 2.1], "weight": 0.2},
            ],
            max_weight=0.7,
        )
        self.assertLessEqual(max(report["normalized_weights"].values()), 0.7 + 1e-12)

    def test_lifecycle_recalibration_trigger(self) -> None:
        report = lifecycle_status(
            {
                "last_calibrated_at": "2025-01-01T00:00:00Z",
                "recalibration_interval_days": 30,
                "drift_status": "stable",
            },
            now="2026-01-01T00:00:00Z",
        )
        self.assertEqual(report["status"], "RECALIBRATION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
