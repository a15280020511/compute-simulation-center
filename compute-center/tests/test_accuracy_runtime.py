#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accuracy_runtime import derive_evidence_maturity, execute_calibration, execute_experiment, execute_validation
from assumption_runtime import build_assumption_plan


class AccuracyRuntimeTests(unittest.TestCase):
    @staticmethod
    def linear_handler(inputs):
        slope = float(inputs["parameters"]["slope"])
        intercept = float(inputs["parameters"]["intercept"])
        x = [float(item) for item in inputs["x"]]
        return {"prediction": [slope * item + intercept for item in x]}

    def test_executable_calibration_recovers_parameters(self) -> None:
        inputs = {"parameters": {"slope": 1.0, "intercept": 0.0}, "x": [0, 1, 2, 3]}
        profile = {
            "backend": "least_squares",
            "objective": "rmse",
            "observations": [1, 3, 5, 7],
            "result_paths": ["prediction"],
            "multi_start": 3,
            "seed": 7,
            "parameters": [
                {"name": "slope", "input_path": "parameters.slope", "initial": 0.5, "minimum": 0, "maximum": 4},
                {"name": "intercept", "input_path": "parameters.intercept", "initial": 0, "minimum": -2, "maximum": 3},
            ],
        }
        calibrated, report = execute_calibration(self.linear_handler, inputs, profile)
        self.assertEqual(report["execution_status"], "EXECUTED")
        self.assertAlmostEqual(calibrated["parameters"]["slope"], 2.0, places=5)
        self.assertAlmostEqual(calibrated["parameters"]["intercept"], 1.0, places=5)

    @staticmethod
    def stochastic_handler(inputs):
        seed = int(inputs["seed"])
        return {"metric": float(seed % 5)}

    def test_replications_are_actually_executed(self) -> None:
        report = execute_experiment(
            self.stochastic_handler,
            {"seed": 0},
            {
                "design_id": "multi-seed-replication",
                "base_seed": 10,
                "seed_path": "seed",
                "replications": 5,
                "result_paths": ["metric"],
                "stopping_rule": "fixed replications",
            },
        )
        self.assertEqual(report["executed_replications"], 5)
        self.assertEqual(report["seeds"], [10, 11, 12, 13, 14])

    def test_validation_rejects_candidate_below_baseline(self) -> None:
        report = execute_validation(
            {"prediction": [1, 2, 5]},
            {
                "strategy": "holdout",
                "baseline_required": True,
                "baseline_model_id": "baseline",
                "baseline_predictions": [1, 2, 3.1],
                "structure_comparison_required": True,
                "actual_values": [1, 2, 3],
                "result_paths": ["prediction"],
                "minimum_improvement_over_baseline": 0.01,
            },
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("CANDIDATE_DID_NOT_BEAT_BASELINE", report["threshold_failures"])

    def test_missing_data_never_becomes_silent_point_estimate(self) -> None:
        ticket = {
            "task_id": "assumption-test-001",
            "operation": "monte_carlo",
            "inputs": {},
            "quality_profile": {"decision_class": "formal"},
            "data_context": {
                "variables": [{
                    "name": "demand",
                    "required": True,
                    "source_type": "gpts_assumption",
                    "confidence": "low",
                    "missing": True,
                    "expected_range": {"minimum": 80, "maximum": 140},
                    "replacement_strategy": "assumption",
                }]
            },
        }
        plan = build_assumption_plan(ticket)
        self.assertEqual(plan["status"], "BLOCKED")
        self.assertFalse(plan["assumption_candidates"][0]["point_estimate_allowed"])
        self.assertEqual(plan["assumption_candidates"][0]["candidate_distribution"], "uniform")

    def test_maturity_requires_validation_feedback_and_review(self) -> None:
        model = {"maturity": "production", "calibration_supported": True, "benchmark_ids": ["golden-1"]}
        report = derive_evidence_maturity(
            model,
            {"execution_status": "EXECUTED"},
            {"status": "PASS"},
            {"status": "PASS"},
            {"operational_feedback_evidence_sha256": "a" * 64, "technical_review_evidence_sha256": "b" * 64},
        )
        self.assertEqual(report["evidence_maturity"], "decision-grade")


if __name__ == "__main__":
    unittest.main()
