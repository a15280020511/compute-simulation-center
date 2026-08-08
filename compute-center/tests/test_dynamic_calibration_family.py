from __future__ import annotations

import importlib.util
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_calibration_planner import (
    DynamicCalibrationError,
    plan_dynamic_calibration,
    run_dynamic_calibration_ticket,
)
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from operation_validation import validate_operation_inputs


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "calibration dynamic-family test",
    "upstream_refs": [],
}


def calibration_ticket(*, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    x = [0.5 * i for i in range(11)]
    noise = [0.005, -0.003, 0.004, -0.002, 0.003, -0.004, 0.002, -0.001, 0.003, -0.002, 0.001]
    y = [3.0 * math.exp(-0.4 * value) + 2.0 + noise[index] for index, value in enumerate(x)]
    return {
        "task_id": "dynamic-calibration-test",
        "objective": "Objective prose must never select calibration validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "lmfit_exponential_calibration",
            "x": x,
            "y": y,
            "initial": {"amplitude": 2.5, "decay": 0.3, "offset": 1.5},
            "calibration_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicCalibrationFamilyTests(unittest.TestCase):
    def test_router_runtime_and_existing_preflight(self) -> None:
        value = calibration_ticket()
        self.assertEqual(resolve_dynamic_family(value), "calibration")
        metadata = family_runtime_metadata(value)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-lmfit.txt"])
        validate_operation_inputs(value)

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_calibration(calibration_ticket())
        self.assertEqual(plan["stage_order"], ["exponential_calibration"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_rmse_request_requires_residual_statistics(self) -> None:
        plan = plan_dynamic_calibration(calibration_ticket(context={
            "rmse_consistency_requested": True,
            "rmse_consistency_tolerance": 1e-10,
        }))
        self.assertEqual(
            plan["stage_order"],
            ["exponential_calibration", "residual_statistics", "rmse_consistency_audit"],
        )
        self.assertTrue(plan["optimization"]["required_by_node"]["residual_statistics"])
        self.assertTrue(plan["optimization"]["required_by_node"]["rmse_consistency_audit"])

    def test_bias_target_requires_residual_statistics(self) -> None:
        plan = plan_dynamic_calibration(calibration_ticket(context={"maximum_abs_residual_mean": 0.01}))
        self.assertEqual(
            plan["stage_order"],
            ["exponential_calibration", "residual_statistics", "residual_bias_audit"],
        )

    def test_parameter_targets_select_direct_audit(self) -> None:
        plan = plan_dynamic_calibration(calibration_ticket(context={
            "expected_amplitude": 3.0,
            "amplitude_tolerance": 0.05,
            "expected_decay": 0.4,
            "decay_tolerance": 0.05,
            "expected_offset": 2.0,
            "offset_tolerance": 0.05,
        }))
        self.assertEqual(plan["stage_order"], ["exponential_calibration", "parameter_target_audit"])
        self.assertEqual(plan["planning_features"]["parameter_target_count"], 3)

    def test_all_signals_produce_unique_optimum(self) -> None:
        context = {
            "residual_profile_requested": True,
            "rmse_consistency_requested": True,
            "rmse_consistency_tolerance": 1e-10,
            "maximum_abs_residual_mean": 0.01,
            "expected_amplitude": 3.0,
            "amplitude_tolerance": 0.05,
            "expected_decay": 0.4,
            "decay_tolerance": 0.05,
            "expected_offset": 2.0,
            "offset_tolerance": 0.05,
        }
        plan = plan_dynamic_calibration(calibration_ticket(context=context))
        self.assertEqual(
            plan["stage_order"],
            ["exponential_calibration", "residual_statistics", "rmse_consistency_audit", "residual_bias_audit", "parameter_target_audit"],
        )
        self.assertEqual(plan["optimization"]["objective_value"], 745)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_dangling_tolerance_fails_closed(self) -> None:
        with self.assertRaises(DynamicCalibrationError):
            plan_dynamic_calibration(calibration_ticket(context={"decay_tolerance": 0.01}))
        with self.assertRaises(DynamicCalibrationError):
            plan_dynamic_calibration(calibration_ticket(context={"rmse_consistency_tolerance": 1e-6}))

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicCalibrationError):
            plan_dynamic_calibration(calibration_ticket(context={"run_every_tool": True}))

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = calibration_ticket()
        value["objective"] = "Run residual statistics, RMSE, bias, parameter audits and every calibration tool."
        plan = plan_dynamic_calibration(value)
        self.assertEqual(plan["stage_order"], ["exponential_calibration"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(importlib.util.find_spec("lmfit") is not None, "lmfit is a managed optional dependency; real execution is enforced by calibration-family CI")
    def test_real_cross_tool_pipeline(self) -> None:
        context = {
            "residual_profile_requested": True,
            "rmse_consistency_requested": True,
            "rmse_consistency_tolerance": 1e-10,
            "maximum_abs_residual_mean": 0.01,
            "expected_amplitude": 3.0,
            "amplitude_tolerance": 0.05,
            "expected_decay": 0.4,
            "decay_tolerance": 0.05,
            "expected_offset": 2.0,
            "offset_tolerance": 0.05,
        }
        value = calibration_ticket(context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-calibration-"))
        try:
            result = run_dynamic_calibration_ticket(
                value,
                root,
                {"finance_decision_analysis": finance_decision_analysis, "descriptive_statistics": descriptive_statistics},
            )
            expected = ["exponential_calibration", "residual_statistics", "rmse_consistency_audit", "residual_bias_audit", "parameter_target_audit"]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            primary = result["results"]["final_result"]
            params = primary["parameters"]
            self.assertLess(abs(params["amplitude"]["value"] - 3.0), 0.05)
            self.assertLess(abs(params["decay"]["value"] - 0.4), 0.05)
            self.assertLess(abs(params["offset"]["value"] - 2.0), 0.05)
            self.assertGreater(primary["rmse"], 0.0)
            validation = result["results"]["validation_results"]
            stats = validation["residual_statistics"]
            reconstructed_rmse = math.sqrt(stats["mean"] ** 2 + stats["standard_deviation_population"] ** 2)
            self.assertAlmostEqual(reconstructed_rmse, primary["rmse"], places=10)
            self.assertEqual(validation["rmse_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["residual_bias_audit"]["status"], "PASS")
            self.assertEqual(validation["parameter_target_audit"]["status"], "PASS")
            self.assertEqual(validation["parameter_target_audit"]["candidate_count"], 3)
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 745)
            self.assertTrue(result["results"]["optimization"]["global_optimal_proven"])
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["passed"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("lmfit") is not None, "lmfit is a managed optional dependency")
    def test_parameter_target_failure_is_informative(self) -> None:
        value = calibration_ticket(context={"expected_decay": 1.0, "decay_tolerance": 0.01})
        root = Path(tempfile.mkdtemp(prefix="dynamic-calibration-target-fail-"))
        try:
            result = run_dynamic_calibration_ticket(
                value,
                root,
                {"finance_decision_analysis": finance_decision_analysis, "descriptive_statistics": descriptive_statistics},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["parameter_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
