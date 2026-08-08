from __future__ import annotations

import importlib.util
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_control_response_planner import (
    DynamicControlResponseError,
    plan_dynamic_control_response,
    run_dynamic_control_response_ticket,
)
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from operation_validation import validate_operation_inputs


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "control-response dynamic-family test",
    "upstream_refs": [],
}


def control_ticket(*, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    return {
        "task_id": "dynamic-control-response-test",
        "objective": "Objective prose must never select control validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "control_step_response",
            "numerator": [1.0],
            "denominator": [1.0, 1.0],
            "time_end": 10.0,
            "points": 201,
            "control_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicControlResponseFamilyTests(unittest.TestCase):
    def test_router_runtime_and_existing_preflight(self) -> None:
        value = control_ticket()
        self.assertEqual(resolve_dynamic_family(value), "control-response")
        metadata = family_runtime_metadata(value)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-control.txt"])
        validate_operation_inputs(value)

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_control_response(control_ticket())
        self.assertEqual(plan["stage_order"], ["control_step_response"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])
        self.assertAlmostEqual(plan["planning_features"]["independent_dc_gain"], 1.0, places=15)
        self.assertEqual(plan["planning_features"]["independent_poles"], [[-1.0, 0.0]])

    def test_dc_gain_request_selects_independent_audit(self) -> None:
        plan = plan_dynamic_control_response(control_ticket(context={
            "dc_gain_consistency_requested": True,
            "dc_gain_tolerance": 1e-4,
        }))
        self.assertEqual(plan["stage_order"], ["control_step_response", "dc_gain_consistency_audit"])
        self.assertEqual(plan["optimization"]["objective_value"], 165)

    def test_unstable_dc_gain_request_fails_closed(self) -> None:
        value = control_ticket(context={"dc_gain_consistency_requested": True})
        value["inputs"]["denominator"] = [1.0, -1.0]
        with self.assertRaises(DynamicControlResponseError):
            plan_dynamic_control_response(value)

    def test_tail_target_requires_tail_statistics(self) -> None:
        plan = plan_dynamic_control_response(control_ticket(context={
            "maximum_tail_standard_deviation": 0.001,
        }))
        self.assertEqual(plan["stage_order"], ["control_step_response", "tail_response_statistics", "tail_stability_audit"])
        self.assertTrue(plan["optimization"]["required_by_node"]["tail_response_statistics"])
        self.assertTrue(plan["optimization"]["required_by_node"]["tail_stability_audit"])

    def test_all_signals_produce_unique_optimum(self) -> None:
        context = {
            "tail_profile_requested": True,
            "tail_fraction": 0.2,
            "maximum_tail_standard_deviation": 0.001,
            "tail_standard_deviation_tolerance": 0.0,
            "dc_gain_consistency_requested": True,
            "dc_gain_tolerance": 1e-4,
            "maximum_overshoot_percent": 0.1,
            "minimum_final_value": 0.999,
            "maximum_final_value": 1.001,
            "final_value_tolerance": 0.0,
        }
        plan = plan_dynamic_control_response(control_ticket(context=context))
        self.assertEqual(
            plan["stage_order"],
            ["control_step_response", "tail_response_statistics", "tail_stability_audit", "dc_gain_consistency_audit", "control_target_audit"],
        )
        self.assertEqual(plan["optimization"]["objective_value"], 685)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_dangling_tolerance_fails_closed(self) -> None:
        with self.assertRaises(DynamicControlResponseError):
            plan_dynamic_control_response(control_ticket(context={"dc_gain_tolerance": 0.01}))

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicControlResponseError):
            plan_dynamic_control_response(control_ticket(context={"run_every_tool": True}))

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = control_ticket()
        value["objective"] = "Run tail statistics, DC gain audit, target audit and every control tool."
        plan = plan_dynamic_control_response(value)
        self.assertEqual(plan["stage_order"], ["control_step_response"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(importlib.util.find_spec("control") is not None, "python-control is a managed optional dependency; real execution is enforced by control-family CI")
    def test_real_cross_tool_pipeline(self) -> None:
        context = {
            "tail_profile_requested": True,
            "tail_fraction": 0.2,
            "maximum_tail_standard_deviation": 0.001,
            "dc_gain_consistency_requested": True,
            "dc_gain_tolerance": 1e-4,
            "maximum_overshoot_percent": 0.1,
            "minimum_final_value": 0.999,
            "maximum_final_value": 1.001,
        }
        value = control_ticket(context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-control-response-"))
        try:
            result = run_dynamic_control_response_ticket(
                value,
                root,
                {"finance_decision_analysis": finance_decision_analysis, "descriptive_statistics": descriptive_statistics},
            )
            expected = ["control_step_response", "tail_response_statistics", "tail_stability_audit", "dc_gain_consistency_audit", "control_target_audit"]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            primary = result["results"]["final_result"]
            self.assertAlmostEqual(primary["final_value"], 1.0 - math.exp(-10.0), places=10)
            self.assertAlmostEqual(primary["overshoot_percent"], 0.0, places=12)
            validation = result["results"]["validation_results"]
            self.assertLess(validation["tail_response_statistics"]["standard_deviation_population"], 0.001)
            self.assertEqual(validation["tail_stability_audit"]["status"], "PASS")
            self.assertEqual(validation["dc_gain_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["control_target_audit"]["status"], "PASS")
            self.assertEqual(validation["control_target_audit"]["candidate_count"], 3)
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 685)
            self.assertTrue(result["results"]["optimization"]["global_optimal_proven"])
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["passed"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("control") is not None, "python-control is a managed optional dependency")
    def test_target_failure_is_informative_not_execution_failure(self) -> None:
        value = control_ticket(context={"maximum_final_value": 0.9})
        root = Path(tempfile.mkdtemp(prefix="dynamic-control-target-fail-"))
        try:
            result = run_dynamic_control_response_ticket(
                value,
                root,
                {"finance_decision_analysis": finance_decision_analysis, "descriptive_statistics": descriptive_statistics},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["control_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
