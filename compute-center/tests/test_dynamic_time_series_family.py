from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import compute_dispatch
from dynamic_family_router import DynamicFamilyRoutingError, resolve_dynamic_family
from dynamic_time_series_planner import plan_dynamic_time_series, run_dynamic_time_series_ticket
from tool_registry import managed_runtime_plan, requirement_files_for_ticket


def dynamic_pipeline() -> dict[str, object]:
    return {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "time-series dynamic family test",
        "upstream_refs": [],
    }


def series_ticket(
    *,
    task_id: str,
    count: int = 12,
    horizon: int = 1,
    diagnostics: bool = False,
    assumption: bool = False,
) -> dict[str, object]:
    data = [10.0 + index + (0.25 if index % 3 == 0 else 0.0) for index in range(count)]
    inputs: dict[str, object] = {"data": data, "horizon": horizon}
    if diagnostics:
        inputs["dynamic_context"] = {"time_series_diagnostics": True}
    if assumption:
        inputs.update({"expected_mean": sum(data) / len(data), "mean_tolerance": 10.0})
    return {
        "task_id": task_id,
        "operation": "time_series_forecast",
        "inputs": inputs,
        "pipeline": dynamic_pipeline(),
        "quality_profile": {
            "decision_class": "exploratory",
            "probabilistic_claim": False,
        },
    }


class DynamicTimeSeriesFamilyTests(unittest.TestCase):
    def test_router_preserves_scenario_family(self) -> None:
        ticket = {
            "task_id": "family-router-scenario",
            "operation": "scenario_compare",
            "inputs": {
                "model": {"intercept": 0.0, "coefficients": {"x": 1.0}},
                "scenarios": [
                    {"name": "a", "values": {"x": 1.0}},
                    {"name": "b", "values": {"x": 2.0}},
                ],
            },
            "pipeline": dynamic_pipeline(),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "scenario-decision")

    def test_router_rejects_unadmitted_dynamic_operation(self) -> None:
        ticket = {
            "task_id": "family-router-reject",
            "operation": "descriptive_statistics",
            "inputs": {"data": [1.0, 2.0, 3.0, 4.0, 5.0]},
            "pipeline": dynamic_pipeline(),
        }
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    def test_short_series_selects_statistics_then_forecast(self) -> None:
        ticket = series_ticket(task_id="time-series-short", count=8)
        plan = plan_dynamic_time_series(ticket)
        self.assertEqual(resolve_dynamic_family(ticket), "time-series")
        self.assertEqual(plan["stage_order"], ["series_statistics", "forecast"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])
        self.assertEqual(plan["optimization"]["exhaustive_cross_check"]["optional_node_count"], 3)

    def test_explicit_diagnostics_selects_pattern_stage(self) -> None:
        ticket = series_ticket(task_id="time-series-diagnostics", diagnostics=True)
        plan = plan_dynamic_time_series(ticket)
        self.assertEqual(
            plan["stage_order"],
            ["series_statistics", "pattern_diagnostics", "forecast"],
        )
        self.assertTrue(plan["planning_features"]["diagnostics_requested"])

    def test_supplied_assumption_forces_assumption_validation(self) -> None:
        ticket = series_ticket(task_id="time-series-assumption", assumption=True)
        plan = plan_dynamic_time_series(ticket)
        self.assertEqual(
            plan["stage_order"],
            ["series_statistics", "assumption_checks", "forecast"],
        )
        self.assertTrue(plan["planning_features"]["assumptions_supplied"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["assumption_checks"])

    def test_long_horizon_plus_assumption_selects_full_family(self) -> None:
        ticket = series_ticket(
            task_id="time-series-full",
            horizon=5,
            assumption=True,
        )
        plan = plan_dynamic_time_series(ticket)
        expected = [
            "series_statistics",
            "pattern_diagnostics",
            "assumption_checks",
            "forecast",
        ]
        self.assertEqual(plan["stage_order"], expected)
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["performed"])
        self.assertTrue(cross["passed"])
        self.assertEqual(cross["optional_node_count"], 3)

        runtime = managed_runtime_plan(ticket)
        self.assertEqual(runtime["capability_pack"], "dynamic-orchestration")
        self.assertEqual(runtime["dynamic_family"], "time-series")
        self.assertEqual(runtime["dynamic_entry_contract"], "time_series_forecast")
        self.assertEqual(len(runtime["requirements"]), 1)
        self.assertTrue(runtime["requirements"][0].endswith("requirements-ortools.txt"))
        self.assertEqual(runtime["network_policy"], "deny")

        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_time_series_ticket(ticket, Path(directory), compute_dispatch.OPERATIONS)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["dynamic_family"], "time-series")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(result["results"]["final_stage"], "forecast")
            self.assertEqual(len(result["results"]["stage_receipts"]), 4)
            self.assertTrue(all(row["status"] == "PASS" for row in result["results"]["stage_receipts"]))
            self.assertEqual(len(result["results"]["final_result"]["forecast"]), 5)
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            state = json.loads((Path(directory) / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "PASS")
            self.assertEqual(state["family"], "time-series")
            self.assertTrue(all("input_sha256" in row and "output_sha256" in row for row in state["stages"]))

    def test_objective_text_does_not_trigger_diagnostics(self) -> None:
        ticket = series_ticket(task_id="time-series-objective-text", count=12, horizon=1)
        ticket["objective"] = "Please run every diagnostic and pattern analysis tool available"
        plan = plan_dynamic_time_series(ticket)
        self.assertEqual(plan["stage_order"], ["series_statistics", "forecast"])
        self.assertFalse(plan["planning_features"]["diagnostics_requested"])
        self.assertFalse(plan["objective_text_used"])


if __name__ == "__main__":
    unittest.main()
