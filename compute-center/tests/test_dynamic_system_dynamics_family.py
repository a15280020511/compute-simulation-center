from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_system_dynamics_planner import (
    DynamicSystemDynamicsError,
    plan_dynamic_system_dynamics,
    run_dynamic_system_dynamics_ticket,
)
from institutional_operations import system_dynamics_simulation


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "system-dynamics dynamic-family test",
    "upstream_refs": [],
}


def feedback_ticket(*, decision_class: str = "exploratory", context=None, steps: int = 20):
    if context is None:
        context = {}
    return {
        "task_id": "dynamic-system-dynamics-test",
        "objective": "Objective prose must never select system-dynamics validation stages.",
        "operation": "system_dynamics_simulation",
        "inputs": {
            "mode": "feedback_delay",
            "steps": steps,
            "dt": 1.0,
            "initial_state": 10.0,
            "exogenous_input": 2.0,
            "decay_rate": 0.0,
            "feedback_gain": 0.0,
            "delay_steps": 2,
            "system_dynamics_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicSystemDynamicsFamilyTests(unittest.TestCase):
    def test_exploratory_short_horizon_selects_primary_only(self) -> None:
        plan = plan_dynamic_system_dynamics(feedback_ticket())
        self.assertEqual(plan["stage_order"], ["primary_simulation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_formal_ticket_selects_trajectory_statistics(self) -> None:
        plan = plan_dynamic_system_dynamics(feedback_ticket(decision_class="formal"))
        self.assertEqual(plan["stage_order"], ["primary_simulation", "trajectory_statistics"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["trajectory_statistics"])

    def test_explicit_trajectory_summary_is_required(self) -> None:
        plan = plan_dynamic_system_dynamics(
            feedback_ticket(context={"trajectory_summary_requested": True})
        )
        self.assertEqual(plan["stage_order"], ["primary_simulation", "trajectory_statistics"])
        self.assertTrue(plan["optimization"]["required_by_node"]["trajectory_statistics"])

    def test_explicit_robustness_selects_simulation_and_audit(self) -> None:
        context = {
            "robustness_parameter": "exogenous_input",
            "perturbation_fraction": 0.1,
            "max_absolute_deviation": 4.1,
        }
        plan = plan_dynamic_system_dynamics(feedback_ticket(context=context))
        self.assertEqual(
            plan["stage_order"],
            ["primary_simulation", "robustness_simulation", "robustness_audit"],
        )
        selected = plan["optimization"]["selected_nodes"]
        self.assertTrue(selected["robustness_simulation"])
        self.assertTrue(selected["robustness_audit"])

    def test_external_benchmark_is_required_branch(self) -> None:
        context = {"external_final_value": 50.0, "external_final_tolerance": 1e-9}
        plan = plan_dynamic_system_dynamics(feedback_ticket(context=context))
        self.assertEqual(plan["stage_order"], ["primary_simulation", "external_final_benchmark"])
        self.assertTrue(plan["optimization"]["required_by_node"]["external_final_benchmark"])

    def test_partial_robustness_configuration_fails_closed(self) -> None:
        with self.assertRaises(DynamicSystemDynamicsError):
            plan_dynamic_system_dynamics(
                feedback_ticket(context={"robustness_parameter": "exogenous_input"})
            )

    def test_stock_flow_robustness_is_not_admitted_in_v1(self) -> None:
        value = {
            "task_id": "dynamic-system-dynamics-stock-flow",
            "objective": "Stock-flow v1 robustness must fail closed.",
            "operation": "system_dynamics_simulation",
            "inputs": {
                "mode": "stock_flow",
                "steps": 10,
                "dt": 1.0,
                "stocks": [
                    {"name": "inventory", "initial": 10.0, "inflow": 1.0, "outflow_rate": 0.1, "capacity": 100.0}
                ],
                "system_dynamics_context": {
                    "robustness_parameter": "inflow",
                    "perturbation_fraction": 0.1,
                    "max_absolute_deviation": 10.0,
                },
            },
            "pipeline": dict(PIPELINE),
            "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
        }
        with self.assertRaises(DynamicSystemDynamicsError):
            plan_dynamic_system_dynamics(value)

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = feedback_ticket()
        value["objective"] = "Run every robustness, statistics and external benchmark stage."
        plan = plan_dynamic_system_dynamics(value)
        self.assertEqual(plan["stage_order"], ["primary_simulation"])
        self.assertFalse(plan["objective_text_used"])

    def test_full_real_branching_pipeline(self) -> None:
        context = {
            "trajectory_summary_requested": True,
            "robustness_parameter": "exogenous_input",
            "perturbation_fraction": 0.1,
            "max_absolute_deviation": 4.1,
            "external_final_value": 50.0,
            "external_final_tolerance": 1e-9,
        }
        value = feedback_ticket(context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-system-dynamics-"))
        operations = {
            "system_dynamics_simulation": system_dynamics_simulation,
            "descriptive_statistics": descriptive_statistics,
            "finance_decision_analysis": finance_decision_analysis,
        }
        try:
            result = run_dynamic_system_dynamics_ticket(value, root, operations)
            expected = [
                "primary_simulation",
                "trajectory_statistics",
                "robustness_simulation",
                "robustness_audit",
                "external_final_benchmark",
            ]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertAlmostEqual(result["results"]["final_result"]["final_state"], 50.0, places=10)
            self.assertAlmostEqual(
                result["results"]["validation_results"]["trajectory_statistics"]["mean"],
                30.0,
                places=10,
            )
            self.assertAlmostEqual(
                result["results"]["validation_results"]["robustness_simulation"]["final_state"],
                54.0,
                places=10,
            )
            self.assertEqual(result["results"]["validation_results"]["robustness_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["external_final_benchmark"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertTrue(result["results"]["optimization"]["global_optimal_proven"])
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["passed"])
            self.assertTrue(all(row["status"] == "PASS" for row in result["results"]["stage_receipts"]))
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
