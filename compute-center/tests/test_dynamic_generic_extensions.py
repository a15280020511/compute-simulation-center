from __future__ import annotations

import copy
import unittest

import compute_runner
from dynamic_pipeline_planner import DynamicPlanningError, execute_dynamic_pipeline, plan_dynamic_pipeline
from pipeline_adapters import scenario_ranking_to_sensitivity


BASE = {
    "task_id": "dynamic-generic-base",
    "operation": "scenario_compare",
    "inputs": {
        "model": {
            "intercept": 10.0,
            "coefficients": {"demand": 2.0, "cost": -1.0},
        },
        "scenarios": [
            {"name": "weak", "values": {"demand": 1.0, "cost": 4.0}},
            {"name": "base", "values": {"demand": 2.0, "cost": 3.0}},
            {"name": "strong", "values": {"demand": 4.0, "cost": 1.0}},
            {"name": "stretch", "values": {"demand": 5.0, "cost": 2.0}},
        ],
    },
    "quality_profile": {
        "decision_class": "exploratory",
        "probabilistic_claim": False,
    },
}


class DynamicGenericExtensionTests(unittest.TestCase):
    def test_four_scenarios_add_statistics_and_sensitivity(self) -> None:
        value = copy.deepcopy(BASE)
        value["task_id"] = "dynamic-generic-0001"
        plan = plan_dynamic_pipeline(value)
        self.assertEqual(
            plan["stage_order"],
            ["scenarios", "scenario_statistics", "sensitivity"],
        )
        self.assertTrue(plan["optimization"]["scenario_statistics"])
        self.assertTrue(plan["optimization"]["sensitivity"])
        self.assertFalse(plan["optimization"]["monte_carlo"])
        self.assertFalse(plan["optimization"]["constrained_optimization"])
        self.assertEqual(plan["optimization"]["exhaustive_cross_check"]["optional_node_count"], 4)
        self.assertTrue(plan["optimization"]["global_optimal_proven"])

    def test_explicit_continuous_decision_authorization_adds_optimization(self) -> None:
        value = copy.deepcopy(BASE)
        value["task_id"] = "dynamic-generic-0002"
        value["inputs"]["scenarios"] = value["inputs"]["scenarios"][:3]
        value["inputs"]["dynamic_context"] = {
            "continuous_decision_optimization": True,
            "allow_continuous_interpolation": True,
            "controllable_variables": ["demand", "cost"],
        }
        plan = plan_dynamic_pipeline(value)
        self.assertEqual(
            plan["stage_order"],
            ["scenarios", "sensitivity", "decision_optimization"],
        )
        self.assertTrue(plan["planning_features"]["continuous_decision_optimization"])
        self.assertTrue(plan["optimization"]["constrained_optimization"])
        result = execute_dynamic_pipeline(value, compute_runner.OPERATIONS)
        self.assertEqual(result["status"], "PASS")
        self.assertIn("objective_value", result["final_result"])
        self.assertIn("solution", result["final_result"])

    def test_optimization_is_fail_closed_without_complete_authorization(self) -> None:
        missing_interpolation = copy.deepcopy(BASE)
        missing_interpolation["task_id"] = "dynamic-generic-0003"
        missing_interpolation["inputs"]["dynamic_context"] = {
            "continuous_decision_optimization": True,
            "controllable_variables": ["demand", "cost"],
        }
        with self.assertRaises(DynamicPlanningError):
            plan_dynamic_pipeline(missing_interpolation)

        incomplete_variables = copy.deepcopy(BASE)
        incomplete_variables["task_id"] = "dynamic-generic-0004"
        incomplete_variables["inputs"]["dynamic_context"] = {
            "continuous_decision_optimization": True,
            "allow_continuous_interpolation": True,
            "controllable_variables": ["demand"],
        }
        with self.assertRaises(DynamicPlanningError):
            plan_dynamic_pipeline(incomplete_variables)

    def test_objective_text_cannot_authorize_optimization(self) -> None:
        value = copy.deepcopy(BASE)
        value["task_id"] = "dynamic-generic-0005"
        value["inputs"]["scenarios"] = value["inputs"]["scenarios"][:3]
        value["objective"] = "Optimize every variable continuously and use every tool available"
        plan = plan_dynamic_pipeline(value)
        self.assertFalse(plan["planning_features"]["continuous_decision_optimization"])
        self.assertFalse(plan["optimization"]["constrained_optimization"])
        self.assertNotIn("decision_optimization", plan["stage_order"])
        self.assertFalse(plan["objective_text_used"])

    def test_constant_model_variable_no_longer_breaks_sensitivity_adapter(self) -> None:
        initial_inputs = {
            "model": {
                "intercept": 1.0,
                "coefficients": {"varying": 2.0, "constant": 3.0},
            }
        }
        stage_results = {
            "scenarios": {
                "ranking": [
                    {"values": {"varying": 3.0, "constant": 5.0}},
                    {"values": {"varying": 1.0, "constant": 5.0}},
                ]
            }
        }
        adapted = scenario_ranking_to_sensitivity(initial_inputs, stage_results, {})
        self.assertEqual(adapted["model"]["coefficients"], {"varying": 2.0})
        self.assertEqual(adapted["model"]["intercept"], 16.0)
        self.assertEqual([row["name"] for row in adapted["variables"]], ["varying"])

    def test_full_five_stage_plan_executes_serially(self) -> None:
        value = copy.deepcopy(BASE)
        value["task_id"] = "dynamic-generic-0006"
        value["quality_profile"]["probabilistic_claim"] = True
        value["inputs"]["dynamic_context"] = {
            "continuous_decision_optimization": True,
            "allow_continuous_interpolation": True,
            "controllable_variables": ["demand", "cost"],
        }
        result = execute_dynamic_pipeline(value, compute_runner.OPERATIONS)
        self.assertEqual(
            result["stage_order"],
            [
                "scenarios",
                "scenario_statistics",
                "sensitivity",
                "risk_simulation",
                "decision_optimization",
            ],
        )
        self.assertEqual(len(result["receipts"]), 5)
        self.assertTrue(all(row["status"] == "PASS" for row in result["receipts"]))
        self.assertEqual(result["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(result["optimization"]["global_optimal_proven"])
        self.assertTrue(result["optimization"]["exhaustive_cross_check"]["passed"])
        self.assertFalse(result["network_used"])
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["automatic_parallel_execution"])


if __name__ == "__main__":
    unittest.main()
