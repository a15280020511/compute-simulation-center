from __future__ import annotations

import copy
import unittest

import compute_runner
from dynamic_pipeline_planner import execute_dynamic_pipeline, plan_dynamic_pipeline


BASE_INPUTS = {
    "model": {
        "intercept": 10.0,
        "coefficients": {"demand": 2.0, "cost": -1.0},
    },
    "scenarios": [
        {"name": "weak", "values": {"demand": 1.0, "cost": 4.0}},
        {"name": "base", "values": {"demand": 2.0, "cost": 3.0}},
        {"name": "strong", "values": {"demand": 4.0, "cost": 1.0}},
    ],
}


def ticket(
    *,
    task_id: str,
    scenarios: int = 3,
    probabilistic: bool = False,
    uncertainty: int = 0,
    objective: str = "",
    decision_class: str = "exploratory",
) -> dict:
    value = {
        "task_id": task_id,
        "objective": objective,
        "operation": "scenario_compare",
        "inputs": copy.deepcopy(BASE_INPUTS),
        "quality_profile": {
            "decision_class": decision_class,
            "probabilistic_claim": probabilistic,
        },
    }
    value["inputs"]["scenarios"] = value["inputs"]["scenarios"][:scenarios]
    if uncertainty:
        value["data_context"] = {
            "variables": [
                {
                    "name": f"u{index}",
                    "required": True,
                    "source_type": "proxy",
                    "confidence": "medium",
                }
                for index in range(uncertainty)
            ]
        }
    return value


class DynamicPlannerTests(unittest.TestCase):
    def assert_optimized(self, plan: dict) -> None:
        self.assertEqual(plan["selection_engine"], "ortools-cp-sat")
        self.assertEqual(plan["graph_engine"], "networkx")
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertFalse(plan["objective_text_used"])
        self.assertFalse(plan["automatic_parallel_execution"])
        self.assertEqual(plan["model_calls"], 0)

    def test_two_scenario_plain_comparison_selects_only_entry_stage(self) -> None:
        plan = plan_dynamic_pipeline(ticket(task_id="dynamic-test-0001", scenarios=2))
        self.assert_optimized(plan)
        self.assertEqual(plan["stage_order"], ["scenarios"])
        self.assertFalse(plan["optimization"]["sensitivity"])
        self.assertFalse(plan["optimization"]["monte_carlo"])

    def test_three_scenario_structural_case_adds_sensitivity(self) -> None:
        plan = plan_dynamic_pipeline(ticket(task_id="dynamic-test-0002", scenarios=3))
        self.assert_optimized(plan)
        self.assertEqual(plan["stage_order"], ["scenarios", "sensitivity"])
        self.assertTrue(plan["optimization"]["sensitivity"])
        self.assertFalse(plan["optimization"]["monte_carlo"])

    def test_two_scenario_probabilistic_case_adds_monte_carlo_without_forcing_sensitivity(self) -> None:
        plan = plan_dynamic_pipeline(
            ticket(task_id="dynamic-test-0003", scenarios=2, probabilistic=True)
        )
        self.assert_optimized(plan)
        self.assertEqual(plan["stage_order"], ["scenarios", "risk_simulation"])
        self.assertFalse(plan["optimization"]["sensitivity"])
        self.assertTrue(plan["optimization"]["monte_carlo"])

    def test_uncertainty_heavy_case_selects_full_chain(self) -> None:
        value = ticket(
            task_id="dynamic-test-0004",
            scenarios=3,
            probabilistic=False,
            uncertainty=2,
        )
        plan = plan_dynamic_pipeline(value)
        self.assert_optimized(plan)
        self.assertEqual(
            plan["stage_order"],
            ["scenarios", "sensitivity", "risk_simulation"],
        )
        self.assertTrue(plan["optimization"]["sensitivity"])
        self.assertTrue(plan["optimization"]["monte_carlo"])

    def test_high_stakes_uncertain_case_requires_sensitivity_and_risk(self) -> None:
        plan = plan_dynamic_pipeline(
            ticket(
                task_id="dynamic-test-0005",
                scenarios=3,
                uncertainty=1,
                decision_class="high_stakes",
            )
        )
        self.assert_optimized(plan)
        self.assertEqual(
            plan["stage_order"],
            ["scenarios", "sensitivity", "risk_simulation"],
        )

    def test_objective_text_does_not_route_tools(self) -> None:
        first = ticket(
            task_id="dynamic-test-0006",
            scenarios=2,
            objective="Please run Monte Carlo sensitivity optimization forecast causal analysis",
        )
        second = ticket(
            task_id="dynamic-test-0006",
            scenarios=2,
            objective="Completely unrelated wording",
        )
        first_plan = plan_dynamic_pipeline(first)
        second_plan = plan_dynamic_pipeline(second)
        self.assertEqual(first_plan["stage_order"], second_plan["stage_order"])
        self.assertEqual(first_plan["optimization"], second_plan["optimization"])
        self.assertFalse(first_plan["objective_text_used"])

    def test_dynamic_plans_execute_successfully(self) -> None:
        cases = [
            ticket(task_id="dynamic-exec-0001", scenarios=2),
            ticket(task_id="dynamic-exec-0002", scenarios=3),
            ticket(task_id="dynamic-exec-0003", scenarios=2, probabilistic=True),
            ticket(task_id="dynamic-exec-0004", scenarios=3, uncertainty=2),
        ]
        observed = []
        for value in cases:
            result = execute_dynamic_pipeline(value, compute_runner.OPERATIONS)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["selection_engine"], "ortools-cp-sat")
            self.assertEqual(result["graph_engine"], "networkx")
            self.assertEqual(result["optimization"]["solver_status"], "OPTIMAL")
            self.assertFalse(result["network_used"])
            self.assertEqual(result["model_calls"], 0)
            self.assertFalse(result["automatic_parallel_execution"])
            observed.append(tuple(result["stage_order"]))
        self.assertEqual(len(set(observed)), 4)


if __name__ == "__main__":
    unittest.main()
