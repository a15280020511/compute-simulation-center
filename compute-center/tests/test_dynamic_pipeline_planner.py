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


def ticket(*, task_id: str, scenarios: int = 3, probabilistic: bool = False, uncertainty: int = 0, objective: str = "") -> dict:
    value = {
        "task_id": task_id,
        "objective": objective,
        "operation": "scenario_compare",
        "inputs": copy.deepcopy(BASE_INPUTS),
        "quality_profile": {
            "decision_class": "exploratory",
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
    def test_two_scenario_plain_comparison_selects_only_entry_stage(self) -> None:
        plan = plan_dynamic_pipeline(ticket(task_id="dynamic-test-0001", scenarios=2))
        self.assertEqual(plan["stage_order"], ["scenarios"])

    def test_three_scenario_structural_case_adds_sensitivity(self) -> None:
        plan = plan_dynamic_pipeline(ticket(task_id="dynamic-test-0002", scenarios=3))
        self.assertEqual(plan["stage_order"], ["scenarios", "sensitivity"])

    def test_two_scenario_probabilistic_case_adds_monte_carlo_without_forcing_sensitivity(self) -> None:
        plan = plan_dynamic_pipeline(
            ticket(task_id="dynamic-test-0003", scenarios=2, probabilistic=True)
        )
        self.assertEqual(plan["stage_order"], ["scenarios", "risk_simulation"])

    def test_uncertainty_heavy_case_selects_full_chain(self) -> None:
        value = ticket(
            task_id="dynamic-test-0004",
            scenarios=3,
            probabilistic=False,
            uncertainty=2,
        )
        plan = plan_dynamic_pipeline(value)
        self.assertEqual(
            plan["stage_order"],
            ["scenarios", "sensitivity", "risk_simulation"],
        )

    def test_objective_text_does_not_route_tools(self) -> None:
        first = ticket(
            task_id="dynamic-test-0005",
            scenarios=2,
            objective="Please run Monte Carlo sensitivity optimization forecast causal analysis",
        )
        second = ticket(
            task_id="dynamic-test-0005",
            scenarios=2,
            objective="Completely unrelated wording",
        )
        self.assertEqual(
            plan_dynamic_pipeline(first)["stage_order"],
            plan_dynamic_pipeline(second)["stage_order"],
        )
        self.assertFalse(plan_dynamic_pipeline(first)["objective_text_used"])

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
            self.assertFalse(result["network_used"])
            self.assertEqual(result["model_calls"], 0)
            self.assertFalse(result["automatic_parallel_execution"])
            observed.append(tuple(result["stage_order"]))
        self.assertEqual(len(set(observed)), 4)


if __name__ == "__main__":
    unittest.main()
