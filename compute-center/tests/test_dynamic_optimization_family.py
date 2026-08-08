from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_optimization_planner import (
    DynamicOptimizationError,
    plan_dynamic_optimization,
    run_dynamic_optimization_ticket,
)


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "optimization dynamic-family test",
    "upstream_refs": [],
}


def ticket(*, discrete: bool = False, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    variable_type = "integer" if discrete else "continuous"
    return {
        "task_id": "dynamic-optimization-test",
        "objective": "Objective prose must never select optimization validation tools.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "mixed_integer_optimization",
            "variables": [
                {
                    "name": "x",
                    "type": variable_type,
                    "lower_bound": 0.0,
                    "upper_bound": 4.0,
                    "objective_coefficient": 3.0,
                },
                {
                    "name": "y",
                    "type": "continuous",
                    "lower_bound": 0.0,
                    "upper_bound": 8.0,
                    "objective_coefficient": 2.0,
                },
            ],
            "constraints": [
                {"coefficients": {"x": 2.0, "y": 1.0}, "relation": "<=", "rhs": 8.0},
                {"coefficients": {"x": 1.0, "y": 2.0}, "relation": "<=", "rhs": 8.0},
            ],
            "maximize": True,
            "time_limit_seconds": 20,
            "optimization_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicOptimizationFamilyTests(unittest.TestCase):
    def test_exploratory_continuous_problem_selects_primary_only(self) -> None:
        plan = plan_dynamic_optimization(ticket())
        self.assertEqual(plan["stage_order"], ["primary_optimization"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_discrete_problem_selects_independent_relaxation_bundle(self) -> None:
        plan = plan_dynamic_optimization(ticket(discrete=True))
        self.assertEqual(
            plan["stage_order"],
            ["primary_optimization", "independent_relaxation", "relaxation_bound_audit"],
        )
        selected = plan["optimization"]["selected_nodes"]
        self.assertTrue(selected["independent_relaxation"])
        self.assertTrue(selected["relaxation_bound_audit"])
        self.assertFalse(selected["external_objective_benchmark"])

    def test_formal_continuous_problem_selects_independent_crosscheck(self) -> None:
        plan = plan_dynamic_optimization(ticket(decision_class="formal"))
        self.assertEqual(
            plan["stage_order"],
            ["primary_optimization", "independent_relaxation", "relaxation_bound_audit"],
        )

    def test_external_benchmark_is_explicit_required_branch(self) -> None:
        context = {"external_objective_value": 40.0 / 3.0, "external_objective_tolerance": 1e-6}
        plan = plan_dynamic_optimization(ticket(context=context))
        self.assertEqual(plan["stage_order"], ["primary_optimization", "external_objective_benchmark"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["external_objective_benchmark"])

    def test_explicit_crosscheck_fails_closed_for_nonzero_lower_bound(self) -> None:
        value = ticket(context={"independent_relaxation_crosscheck": True})
        value["inputs"]["variables"][0]["lower_bound"] = 1.0
        with self.assertRaises(DynamicOptimizationError):
            plan_dynamic_optimization(value)

    def test_objective_text_does_not_select_crosschecks(self) -> None:
        value = ticket()
        value["objective"] = "Run every cross-check, HiGHS relaxation, benchmark and robustness stage."
        plan = plan_dynamic_optimization(value)
        self.assertEqual(plan["stage_order"], ["primary_optimization"])
        self.assertFalse(plan["objective_text_used"])

    def test_relaxation_and_bound_audit_are_always_coupled(self) -> None:
        for decision_class in ("exploratory", "formal", "high_stakes"):
            plan = plan_dynamic_optimization(ticket(discrete=True, decision_class=decision_class))
            selected = plan["optimization"]["selected_nodes"]
            self.assertEqual(selected["independent_relaxation"], selected["relaxation_bound_audit"])

    @unittest.skipUnless(
        importlib.util.find_spec("pyomo") is not None and importlib.util.find_spec("highspy") is not None,
        "Pyomo/HiGHS are managed optional dependencies; real execution is enforced by optimization-family CI",
    )
    def test_real_gateways_execute_branch_and_join_serially(self) -> None:
        context = {
            "independent_relaxation_crosscheck": True,
            "crosscheck_tolerance": 1e-7,
            "external_objective_value": 13.0,
            "external_objective_tolerance": 1e-7,
        }
        value = ticket(discrete=True, decision_class="formal", context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-optimization-"))
        operations = {"finance_decision_analysis": finance_decision_analysis}
        try:
            result = run_dynamic_optimization_ticket(value, root, operations)
            expected = [
                "primary_optimization",
                "independent_relaxation",
                "relaxation_bound_audit",
                "external_objective_benchmark",
            ]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(
                result["results"]["stage_dependencies"]["relaxation_bound_audit"],
                ["primary_optimization", "independent_relaxation"],
            )
            self.assertEqual(result["results"]["final_result"]["mode"], "mixed_integer_optimization")
            self.assertEqual(result["results"]["final_result"]["status"], "optimal")
            self.assertAlmostEqual(result["results"]["final_result"]["objective_value"], 13.0, places=7)
            self.assertGreaterEqual(
                result["results"]["validation_results"]["independent_relaxation"]["objective_value"],
                result["results"]["final_result"]["objective_value"],
            )
            self.assertEqual(result["results"]["validation_results"]["relaxation_bound_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["external_objective_benchmark"]["status"], "PASS")
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
