from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_assignment_optimization_planner import (
    DynamicAssignmentOptimizationError,
    plan_dynamic_assignment_optimization,
    run_dynamic_assignment_optimization_ticket,
)
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family

PIPELINE = {"pipeline_id": "dynamic-auto-v1", "stage_id": "dynamic", "sequence_reason": "assignment dynamic-family test", "upstream_refs": []}
COSTS = [[9.0, 2.0, 7.0], [6.0, 4.0, 3.0], [5.0, 8.0, 1.0], [7.0, 6.0, 9.0]]


def assignment_ticket(*, context=None, maximize=False):
    return {
        "task_id": "dynamic-assignment-optimization-test",
        "objective": "Objective prose must never select optional validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "assignment_optimization",
            "workers": ["A", "B", "C", "D"],
            "tasks": ["T1", "T2", "T3"],
            "costs": [list(row) for row in COSTS],
            "maximize": maximize,
            "require_all_tasks": True,
            "assignment_optimization_context": {} if context is None else dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }


class DynamicAssignmentOptimizationFamilyTests(unittest.TestCase):
    def test_router_and_runtime_are_narrow(self) -> None:
        ticket = assignment_ticket()
        self.assertEqual(resolve_dynamic_family(ticket), "assignment-optimization")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt"])

    def test_exact_crosscheck_is_always_required(self) -> None:
        plan = plan_dynamic_assignment_optimization(assignment_ticket())
        self.assertEqual(plan["stage_order"], ["assignment_optimization", "scipy_exact_assignment_audit"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertEqual(plan["optimization"]["objective_value"], 215)
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["unique_optimum"])

    def test_objective_target_adds_branch(self) -> None:
        plan = plan_dynamic_assignment_optimization(assignment_ticket(context={"maximum_objective_value": 9.5}))
        self.assertEqual(plan["stage_order"], ["assignment_optimization", "scipy_exact_assignment_audit", "objective_target_audit"])
        self.assertEqual(plan["optimization"]["objective_value"], 350)
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["unique_optimum"])

    def test_maximize_uses_minimum_target(self) -> None:
        plan = plan_dynamic_assignment_optimization(assignment_ticket(maximize=True, context={"minimum_objective_value": 22.0}))
        self.assertEqual(plan["stage_order"], ["assignment_optimization", "scipy_exact_assignment_audit", "objective_target_audit"])

    def test_wrong_target_direction_fails_closed(self) -> None:
        with self.assertRaises(DynamicAssignmentOptimizationError):
            plan_dynamic_assignment_optimization(assignment_ticket(maximize=True, context={"maximum_objective_value": 50.0}))

    def test_require_all_tasks_false_fails_closed(self) -> None:
        ticket = assignment_ticket(); ticket["inputs"]["require_all_tasks"] = False
        with self.assertRaises(DynamicFamilyRoutingError):
            plan_dynamic_assignment_optimization(ticket)

    def test_workers_less_than_tasks_fails_closed(self) -> None:
        ticket = assignment_ticket(); ticket["inputs"]["workers"] = ["A", "B"]; ticket["inputs"]["costs"] = COSTS[:2]
        with self.assertRaises(DynamicFamilyRoutingError):
            plan_dynamic_assignment_optimization(ticket)

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicAssignmentOptimizationError):
            plan_dynamic_assignment_optimization(assignment_ticket(context={"run_every_solver": True}))

    def test_objective_text_does_not_select_target_node(self) -> None:
        ticket = assignment_ticket(); ticket["objective"] = "Run every assignment solver and require objective target validation."
        plan = plan_dynamic_assignment_optimization(ticket)
        self.assertEqual(plan["stage_order"], ["assignment_optimization", "scipy_exact_assignment_audit"])
        self.assertFalse(plan["objective_text_used"])

    def test_tampered_primary_result_fails_exact_crosscheck(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-assignment-tamper-"))
        try:
            def operation(inputs):
                if inputs.get("mode") == "assignment_optimization":
                    return {
                        "mode": "assignment_optimization",
                        "objective_value": 99.0,
                        "maximize": False,
                        "assignments": [
                            {"worker": "B", "task": "T1", "value": 6.0},
                            {"worker": "A", "task": "T2", "value": 2.0},
                            {"worker": "C", "task": "T3", "value": 1.0},
                        ],
                        "unassigned_workers": ["D"],
                        "unassigned_tasks": [],
                        "decision_support_only": True,
                    }
                return finance_decision_analysis(inputs)
            with self.assertRaises(DynamicAssignmentOptimizationError):
                run_dynamic_assignment_optimization_ticket(assignment_ticket(), root, {"finance_decision_analysis": operation})
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_real_ortools_vs_scipy_pipeline(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-assignment-"))
        try:
            result = run_dynamic_assignment_optimization_ticket(assignment_ticket(context={"exact_consistency_tolerance": 1e-9, "maximum_objective_value": 9.5}), root, {"finance_decision_analysis": finance_decision_analysis})
            primary = result["results"]["final_result"]; validation = result["results"]["validation_results"]
            self.assertEqual(result["status"], "success")
            self.assertAlmostEqual(primary["objective_value"], 9.0, places=9)
            self.assertEqual(len(primary["assignments"]), 3)
            self.assertEqual(validation["scipy_exact_assignment_audit"]["status"], "PASS")
            self.assertEqual(validation["scipy_exact_assignment_audit"]["candidate_count"], 2)
            self.assertEqual(validation["objective_target_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 350)
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_target_failure_is_informative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-assignment-target-fail-"))
        try:
            result = run_dynamic_assignment_optimization_ticket(assignment_ticket(context={"maximum_objective_value": 8.0}), root, {"finance_decision_analysis": finance_decision_analysis})
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["scipy_exact_assignment_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["objective_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
