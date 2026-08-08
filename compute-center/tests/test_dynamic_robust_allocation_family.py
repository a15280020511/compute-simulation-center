from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from dynamic_robust_allocation_planner import (
    DynamicRobustAllocationError,
    plan_dynamic_robust_allocation,
    run_dynamic_robust_allocation_ticket,
)


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "robust-allocation dynamic-family test",
    "upstream_refs": [],
}

MATRIX = [
    [0.10, 0.02, 0.05],
    [0.00, 0.09, 0.05],
    [0.08, 0.03, 0.05],
    [0.02, 0.08, 0.05],
]

ALL_SIGNALS = {
    "independent_crosscheck_requested": True,
    "objective_consistency_tolerance": 1e-8,
    "feasibility_audit_requested": True,
    "feasibility_tolerance": 1e-8,
    "minimum_worst_case_return": 0.051,
    "minimum_mean_return": 0.052,
    "maximum_single_asset_weight": 0.58,
    "allocation_target_tolerance": 0.0,
}


def allocation_ticket(*, decision_class: str = "exploratory", context=None):
    return {
        "task_id": "dynamic-robust-allocation-test",
        "objective": "Objective prose must never select optimization cross-check stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "rsome_robust_allocation",
            "scenario_returns": MATRIX,
            "asset_names": ["A", "B", "C"],
            "robust_allocation_context": {} if context is None else dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {
            "decision_class": decision_class,
            "publication_policy": "status_only",
        },
    }


class DynamicRobustAllocationFamilyTests(unittest.TestCase):
    def test_router_and_runtime_are_narrow(self) -> None:
        ticket = allocation_ticket()
        self.assertEqual(resolve_dynamic_family(ticket), "robust-allocation")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-rsome.txt"])

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_robust_allocation(allocation_ticket())
        self.assertEqual(plan["stage_order"], ["rsome_robust_allocation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_crosscheck_selects_objective_audit_dependency(self) -> None:
        plan = plan_dynamic_robust_allocation(
            allocation_ticket(context={"independent_crosscheck_requested": True})
        )
        self.assertEqual(
            plan["stage_order"],
            ["rsome_robust_allocation", "ortools_maximin_crosscheck", "worst_case_objective_consistency_audit"],
        )

    def test_formal_ticket_requires_crosscheck_and_feasibility(self) -> None:
        plan = plan_dynamic_robust_allocation(allocation_ticket(decision_class="formal"))
        self.assertEqual(
            plan["stage_order"],
            [
                "rsome_robust_allocation",
                "ortools_maximin_crosscheck",
                "worst_case_objective_consistency_audit",
                "allocation_feasibility_audit",
            ],
        )

    def test_targets_select_only_target_audit(self) -> None:
        plan = plan_dynamic_robust_allocation(
            allocation_ticket(context={"minimum_worst_case_return": 0.04})
        )
        self.assertEqual(plan["stage_order"], ["rsome_robust_allocation", "allocation_target_audit"])

    def test_all_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_robust_allocation(allocation_ticket(context=ALL_SIGNALS))
        self.assertEqual(plan["stage_order"], [
            "rsome_robust_allocation",
            "ortools_maximin_crosscheck",
            "worst_case_objective_consistency_audit",
            "allocation_feasibility_audit",
            "allocation_target_audit",
        ])
        self.assertEqual(plan["optimization"]["objective_value"], 645)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_dangling_target_tolerance_fails_closed(self) -> None:
        with self.assertRaises(DynamicRobustAllocationError):
            plan_dynamic_robust_allocation(
                allocation_ticket(context={"allocation_target_tolerance": 0.01})
            )

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicRobustAllocationError):
            plan_dynamic_robust_allocation(
                allocation_ticket(context={"run_every_optimizer": True})
            )

    def test_objective_text_does_not_select_optional_nodes(self) -> None:
        ticket = allocation_ticket()
        ticket["objective"] = "Run RSOME, OR-Tools, all cross-checks and every portfolio validator."
        plan = plan_dynamic_robust_allocation(ticket)
        self.assertEqual(plan["stage_order"], ["rsome_robust_allocation"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("rsome") is not None,
        "rsome is a managed optional dependency; real execution is enforced by robust-allocation CI",
    )
    def test_real_cross_solver_pipeline(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-robust-allocation-"))
        try:
            result = run_dynamic_robust_allocation_ticket(
                allocation_ticket(context=ALL_SIGNALS),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            primary = result["results"]["final_result"]
            validation = result["results"]["validation_results"]
            self.assertAlmostEqual(primary["weights"][0], 3.0 / 7.0, places=6)
            self.assertAlmostEqual(primary["weights"][1], 4.0 / 7.0, places=6)
            self.assertAlmostEqual(primary["weights"][2], 0.0, places=6)
            self.assertAlmostEqual(primary["worst_case_return"], 9.0 / 175.0, places=8)
            self.assertAlmostEqual(primary["mean_return"], 37.0 / 700.0, places=8)
            cross = validation["ortools_maximin_crosscheck"]
            self.assertEqual(cross["status"], "optimal")
            self.assertFalse(cross["optimality_not_guaranteed"])
            self.assertAlmostEqual(cross["objective_value"], primary["worst_case_return"], places=8)
            self.assertEqual(validation["worst_case_objective_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["allocation_feasibility_audit"]["status"], "PASS")
            self.assertEqual(validation["allocation_target_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 645)
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("rsome") is not None, "rsome is optional")
    def test_target_failure_is_informative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-robust-target-fail-"))
        try:
            result = run_dynamic_robust_allocation_ticket(
                allocation_ticket(context={"minimum_worst_case_return": 0.06}),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["allocation_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
