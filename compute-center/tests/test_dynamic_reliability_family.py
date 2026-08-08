from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from compute_runner import OPERATIONS as CORE_OPERATIONS
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_reliability_planner import (
    DynamicReliabilityError,
    plan_dynamic_reliability,
    run_dynamic_reliability_ticket,
)


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "reliability dynamic-family test",
    "upstream_refs": [],
}


def sample(repeats: int = 2) -> list[float]:
    return [value for _ in range(repeats) for value in (8.0, 9.0, 10.0, 11.0, 12.0)]


def ticket(*, data=None, decision_class="exploratory", context=None):
    if data is None:
        data = sample(2)
    if context is None:
        context = {"threshold": 8.0, "tail": "lower"}
    return {
        "task_id": "dynamic-reliability-test",
        "objective": "Objective text must not select validation tools.",
        "operation": "descriptive_statistics",
        "inputs": {
            "data": data,
            "reliability_context": context,
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicReliabilityFamilyTests(unittest.TestCase):
    def test_exploratory_small_sample_selects_analytic_only(self) -> None:
        plan = plan_dynamic_reliability(ticket())
        self.assertEqual(plan["stage_order"], ["sample_statistics", "analytic_reliability"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])

    def test_formal_case_selects_coupled_monte_carlo_bundle(self) -> None:
        plan = plan_dynamic_reliability(ticket(decision_class="formal"))
        self.assertEqual(
            plan["stage_order"],
            ["sample_statistics", "analytic_reliability", "monte_carlo_validation", "analytic_mc_agreement"],
        )
        selected = plan["optimization"]["selected_nodes"]
        self.assertTrue(selected["monte_carlo_validation"])
        self.assertTrue(selected["analytic_mc_agreement"])
        self.assertFalse(selected["external_benchmark"])

    def test_large_sample_selects_coupled_validation_by_utility(self) -> None:
        plan = plan_dynamic_reliability(ticket(data=sample(8)))
        self.assertTrue(plan["optimization"]["selected_nodes"]["monte_carlo_validation"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["analytic_mc_agreement"])

    def test_external_benchmark_is_explicit_required_branch(self) -> None:
        context = {
            "threshold": 8.0,
            "tail": "lower",
            "external_failure_probability": 0.08,
            "external_benchmark_tolerance": 0.03,
        }
        plan = plan_dynamic_reliability(ticket(context=context))
        self.assertEqual(plan["stage_order"], ["sample_statistics", "analytic_reliability", "external_benchmark"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["external_benchmark"])

    def test_explicit_monte_carlo_request_fails_closed_for_too_small_sample(self) -> None:
        context = {"threshold": 8.0, "tail": "lower", "monte_carlo_crosscheck": True}
        with self.assertRaises(DynamicReliabilityError):
            plan_dynamic_reliability(ticket(data=[8.0, 9.0, 10.0, 11.0], context=context))

    def test_constant_sample_fails_closed(self) -> None:
        with self.assertRaises(DynamicReliabilityError):
            plan_dynamic_reliability(ticket(data=[10.0] * 10))

    def test_objective_text_does_not_select_crosschecks(self) -> None:
        value = ticket()
        value["objective"] = "Run Monte Carlo and external benchmark and every reliability check."
        plan = plan_dynamic_reliability(value)
        self.assertEqual(plan["stage_order"], ["sample_statistics", "analytic_reliability"])
        self.assertFalse(plan["objective_text_used"])

    def test_monte_carlo_and_agreement_are_always_coupled(self) -> None:
        for decision_class in ("exploratory", "formal", "high_stakes"):
            plan = plan_dynamic_reliability(ticket(data=sample(8), decision_class=decision_class))
            selected = plan["optimization"]["selected_nodes"]
            self.assertEqual(selected["monte_carlo_validation"], selected["analytic_mc_agreement"])

    @unittest.skipUnless(
        importlib.util.find_spec("openturns") is not None,
        "OpenTURNS is an optional managed capability dependency; real execution is enforced by the dedicated reliability CI",
    )
    def test_real_gateways_execute_branch_and_join_serially(self) -> None:
        context = {
            "threshold": 8.0,
            "tail": "lower",
            "monte_carlo_crosscheck": True,
            "monte_carlo_iterations": 50000,
            "monte_carlo_seed": 7,
            "mc_agreement_tolerance": 0.02,
            "external_failure_probability": 0.079,
            "external_benchmark_tolerance": 0.02,
        }
        value = ticket(data=sample(8), context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-reliability-"))
        operations = {
            "descriptive_statistics": CORE_OPERATIONS["descriptive_statistics"],
            "monte_carlo": CORE_OPERATIONS["monte_carlo"],
            "finance_decision_analysis": finance_decision_analysis,
        }
        try:
            result = run_dynamic_reliability_ticket(value, root, operations)
            expected = [
                "sample_statistics",
                "analytic_reliability",
                "monte_carlo_validation",
                "analytic_mc_agreement",
                "external_benchmark",
            ]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(result["results"]["stage_dependencies"]["analytic_mc_agreement"], ["analytic_reliability", "monte_carlo_validation"])
            self.assertEqual(result["results"]["final_result"]["mode"], "openturns_reliability_probability")
            self.assertEqual(result["results"]["validation_results"]["analytic_mc_agreement"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["external_benchmark"]["status"], "PASS")
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
