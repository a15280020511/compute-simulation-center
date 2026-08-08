from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from dynamic_policy_simulation_planner import (
    DynamicPolicySimulationError,
    plan_dynamic_policy_simulation,
    run_dynamic_policy_simulation_ticket,
)
from operation_validation import validate_operation_inputs


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "policy-simulation dynamic-family test",
    "upstream_refs": [],
}


def policy_ticket(*, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    return {
        "task_id": "dynamic-policy-simulation-test",
        "objective": "Objective prose must never select policy validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "policy_microsimulation",
            "incomes": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "tax_brackets": [
                {"threshold": 50.0, "rate": 0.1},
                {"threshold": 100.0, "rate": 0.2},
            ],
            "universal_transfer": 5.0,
            "poverty_line": 40.0,
            "policy_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicPolicySimulationFamilyTests(unittest.TestCase):
    def test_router_runtime_and_existing_preflight(self) -> None:
        value = policy_ticket()
        self.assertEqual(resolve_dynamic_family(value), "policy-simulation")
        metadata = family_runtime_metadata(value)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt"])
        validate_operation_inputs(value)

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_policy_simulation(policy_ticket())
        self.assertEqual(plan["stage_order"], ["policy_microsimulation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_distribution_request_selects_statistics(self) -> None:
        plan = plan_dynamic_policy_simulation(policy_ticket(context={"distribution_profile_requested": True}))
        self.assertEqual(plan["stage_order"], ["policy_microsimulation", "disposable_distribution_statistics"])

    def test_mean_consistency_requires_statistics(self) -> None:
        plan = plan_dynamic_policy_simulation(policy_ticket(context={"mean_consistency_requested": True}))
        self.assertEqual(
            plan["stage_order"],
            ["policy_microsimulation", "disposable_distribution_statistics", "mean_consistency_audit"],
        )
        self.assertTrue(plan["optimization"]["required_by_node"]["disposable_distribution_statistics"])
        self.assertTrue(plan["optimization"]["required_by_node"]["mean_consistency_audit"])

    def test_directional_targets_select_target_audit(self) -> None:
        plan = plan_dynamic_policy_simulation(policy_ticket(context={
            "minimum_net_fiscal_balance": 20.0,
            "maximum_gini_after": 0.27,
            "maximum_poverty_rate_after": 0.30,
        }))
        self.assertEqual(plan["stage_order"], ["policy_microsimulation", "policy_target_audit"])
        self.assertEqual(plan["planning_features"]["policy_target_count"], 3)

    def test_all_signals_produce_unique_optimum(self) -> None:
        context = {
            "distribution_profile_requested": True,
            "mean_consistency_requested": True,
            "mean_consistency_tolerance": 1e-12,
            "minimum_net_fiscal_balance": 20.0,
            "net_fiscal_balance_tolerance": 0.0,
            "maximum_gini_after": 0.27,
            "gini_after_tolerance": 0.0,
            "maximum_poverty_rate_after": 0.30,
            "poverty_rate_after_tolerance": 0.0,
        }
        plan = plan_dynamic_policy_simulation(policy_ticket(context=context))
        self.assertEqual(
            plan["stage_order"],
            ["policy_microsimulation", "disposable_distribution_statistics", "mean_consistency_audit", "policy_target_audit"],
        )
        self.assertEqual(plan["optimization"]["objective_value"], 530)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_dangling_tolerance_fails_closed(self) -> None:
        with self.assertRaises(DynamicPolicySimulationError):
            plan_dynamic_policy_simulation(policy_ticket(context={"gini_after_tolerance": 0.01}))

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicPolicySimulationError):
            plan_dynamic_policy_simulation(policy_ticket(context={"run_every_tool": True}))

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = policy_ticket()
        value["objective"] = "Run distribution statistics, consistency and every policy target audit."
        plan = plan_dynamic_policy_simulation(value)
        self.assertEqual(plan["stage_order"], ["policy_microsimulation"])
        self.assertFalse(plan["objective_text_used"])

    def test_real_cross_tool_pipeline(self) -> None:
        context = {
            "distribution_profile_requested": True,
            "mean_consistency_requested": True,
            "mean_consistency_tolerance": 1e-12,
            "minimum_net_fiscal_balance": 20.0,
            "maximum_gini_after": 0.27,
            "maximum_poverty_rate_after": 0.30,
        }
        value = policy_ticket(context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-policy-simulation-"))
        try:
            result = run_dynamic_policy_simulation_ticket(
                value,
                root,
                {
                    "finance_decision_analysis": finance_decision_analysis,
                    "descriptive_statistics": descriptive_statistics,
                },
            )
            expected = ["policy_microsimulation", "disposable_distribution_statistics", "mean_consistency_audit", "policy_target_audit"]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            primary = result["results"]["final_result"]
            self.assertAlmostEqual(primary["tax_revenue"], 70.0, places=12)
            self.assertAlmostEqual(primary["transfer_cost"], 50.0, places=12)
            self.assertAlmostEqual(primary["net_fiscal_balance"], 20.0, places=12)
            self.assertAlmostEqual(primary["mean_disposable_income"], 53.0, places=12)
            self.assertAlmostEqual(primary["gini_before"], 0.3, places=12)
            self.assertAlmostEqual(primary["gini_after"], 0.26226415094339606, places=12)
            self.assertAlmostEqual(primary["poverty_rate_after"], 0.3, places=12)
            validation = result["results"]["validation_results"]
            self.assertAlmostEqual(validation["disposable_distribution_statistics"]["mean"], 53.0, places=12)
            self.assertEqual(validation["mean_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["policy_target_audit"]["status"], "PASS")
            self.assertEqual(validation["policy_target_audit"]["candidate_count"], 3)
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 530)
            self.assertTrue(result["results"]["optimization"]["global_optimal_proven"])
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["passed"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_policy_target_failure_is_informative_not_execution_failure(self) -> None:
        value = policy_ticket(context={"maximum_gini_after": 0.20})
        root = Path(tempfile.mkdtemp(prefix="dynamic-policy-target-fail-"))
        try:
            result = run_dynamic_policy_simulation_ticket(
                value,
                root,
                {
                    "finance_decision_analysis": finance_decision_analysis,
                    "descriptive_statistics": descriptive_statistics,
                },
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["policy_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
