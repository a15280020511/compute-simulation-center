from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_state_estimation_planner import (
    DynamicStateEstimationError,
    plan_dynamic_state_estimation,
    run_dynamic_state_estimation_ticket,
)


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "state-estimation dynamic-family test",
    "upstream_refs": [],
}


def ticket(*, observations=None, decision_class="exploratory", context=None, benchmark=False):
    if observations is None:
        observations = [[1.0], [1.4], [1.9], [2.3], [2.8], [3.2]]
    inputs = {
        "mode": "bounded_linear_kalman_filter",
        "transition_matrix": [[1.0]],
        "observation_matrix": [[1.0]],
        "process_covariance": [[0.05]],
        "observation_covariance": [[0.2]],
        "initial_covariance": [[1.0]],
        "initial_state": [0.0],
        "observations": observations,
    }
    if context is not None:
        inputs["dynamic_context"] = context
    if benchmark:
        inputs["benchmark_state"] = [3.0]
        inputs["benchmark_tolerance"] = 0.6
    return {
        "task_id": "dynamic-state-estimation-test",
        "objective": "Objective text must not choose validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": inputs,
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicStateEstimationFamilyTests(unittest.TestCase):
    def test_exploratory_selects_only_mandatory_estimator(self) -> None:
        plan = plan_dynamic_state_estimation(ticket(observations=[[1.0], [1.2], [1.4]]))
        self.assertEqual(plan["stage_order"], ["state_estimation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])

    def test_formal_scalar_series_adds_realized_feedback_by_utility(self) -> None:
        plan = plan_dynamic_state_estimation(ticket(decision_class="formal"))
        self.assertEqual(plan["stage_order"], ["state_estimation", "realized_feedback"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["realized_feedback"])
        self.assertFalse(plan["optimization"]["selected_nodes"]["benchmark_check"])

    def test_explicit_requests_select_branching_validation_dag(self) -> None:
        plan = plan_dynamic_state_estimation(
            ticket(context={"realized_feedback": True, "benchmark_check": True}, benchmark=True)
        )
        self.assertEqual(
            plan["stage_order"],
            ["state_estimation", "realized_feedback", "benchmark_check"],
        )
        self.assertEqual(plan["stage_map"]["state_estimation"]["depends_on"], [])
        self.assertEqual(plan["stage_map"]["realized_feedback"]["depends_on"], ["state_estimation"])
        self.assertEqual(plan["stage_map"]["benchmark_check"]["depends_on"], ["state_estimation"])
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["performed"])
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])
        self.assertEqual(plan["optimization"]["objective_value"], cross["best_objective"])

    def test_requested_feedback_fails_closed_for_vector_observations(self) -> None:
        bad = ticket(
            observations=[[1.0, 0.5], [1.2, 0.6], [1.4, 0.7], [1.6, 0.8]],
            context={"realized_feedback": True},
        )
        bad["inputs"]["observation_matrix"] = [[1.0], [1.0]]
        bad["inputs"]["observation_covariance"] = [[0.2, 0.0], [0.0, 0.2]]
        with self.assertRaises(DynamicStateEstimationError):
            plan_dynamic_state_estimation(bad)

    def test_requested_benchmark_requires_complete_benchmark(self) -> None:
        bad = ticket(context={"benchmark_check": True})
        with self.assertRaises(DynamicStateEstimationError):
            plan_dynamic_state_estimation(bad)

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = ticket(observations=[[1.0], [1.2], [1.4]])
        value["objective"] = "Please run feedback and benchmark validation and every robustness tool."
        plan = plan_dynamic_state_estimation(value)
        self.assertEqual(plan["stage_order"], ["state_estimation"])
        self.assertFalse(plan["objective_text_used"])

    def test_same_structured_ticket_is_deterministic(self) -> None:
        value = ticket(context={"realized_feedback": True, "benchmark_check": True}, benchmark=True)
        left = plan_dynamic_state_estimation(value)
        right = plan_dynamic_state_estimation(value)
        self.assertEqual(left["stage_order"], right["stage_order"])
        self.assertEqual(left["optimization"], right["optimization"])

    def test_real_gateway_executes_three_stages_serially(self) -> None:
        value = ticket(context={"realized_feedback": True, "benchmark_check": True}, benchmark=True)
        root = Path(tempfile.mkdtemp(prefix="dynamic-state-estimation-"))
        try:
            result = run_dynamic_state_estimation_ticket(
                value,
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(
                result["results"]["stage_order"],
                ["state_estimation", "realized_feedback", "benchmark_check"],
            )
            self.assertEqual(result["results"]["final_result"]["mode"], "bounded_linear_kalman_filter")
            self.assertEqual(result["results"]["validation_results"]["realized_feedback"]["mode"], "realized_outcome_feedback")
            self.assertEqual(result["results"]["validation_results"]["benchmark_check"]["mode"], "benchmark_comparison")
            self.assertEqual(len(result["results"]["stage_receipts"]), 3)
            self.assertTrue(all(row["status"] == "PASS" for row in result["results"]["stage_receipts"]))
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
