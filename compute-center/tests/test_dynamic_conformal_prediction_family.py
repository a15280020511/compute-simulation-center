from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_conformal_prediction_planner import (
    DynamicConformalPredictionError,
    plan_dynamic_conformal_prediction,
    run_dynamic_conformal_prediction_ticket,
)
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "conformal-prediction dynamic-family test",
    "upstream_refs": [],
}


def _fixture():
    train_x = [[index / 10.0] for index in range(40)]
    noise = [-0.12, 0.08, -0.05, 0.10, -0.02, 0.04]
    train_y = [2.0 * row[0] + 1.0 + noise[index % len(noise)] for index, row in enumerate(train_x)]
    predict_x = [[4.0 + index / 10.0] for index in range(10)]
    heldout_noise = [0.05, -0.06, 0.08, -0.04, 0.03, -0.05, 0.07, -0.02, 0.04, -0.03]
    observed = [2.0 * row[0] + 1.0 + heldout_noise[index] for index, row in enumerate(predict_x)]
    return train_x, train_y, predict_x, observed


ALL_SIGNALS = {
    "interval_validation_requested": True,
    "width_consistency_requested": True,
    "point_feedback_requested": True,
    "width_consistency_tolerance": 1e-12,
    "drift_ratio_threshold": 2.0,
    "minimum_empirical_coverage": 0.8,
    "maximum_average_interval_width": 1.0,
    "maximum_mean_interval_score": 2.0,
    "maximum_point_rmse": 0.2,
    "maximum_absolute_bias": 0.1,
    "target_tolerance": 0.0,
}


def conformal_ticket(*, decision_class: str = "exploratory", context=None):
    train_x, train_y, predict_x, observed = _fixture()
    ctx = {} if context is None else dict(context)
    if context is not None and ctx.pop("__include_observed__", True):
        ctx["validation_observed"] = observed
    return {
        "task_id": "dynamic-conformal-prediction-test",
        "objective": "Objective prose must never select conformal validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "mapie_conformal_interval",
            "train_x": train_x,
            "train_y": train_y,
            "predict_x": predict_x,
            "confidence": 0.9,
            "cv": 5,
            "seed": 7,
            "conformal_context": ctx,
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicConformalPredictionFamilyTests(unittest.TestCase):
    def test_router_and_runtime_are_narrow(self) -> None:
        ticket = conformal_ticket()
        self.assertEqual(resolve_dynamic_family(ticket), "conformal-prediction")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-mapie.txt"])

    def test_exploratory_ticket_without_validation_selects_primary_only(self) -> None:
        plan = plan_dynamic_conformal_prediction(conformal_ticket())
        self.assertEqual(plan["stage_order"], ["mapie_conformal_interval"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_interval_validation_selects_validation_only(self) -> None:
        plan = plan_dynamic_conformal_prediction(
            conformal_ticket(context={"interval_validation_requested": True})
        )
        self.assertEqual(plan["stage_order"], ["mapie_conformal_interval", "prediction_interval_validation"])

    def test_width_consistency_requires_interval_validation(self) -> None:
        plan = plan_dynamic_conformal_prediction(
            conformal_ticket(context={"width_consistency_requested": True})
        )
        self.assertEqual(
            plan["stage_order"],
            ["mapie_conformal_interval", "prediction_interval_validation", "interval_width_consistency_audit"],
        )

    def test_point_targets_require_feedback(self) -> None:
        plan = plan_dynamic_conformal_prediction(
            conformal_ticket(context={"maximum_point_rmse": 0.2})
        )
        self.assertEqual(plan["stage_order"], ["mapie_conformal_interval", "realized_outcome_feedback", "point_target_audit"])

    def test_formal_without_heldout_truth_fails_closed(self) -> None:
        with self.assertRaises(DynamicConformalPredictionError):
            plan_dynamic_conformal_prediction(
                conformal_ticket(decision_class="formal", context={"__include_observed__": False})
            )

    def test_requested_validation_without_heldout_truth_fails_closed(self) -> None:
        with self.assertRaises(DynamicConformalPredictionError):
            plan_dynamic_conformal_prediction(
                conformal_ticket(context={"__include_observed__": False, "interval_validation_requested": True})
            )

    def test_all_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_conformal_prediction(conformal_ticket(context=ALL_SIGNALS))
        self.assertEqual(
            plan["stage_order"],
            [
                "mapie_conformal_interval",
                "prediction_interval_validation",
                "interval_width_consistency_audit",
                "interval_target_audit",
                "realized_outcome_feedback",
                "point_target_audit",
            ],
        )
        self.assertEqual(plan["optimization"]["objective_value"], 785)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicConformalPredictionError):
            plan_dynamic_conformal_prediction(
                conformal_ticket(context={"validation_observed": _fixture()[3], "run_every_uncertainty_tool": True})
            )

    def test_validation_length_mismatch_fails_closed(self) -> None:
        ticket = conformal_ticket(context={"interval_validation_requested": True})
        ticket["inputs"]["conformal_context"]["validation_observed"] = [1.0, 2.0]
        with self.assertRaises((DynamicConformalPredictionError, ValueError)):
            plan_dynamic_conformal_prediction(ticket)

    def test_objective_text_does_not_select_optional_nodes(self) -> None:
        ticket = conformal_ticket()
        ticket["objective"] = "Run coverage validation, feedback, interval audits and every uncertainty tool."
        plan = plan_dynamic_conformal_prediction(ticket)
        self.assertEqual(plan["stage_order"], ["mapie_conformal_interval"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("mapie") is not None,
        "MAPIE is a managed optional dependency; real execution is enforced by conformal CI",
    )
    def test_real_cross_tool_pipeline(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-conformal-"))
        try:
            result = run_dynamic_conformal_prediction_ticket(
                conformal_ticket(context=ALL_SIGNALS),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            expected = [
                "mapie_conformal_interval",
                "prediction_interval_validation",
                "interval_width_consistency_audit",
                "interval_target_audit",
                "realized_outcome_feedback",
                "point_target_audit",
            ]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            primary = result["results"]["final_result"]
            validation = result["results"]["validation_results"]
            interval = validation["prediction_interval_validation"]
            feedback = validation["realized_outcome_feedback"]
            self.assertEqual(len(primary["predictions"]), 10)
            self.assertEqual(len(primary["lower_bounds"]), 10)
            self.assertEqual(len(primary["upper_bounds"]), 10)
            self.assertAlmostEqual(primary["mean_interval_width"], interval["average_interval_width"], places=12)
            self.assertGreaterEqual(interval["empirical_coverage"], 0.8)
            self.assertLessEqual(interval["average_interval_width"], 1.0)
            self.assertLessEqual(interval["mean_interval_score"], 2.0)
            self.assertLessEqual(feedback["rmse"], 0.2)
            self.assertLessEqual(abs(feedback["bias"]), 0.1)
            self.assertEqual(validation["interval_width_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["interval_target_audit"]["status"], "PASS")
            self.assertEqual(validation["point_target_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 785)
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("mapie") is not None, "MAPIE is optional")
    def test_quality_target_failure_is_informative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-conformal-target-fail-"))
        try:
            result = run_dynamic_conformal_prediction_ticket(
                conformal_ticket(context={"maximum_average_interval_width": 1e-9}),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["interval_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
