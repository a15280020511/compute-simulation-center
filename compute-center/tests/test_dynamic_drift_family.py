from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_drift_adapters import evidently_to_adwin
from dynamic_drift_planner import DynamicDriftError, plan_dynamic_drift, run_dynamic_drift_ticket
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from operation_validation import _controlled_preview_modes


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "drift dynamic-family test",
    "upstream_refs": [],
}


def drift_ticket(*, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    reference = [[0.01 * (i % 5), float(i % 7)] for i in range(60)]
    current = [[5.0 + 0.01 * (i % 5), float(i % 7)] for i in range(60)]
    return {
        "task_id": "dynamic-drift-test",
        "objective": "Objective prose must never select drift validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "evidently_data_drift",
            "reference": reference,
            "current": current,
            "variable_names": ["shifted", "stable"],
            "drift_share": 0.5,
            "screening_alpha": 0.05,
            "drift_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicDriftFamilyTests(unittest.TestCase):
    def test_router_and_runtime_metadata(self) -> None:
        value = drift_ticket()
        self.assertEqual(resolve_dynamic_family(value), "drift")
        metadata = family_runtime_metadata(value)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"][0], "requirements-ortools.txt")
        self.assertIn("requirements-institutional-evidently.txt", metadata["requirements"])
        self.assertIn("requirements-institutional-river.txt", metadata["requirements"])
        self.assertIn("requirements-thinktank-econometrics.txt", metadata["requirements"])

    def test_preflight_overlay_is_narrow(self) -> None:
        modes = _controlled_preview_modes("finance_decision_analysis")
        self.assertIn("evidently_data_drift", modes)
        self.assertIn("river_adwin_drift", modes)
        self.assertNotIn("causal_pc_discovery", modes)
        self.assertNotIn("skfolio_walk_forward_portfolio", modes)

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_drift(drift_ticket())
        self.assertEqual(plan["stage_order"], ["distribution_drift"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_explicit_cross_checks_select_branches(self) -> None:
        plan = plan_dynamic_drift(drift_ticket(context={
            "adwin_cross_check_requested": True,
            "change_point_cross_check_requested": True,
        }))
        self.assertEqual(plan["stage_order"], ["distribution_drift", "adwin_cross_check", "change_point_cross_check"])
        self.assertEqual(plan["optimization"]["objective_value"], 260)

    def test_expected_share_selects_audit(self) -> None:
        plan = plan_dynamic_drift(drift_ticket(context={
            "expected_drift_share": 0.5,
            "drift_share_tolerance": 0.0,
        }))
        self.assertEqual(plan["stage_order"], ["distribution_drift", "drift_share_audit"])
        self.assertEqual(plan["optimization"]["objective_value"], 135)

    def test_all_structured_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_drift(drift_ticket(context={
            "adwin_cross_check_requested": True,
            "change_point_cross_check_requested": True,
            "expected_drift_share": 0.5,
            "drift_share_tolerance": 0.0,
        }))
        self.assertEqual(plan["stage_order"], ["distribution_drift", "adwin_cross_check", "change_point_cross_check", "drift_share_audit"])
        self.assertEqual(plan["optimization"]["objective_value"], 395)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_partial_share_benchmark_fails_closed(self) -> None:
        with self.assertRaises(DynamicDriftError):
            plan_dynamic_drift(drift_ticket(context={"expected_drift_share": 0.5}))

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicDriftError):
            plan_dynamic_drift(drift_ticket(context={"run_every_tool": True}))

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = drift_ticket()
        value["objective"] = "Run ADWIN, Ruptures and every drift audit."
        plan = plan_dynamic_drift(value)
        self.assertEqual(plan["stage_order"], ["distribution_drift"])
        self.assertFalse(plan["objective_text_used"])

    def test_adapter_uses_maximum_ks_column(self) -> None:
        value = drift_ticket(context={"adwin_delta": 0.01})
        inputs = value["inputs"]
        result = evidently_to_adwin(inputs, {
            "distribution_drift": {
                "columns": [
                    {"column": "shifted", "ks_statistic": 1.0},
                    {"column": "stable", "ks_statistic": 0.0},
                ]
            }
        }, {})
        self.assertEqual(result["mode"], "river_adwin_drift")
        self.assertEqual(result["delta"], 0.01)
        self.assertEqual(len(result["values"]), 120)
        self.assertLess(result["values"][0], 1.0)
        self.assertGreater(result["values"][60], 4.0)

    @unittest.skipUnless(
        importlib.util.find_spec("evidently") is not None
        and importlib.util.find_spec("river") is not None
        and importlib.util.find_spec("ruptures") is not None,
        "Evidently/River/Ruptures are managed optional dependencies; real execution is enforced by drift-family CI",
    )
    def test_real_cross_tool_pipeline(self) -> None:
        context = {
            "adwin_cross_check_requested": True,
            "change_point_cross_check_requested": True,
            "expected_drift_share": 0.5,
            "drift_share_tolerance": 0.0,
            "adwin_delta": 0.002,
            "change_point_cost_model": "l2",
            "change_point_penalty": 5.0,
        }
        value = drift_ticket(context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-drift-"))
        try:
            result = run_dynamic_drift_ticket(value, root, {"finance_decision_analysis": finance_decision_analysis})
            expected = ["distribution_drift", "adwin_cross_check", "change_point_cross_check", "drift_share_audit"]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            primary = result["results"]["final_result"]
            self.assertEqual(primary["drifted_columns_screen"], 1)
            self.assertAlmostEqual(primary["drift_share_screen"], 0.5, places=12)
            validation = result["results"]["validation_results"]
            self.assertEqual(validation["drift_share_audit"]["status"], "PASS")
            self.assertGreaterEqual(validation["adwin_cross_check"]["drift_count"], 1)
            self.assertIn(60, validation["change_point_cross_check"]["change_points"])
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertTrue(result["results"]["optimization"]["global_optimal_proven"])
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["passed"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
