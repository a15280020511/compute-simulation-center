from __future__ import annotations

import importlib.util
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from dynamic_global_sensitivity_planner import (
    DynamicGlobalSensitivityError,
    plan_dynamic_global_sensitivity,
    run_dynamic_global_sensitivity_ticket,
)


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "global-sensitivity dynamic-family test",
    "upstream_refs": [],
}

ALL_SIGNALS = {
    "exact_index_consistency_requested": True,
    "exact_moment_consistency_requested": True,
    "index_consistency_tolerance": 0.03,
    "moment_consistency_tolerance": 0.03,
    "minimum_total_order_by_parameter": {"z": 0.60},
    "target_tolerance": 0.0,
}


def sensitivity_ticket(*, decision_class: str = "exploratory", context=None):
    return {
        "task_id": "dynamic-global-sensitivity-test",
        "objective": "Objective prose must never select sensitivity validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "sobol_sensitivity",
            "parameters": [
                {"name": "x", "minimum": -1.0, "maximum": 1.0},
                {"name": "y", "minimum": -1.0, "maximum": 1.0},
                {"name": "z", "minimum": -1.0, "maximum": 1.0},
            ],
            "model": {
                "intercept": 5.0,
                "linear": {"x": 1.0, "y": 2.0, "z": 3.0},
                "quadratic": {},
                "interactions": [],
            },
            "base_samples": 1024,
            "seed": 11,
            "global_sensitivity_context": {} if context is None else dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicGlobalSensitivityFamilyTests(unittest.TestCase):
    def test_router_and_runtime_are_narrow(self) -> None:
        ticket = sensitivity_ticket()
        self.assertEqual(resolve_dynamic_family(ticket), "global-sensitivity")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-salib.txt"])

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_global_sensitivity(sensitivity_ticket())
        self.assertEqual(plan["stage_order"], ["salib_sobol_sensitivity"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_index_check_selects_only_exact_index_audit(self) -> None:
        plan = plan_dynamic_global_sensitivity(
            sensitivity_ticket(context={"exact_index_consistency_requested": True})
        )
        self.assertEqual(plan["stage_order"], ["salib_sobol_sensitivity", "exact_index_consistency_audit"])

    def test_formal_ticket_requires_both_exact_audits(self) -> None:
        plan = plan_dynamic_global_sensitivity(sensitivity_ticket(decision_class="formal"))
        self.assertEqual(
            plan["stage_order"],
            ["salib_sobol_sensitivity", "exact_index_consistency_audit", "exact_moment_consistency_audit"],
        )

    def test_target_selects_only_target_audit(self) -> None:
        plan = plan_dynamic_global_sensitivity(
            sensitivity_ticket(context={"minimum_total_order_by_parameter": {"z": 0.6}})
        )
        self.assertEqual(plan["stage_order"], ["salib_sobol_sensitivity", "sensitivity_target_audit"])

    def test_all_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_global_sensitivity(sensitivity_ticket(context=ALL_SIGNALS))
        self.assertEqual(
            plan["stage_order"],
            [
                "salib_sobol_sensitivity",
                "exact_index_consistency_audit",
                "exact_moment_consistency_audit",
                "sensitivity_target_audit",
            ],
        )
        self.assertEqual(plan["optimization"]["objective_value"], 475)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_quadratic_model_fails_closed(self) -> None:
        ticket = sensitivity_ticket()
        ticket["inputs"]["model"]["quadratic"] = {"x": 1.0}
        with self.assertRaises(DynamicGlobalSensitivityError):
            plan_dynamic_global_sensitivity(ticket)

    def test_interaction_model_fails_closed(self) -> None:
        ticket = sensitivity_ticket()
        ticket["inputs"]["model"]["interactions"] = [{"left": "x", "right": "y", "coefficient": 1.0}]
        with self.assertRaises(DynamicGlobalSensitivityError):
            plan_dynamic_global_sensitivity(ticket)

    def test_zero_variance_model_fails_closed(self) -> None:
        ticket = sensitivity_ticket()
        ticket["inputs"]["model"]["linear"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        with self.assertRaises(DynamicGlobalSensitivityError):
            plan_dynamic_global_sensitivity(ticket)

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicGlobalSensitivityError):
            plan_dynamic_global_sensitivity(
                sensitivity_ticket(context={"run_every_sensitivity_tool": True})
            )

    def test_objective_text_does_not_select_optional_nodes(self) -> None:
        ticket = sensitivity_ticket()
        ticket["objective"] = "Run exact Sobol checks, uncertainty validation and every sensitivity tool."
        plan = plan_dynamic_global_sensitivity(ticket)
        self.assertEqual(plan["stage_order"], ["salib_sobol_sensitivity"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("SALib") is not None,
        "SALib is a managed optional dependency; real execution is enforced by global-sensitivity CI",
    )
    def test_real_salib_vs_closed_form_pipeline(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-global-sensitivity-"))
        try:
            result = run_dynamic_global_sensitivity_ticket(
                sensitivity_ticket(context=ALL_SIGNALS),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            primary = result["results"]["final_result"]
            validation = result["results"]["validation_results"]
            rows = {row["parameter"]: row for row in primary["ranking"]}
            exact = {"x": 1.0 / 14.0, "y": 4.0 / 14.0, "z": 9.0 / 14.0}
            for name, expected in exact.items():
                self.assertAlmostEqual(rows[name]["first_order"], expected, delta=0.03)
                self.assertAlmostEqual(rows[name]["total_order"], expected, delta=0.03)
            self.assertAlmostEqual(primary["output_distribution"]["mean"], 5.0, delta=0.03)
            self.assertAlmostEqual(primary["output_distribution"]["standard_deviation"], math.sqrt(14.0 / 3.0), delta=0.03)
            self.assertEqual(validation["exact_index_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["exact_moment_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["sensitivity_target_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 475)
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("SALib") is not None, "SALib is optional")
    def test_target_failure_is_informative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-global-sensitivity-target-fail-"))
        try:
            result = run_dynamic_global_sensitivity_ticket(
                sensitivity_ticket(context={"minimum_total_order_by_parameter": {"x": 0.5}}),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["sensitivity_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
