from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from dynamic_game_theory_planner import (
    DynamicGameTheoryError,
    plan_dynamic_game_theory,
    run_dynamic_game_theory_ticket,
)


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "game-theory dynamic-family test",
    "upstream_refs": [],
}


def rps_ticket(*, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    return {
        "task_id": "dynamic-game-theory-test",
        "objective": "Objective prose must never select game-theory validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "open_spiel_policy_evaluation",
            "game_id": "matrix_rps",
            "row_policy": [1.0 / 3.0] * 3,
            "column_policy": [1.0 / 3.0] * 3,
            "game_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": decision_class, "publication_policy": "status_only"},
    }


class DynamicGameTheoryFamilyTests(unittest.TestCase):
    def test_router_and_runtime_metadata(self) -> None:
        value = rps_ticket()
        self.assertEqual(resolve_dynamic_family(value), "game-theory")
        metadata = family_runtime_metadata(value)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(
            metadata["requirements"],
            [
                "requirements-ortools.txt",
                "requirements-strategy-open-spiel.txt",
                "requirements-strategy-pygambit.txt",
            ],
        )

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_game_theory(rps_ticket())
        self.assertEqual(plan["stage_order"], ["policy_evaluation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_explicit_equilibrium_analysis_selects_pygambit(self) -> None:
        plan = plan_dynamic_game_theory(
            rps_ticket(context={"equilibrium_analysis_requested": True})
        )
        self.assertEqual(plan["stage_order"], ["policy_evaluation", "pure_equilibria"])
        self.assertTrue(plan["optimization"]["required_by_node"]["pure_equilibria"])

    def test_expected_equilibrium_count_selects_cross_tool_audit(self) -> None:
        context = {"expected_pure_equilibrium_count": 0}
        plan = plan_dynamic_game_theory(rps_ticket(context=context))
        self.assertEqual(
            plan["stage_order"],
            ["policy_evaluation", "pure_equilibria", "equilibrium_count_audit"],
        )
        selected = plan["optimization"]["selected_nodes"]
        self.assertTrue(selected["pure_equilibria"])
        self.assertTrue(selected["equilibrium_count_audit"])

    def test_expected_policy_utility_selects_direct_audit_branch(self) -> None:
        context = {"expected_policy_utility": [0.0, 0.0], "utility_tolerance": 1e-9}
        plan = plan_dynamic_game_theory(rps_ticket(context=context))
        self.assertEqual(plan["stage_order"], ["policy_evaluation", "expected_utility_audit"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["expected_utility_audit"])

    def test_partial_utility_benchmark_fails_closed(self) -> None:
        with self.assertRaises(DynamicGameTheoryError):
            plan_dynamic_game_theory(rps_ticket(context={"expected_policy_utility": [0.0, 0.0]}))

    def test_non_allowlisted_game_fails_closed(self) -> None:
        value = rps_ticket()
        value["inputs"]["game_id"] = "kuhn_poker"
        with self.assertRaises(Exception):
            plan_dynamic_game_theory(value)

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = rps_ticket()
        value["objective"] = "Run PyGambit, equilibrium audits, and every strategic tool."
        plan = plan_dynamic_game_theory(value)
        self.assertEqual(plan["stage_order"], ["policy_evaluation"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("pyspiel") is not None and importlib.util.find_spec("pygambit") is not None,
        "OpenSpiel/PyGambit are managed optional dependencies; real execution is enforced by game-family CI",
    )
    def test_real_rps_cross_tool_pipeline(self) -> None:
        context = {
            "equilibrium_analysis_requested": True,
            "expected_pure_equilibrium_count": 0,
            "expected_policy_utility": [0.0, 0.0],
            "utility_tolerance": 1e-9,
        }
        value = rps_ticket(context=context)
        root = Path(tempfile.mkdtemp(prefix="dynamic-game-theory-"))
        try:
            result = run_dynamic_game_theory_ticket(
                value,
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            expected = [
                "policy_evaluation",
                "pure_equilibria",
                "equilibrium_count_audit",
                "expected_utility_audit",
            ]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(result["results"]["final_result"]["game_id"], "matrix_rps")
            self.assertEqual(result["results"]["final_result"]["expected_utility"], [0.0, 0.0])
            self.assertEqual(result["results"]["validation_results"]["pure_equilibria"]["pure_equilibria"], [])
            self.assertEqual(result["results"]["validation_results"]["equilibrium_count_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["expected_utility_audit"]["status"], "PASS")
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
