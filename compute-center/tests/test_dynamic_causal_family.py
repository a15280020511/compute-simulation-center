from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import compute_dispatch
from causal_policy_gateway import INTERPRETER
from dynamic_causal_policy_planner import (
    DynamicCausalPolicyError,
    causal_quality_gate,
    plan_dynamic_causal_policy,
    run_dynamic_causal_policy_ticket,
)
from dynamic_family_router import DynamicFamilyRoutingError, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket


def dynamic_pipeline() -> dict[str, object]:
    return {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "causal-policy dynamic family test",
        "upstream_refs": [],
    }


def causal_ticket(
    *,
    task_id: str,
    decision_class: str = "exploratory",
    count: int = 40,
    mode: str = "backdoor_adjustment",
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    confounder = [(index % 10) / 9.0 for index in range(count)]
    treatment = [
        1 if ((index * 7) % 10) < (3 + int(4 * confounder[index])) else 0
        for index in range(count)
    ]
    outcome = [
        2.5 * treatment[index]
        + 1.2 * confounder[index]
        + ((index % 3) - 1) * 0.05
        for index in range(count)
    ]
    inputs: dict[str, object] = {
        "mode": mode,
        "treatment": treatment,
        "outcome": outcome,
        "confounders": {"baseline_risk": confounder},
    }
    if context is not None:
        inputs["dynamic_context"] = dict(context)
    return {
        "task_id": task_id,
        "operation": "causal_policy_evaluation",
        "inputs": inputs,
        "pipeline": dynamic_pipeline(),
        "quality_profile": {
            "decision_class": decision_class,
            "probabilistic_claim": False,
        },
    }


class DynamicCausalFamilyTests(unittest.TestCase):
    def test_router_admits_structured_observational_causal_ticket(self) -> None:
        ticket = causal_ticket(task_id="causal-router")
        self.assertEqual(resolve_dynamic_family(ticket), "causal-policy")

    def test_router_rejects_unadmitted_dynamic_causal_mode(self) -> None:
        ticket = causal_ticket(task_id="causal-router-did")
        ticket["inputs"]["mode"] = "difference_in_differences_refuted"
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    def test_exploratory_ticket_selects_only_primary_effect(self) -> None:
        ticket = causal_ticket(task_id="causal-exploratory")
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(plan["stage_order"], ["primary_effect"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["performed"])
        self.assertTrue(cross["passed"])
        self.assertEqual(cross["optional_node_count"], 3)

    def test_formal_ticket_selects_statistics_and_alternate_estimator(self) -> None:
        ticket = causal_ticket(task_id="causal-formal", decision_class="formal")
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(
            plan["stage_order"],
            ["outcome_statistics", "alternate_estimate", "primary_effect"],
        )
        self.assertTrue(plan["planning_features"]["decision_class"] == "formal")
        self.assertTrue(plan["optimization"]["selected_nodes"]["alternate_estimate"])
        self.assertFalse(plan["optimization"]["selected_nodes"]["placebo_refutation"])

    def test_explicit_placebo_request_forces_refutation_stage(self) -> None:
        ticket = causal_ticket(
            task_id="causal-placebo",
            context={"placebo_refutation": True, "placebo_seed": 7, "placebo_repetitions": 40},
        )
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(plan["stage_order"], ["placebo_refutation", "primary_effect"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["placebo_refutation"])

    def test_high_stakes_selects_full_causal_family(self) -> None:
        ticket = causal_ticket(task_id="causal-high-stakes", decision_class="high_stakes")
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(
            plan["stage_order"],
            ["outcome_statistics", "alternate_estimate", "placebo_refutation", "primary_effect"],
        )
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

        runtime = managed_runtime_plan(ticket)
        self.assertEqual(runtime["capability_pack"], "dynamic-orchestration")
        self.assertEqual(runtime["dynamic_family"], "causal-policy")
        self.assertEqual(runtime["dynamic_entry_contract"], "causal_policy_evaluation")
        requirement_names = [Path(item).name for item in runtime["requirements"]]
        self.assertEqual(requirement_names, ["requirements-ortools.txt"])
        isolated = runtime["isolated_environment"]
        self.assertEqual(isolated["name"], "causal-policy")
        self.assertEqual([Path(item).name for item in isolated["requirements"]], ["requirements-causal.txt"])
        self.assertEqual(isolated["network_policy"], "inherit-deny-at-execution")
        self.assertFalse(isolated["ticket_supplied_requirements_allowed"])
        self.assertEqual(runtime["network_policy"], "deny")
        self.assertFalse(runtime["automatic_parallel_execution"])

    def test_high_stakes_requires_enough_rows_for_placebo(self) -> None:
        ticket = causal_ticket(task_id="causal-high-stakes-short", decision_class="high_stakes", count=12)
        with self.assertRaises(DynamicCausalPolicyError):
            plan_dynamic_causal_policy(ticket)

    def test_objective_text_does_not_change_causal_selection(self) -> None:
        ticket = causal_ticket(task_id="causal-objective-text")
        ticket["objective"] = "Run every diagnostic, refutation, alternate estimator and robustness tool"
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(plan["stage_order"], ["primary_effect"])
        self.assertFalse(plan["objective_text_used"])

    def test_quality_gate_rejects_high_stakes_alternate_gate_failure(self) -> None:
        plan = {
            "stage_order": ["alternate_estimate", "placebo_refutation", "primary_effect"],
            "planning_features": {"decision_class": "high_stakes"},
        }
        stage_results = {
            "alternate_estimate": {
                "effect": 2.4,
                "causal_claim_allowed": False,
            },
            "placebo_refutation": {
                "refutation_passed": True,
                "empirical_p_value": 0.01,
                "repetitions": 200,
            },
            "primary_effect": {
                "mode": "backdoor_adjustment",
                "effect": 2.5,
                "causal_claim_allowed": True,
            },
        }
        gate = causal_quality_gate(stage_results, plan)
        self.assertEqual(gate["status"], "REJECT_CAUSAL_CLAIM")
        self.assertFalse(gate["causal_claim_allowed"])

    def test_real_dowhy_execution_runs_full_high_stakes_family_serially(self) -> None:
        if not INTERPRETER.is_file():
            self.skipTest("isolated DoWhy runtime is exercised in the dedicated dynamic causal workflow")
        ticket = causal_ticket(
            task_id="causal-real-full",
            decision_class="high_stakes",
            context={"placebo_seed": 17, "placebo_repetitions": 40},
        )
        expected = [
            "outcome_statistics",
            "alternate_estimate",
            "placebo_refutation",
            "primary_effect",
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_causal_policy_ticket(ticket, Path(directory), compute_dispatch.OPERATIONS)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["dynamic_family"], "causal-policy")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(result["results"]["final_stage"], "primary_effect")
            self.assertEqual(result["results"]["final_result"]["mode"], "backdoor_adjustment")
            self.assertEqual(
                result["results"]["final_result"]["engine"]["runtime_isolation"],
                "fixed-venv",
            )
            self.assertEqual(len(result["results"]["stage_receipts"]), 4)
            self.assertTrue(all(row["status"] == "PASS" for row in result["results"]["stage_receipts"]))
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertTrue(result["results"]["optimization"]["global_optimal_proven"])
            self.assertIn(result["results"]["quality_gate"]["status"], {"PASS", "WARN", "REJECT_CAUSAL_CLAIM"})
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])

            root = Path(directory)
            state = json.loads((root / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "PASS")
            self.assertEqual(state["family"], "causal-policy")
            self.assertTrue(all("input_sha256" in row and "output_sha256" in row for row in state["stages"]))
            audit = json.loads((root / "compute-audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["solver_status"], "OPTIMAL")
            self.assertTrue(audit["global_optimal_proven"])
            self.assertFalse(audit["network_used"])
            self.assertEqual(audit["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
