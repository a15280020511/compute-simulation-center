from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import compute_dispatch
from dynamic_causal_policy_planner import plan_dynamic_causal_policy, run_dynamic_causal_policy_ticket
from dynamic_family_router import DynamicFamilyRoutingError, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket


def dynamic_pipeline() -> dict[str, object]:
    return {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "causal dynamic family test",
        "upstream_refs": [],
    }


def causal_ticket(
    *,
    task_id: str,
    decision_class: str = "exploratory",
    diagnostics: bool = False,
    refutation: bool = False,
    mode: str = "backdoor_adjustment",
) -> dict[str, object]:
    treatment = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    age = [20, 22, 21, 25, 30, 28, 35, 33, 40, 38, 45, 42]
    outcome = [5.0 + 2.0 * t + 0.1 * a for t, a in zip(treatment, age, strict=True)]
    inputs: dict[str, object] = {
        "mode": mode,
        "treatment": treatment,
        "outcome": outcome,
        "confounders": {"age": age},
    }
    context: dict[str, bool] = {}
    if diagnostics:
        context["causal_diagnostics"] = True
    if refutation:
        context["causal_refutation"] = True
    if context:
        inputs["dynamic_context"] = context
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


def causal_stub(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine = {"name": "dowhy-isolated-fixed-adapter", "version": "0.14", "network_used": False}
    boundary = "Causal language is permitted only when the mode-specific identification and refutation gates pass."
    mode = str(inputs["mode"])
    if mode == "placebo_policy_test":
        return {
            "engine": engine,
            "mode": mode,
            "actual_effect": 2.0,
            "placebo_mean": 0.01,
            "placebo_standard_deviation": 0.2,
            "empirical_p_value": 0.01,
            "refutation_passed": True,
            "repetitions": int(inputs["repetitions"]),
            "seed": int(inputs["seed"]),
            "confounders": sorted(inputs.get("confounders", {})),
            "interpretation_boundary": boundary,
        }
    return {
        "engine": engine,
        "mode": mode,
        "effect": 2.0,
        "identified": True,
        "confounders": sorted(inputs.get("confounders", {})),
        "causal_claim_allowed": True,
        "claim_type": "causal_effect",
        "observation_count": len(inputs["treatment"]),
        "interpretation_boundary": boundary,
    }


class DynamicCausalPolicyFamilyTests(unittest.TestCase):
    def test_exploratory_selects_only_required_estimate(self) -> None:
        ticket = causal_ticket(task_id="causal-exploratory")
        self.assertEqual(resolve_dynamic_family(ticket), "causal-policy")
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(plan["stage_order"], ["causal_estimate"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])
        self.assertEqual(plan["optimization"]["exhaustive_cross_check"]["optional_node_count"], 2)

    def test_explicit_diagnostics_selects_statistics(self) -> None:
        plan = plan_dynamic_causal_policy(causal_ticket(task_id="causal-diagnostics", diagnostics=True))
        self.assertEqual(plan["stage_order"], ["outcome_statistics", "causal_estimate"])
        self.assertTrue(plan["planning_features"]["diagnostics_requested"])

    def test_explicit_refutation_forces_placebo(self) -> None:
        plan = plan_dynamic_causal_policy(causal_ticket(task_id="causal-refutation", refutation=True))
        self.assertEqual(plan["stage_order"], ["causal_estimate", "placebo_refutation"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["placebo_refutation"])

    def test_high_stakes_selects_full_family(self) -> None:
        ticket = causal_ticket(task_id="causal-high-stakes", decision_class="high_stakes")
        plan = plan_dynamic_causal_policy(ticket)
        expected = ["outcome_statistics", "causal_estimate", "placebo_refutation"]
        self.assertEqual(plan["stage_order"], expected)
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["performed"])
        self.assertTrue(cross["passed"])
        self.assertEqual(plan["optimization"]["objective_value"], cross["best_objective"])

        requirements = requirement_files_for_ticket(ticket)
        self.assertEqual([Path(item).name for item in requirements], ["requirements-ortools.txt", "requirements-causal.txt"])
        runtime = managed_runtime_plan(ticket)
        self.assertEqual(runtime["dynamic_family"], "causal-policy")
        self.assertEqual(runtime["dynamic_entry_contract"], "causal_policy_evaluation")
        self.assertEqual(len(runtime["requirements"]), 2)
        self.assertEqual(runtime["network_policy"], "deny")

        operations = {
            "descriptive_statistics": compute_dispatch.OPERATIONS["descriptive_statistics"],
            "causal_policy_evaluation": causal_stub,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_causal_policy_ticket(ticket, Path(directory), operations)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["dynamic_family"], "causal-policy")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(result["results"]["final_stage"], "causal_estimate")
            self.assertTrue(result["results"]["final_result"]["causal_claim_allowed"])
            self.assertTrue(result["results"]["refutation_result"]["refutation_passed"])
            self.assertEqual(len(result["results"]["stage_receipts"]), 3)
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            state = json.loads((Path(directory) / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "PASS")
            self.assertEqual(state["family"], "causal-policy")
            self.assertTrue(all("input_sha256" in row and "output_sha256" in row for row in state["stages"]))

    def test_objective_text_does_not_trigger_refutation(self) -> None:
        ticket = causal_ticket(task_id="causal-objective-text")
        ticket["objective"] = "Run every causal diagnostic and every refutation tool available"
        plan = plan_dynamic_causal_policy(ticket)
        self.assertEqual(plan["stage_order"], ["causal_estimate"])
        self.assertFalse(plan["planning_features"]["refutation_requested"])
        self.assertFalse(plan["objective_text_used"])

    def test_router_rejects_causal_without_confounder(self) -> None:
        ticket = causal_ticket(task_id="causal-no-confounder")
        ticket["inputs"].pop("confounders")
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    def test_router_rejects_unadmitted_causal_mode(self) -> None:
        ticket = causal_ticket(task_id="causal-did", mode="difference_in_differences_refuted")
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)


if __name__ == "__main__":
    unittest.main()
