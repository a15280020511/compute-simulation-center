from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import compute_dispatch
from dynamic_causal_did_planner import plan_dynamic_causal_did, run_dynamic_causal_did_ticket
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HAS_DOWHY = importlib.util.find_spec("dowhy") is not None


def pipeline() -> dict[str, object]:
    return {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "causal DID dynamic family test",
        "upstream_refs": [],
    }


def base_ticket(*, task_id: str, advanced: bool = False) -> dict[str, object]:
    inputs: dict[str, object] = {
        "treated_pre": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "treated_post": [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
        "control_pre": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        "control_post": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "bootstrap_samples": 300,
        "seed": 20260807,
    }
    if advanced:
        inputs["dynamic_context"] = {
            "causal_design": "difference_in_differences",
            "allow_causal_policy_evaluation": True,
        }
        inputs["pretrend_tolerance"] = 0.25
    return {
        "task_id": task_id,
        "operation": "causal_screening",
        "inputs": inputs,
        "pipeline": pipeline(),
        "quality_profile": {
            "decision_class": "exploratory",
            "probabilistic_claim": False,
        },
    }


class DynamicCausalDidFamilyTests(unittest.TestCase):
    def test_screening_only_requires_only_ortools(self) -> None:
        ticket = base_ticket(task_id="causal-did-screening-only")
        self.assertEqual(resolve_dynamic_family(ticket), "causal-did")
        plan = plan_dynamic_causal_did(ticket)
        self.assertEqual(plan["stage_order"], ["screening"])
        self.assertFalse(plan["optimization"]["selected_nodes"]["did_policy_evaluation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        requirements = [Path(item).name for item in requirement_files_for_ticket(ticket)]
        self.assertEqual(requirements, ["requirements-ortools.txt"])

    def test_explicit_did_authorization_adds_advanced_stage_and_dependency(self) -> None:
        ticket = base_ticket(task_id="causal-did-advanced", advanced=True)
        plan = plan_dynamic_causal_did(ticket)
        self.assertEqual(plan["stage_order"], ["screening", "did_policy_evaluation"])
        self.assertTrue(plan["optimization"]["selected_nodes"]["did_policy_evaluation"])
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["performed"])
        self.assertTrue(cross["passed"])
        self.assertEqual(cross["optional_node_count"], 1)
        requirements = [Path(item).name for item in requirement_files_for_ticket(ticket)]
        self.assertEqual(requirements, ["requirements-ortools.txt", "requirements-causal.txt"])
        runtime = managed_runtime_plan(ticket)
        self.assertEqual(runtime["dynamic_family"], "causal-did")
        self.assertEqual(runtime["dynamic_entry_contract"], "causal_screening")
        self.assertEqual(runtime["dynamic_extra_requirements"], ["requirements-causal.txt"])

    def test_objective_text_cannot_authorize_causal_design(self) -> None:
        ticket = base_ticket(task_id="causal-did-objective-text")
        ticket["objective"] = "Prove this is causal using DoWhy and run the strongest causal method"
        plan = plan_dynamic_causal_did(ticket)
        self.assertEqual(plan["stage_order"], ["screening"])
        self.assertFalse(plan["objective_text_used"])
        self.assertFalse(plan["planning_features"]["advanced_causal_evaluation_authorized"])

    def test_wrong_design_fails_closed(self) -> None:
        ticket = base_ticket(task_id="causal-did-wrong-design")
        ticket["inputs"]["dynamic_context"] = {
            "causal_design": "instrumental_variable",
            "allow_causal_policy_evaluation": True,
        }
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    def test_advanced_unequal_windows_fail_before_dependency_install(self) -> None:
        ticket = base_ticket(task_id="causal-did-unaligned", advanced=True)
        ticket["inputs"]["treated_post"] = [17.0, 18.0, 19.0, 20.0, 21.0]
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    @unittest.skipUnless(HAS_DOWHY, "requires optional requirements-causal.txt")
    def test_advanced_execution_respects_parallel_trends_gate(self) -> None:
        ticket = base_ticket(task_id="causal-did-execute", advanced=True)
        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_causal_did_ticket(ticket, Path(directory), compute_dispatch.OPERATIONS)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"]["stage_order"], ["screening", "did_policy_evaluation"])
        final = result["results"]["final_result"]
        self.assertEqual(final["mode"], "difference_in_differences_refuted")
        self.assertTrue(final["parallel_trends_passed"])
        self.assertTrue(final["causal_claim_allowed"])
        self.assertEqual(final["claim_type"], "causal_effect")
        self.assertFalse(final["engine"]["network_used"])
        self.assertFalse(result["execution"]["network_used"])
        self.assertEqual(result["execution"]["model_calls"], 0)

    @unittest.skipUnless(HAS_DOWHY, "requires optional requirements-causal.txt")
    def test_failed_parallel_trend_downgrades_to_association(self) -> None:
        ticket = base_ticket(task_id="causal-did-conflict", advanced=True)
        ticket["inputs"]["treated_pre"] = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        ticket["inputs"]["control_pre"] = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        ticket["inputs"]["treated_post"] = [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
        ticket["inputs"]["control_post"] = [11.0, 11.0, 11.0, 11.0, 11.0, 11.0]
        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_causal_did_ticket(ticket, Path(directory), compute_dispatch.OPERATIONS)
        final = result["results"]["final_result"]
        self.assertFalse(final["parallel_trends_passed"])
        self.assertFalse(final["causal_claim_allowed"])
        self.assertEqual(final["claim_type"], "association_only")

    def test_family_metadata_requires_explicit_advanced_authorization(self) -> None:
        basic = family_runtime_metadata(base_ticket(task_id="causal-meta-basic"))
        advanced = family_runtime_metadata(base_ticket(task_id="causal-meta-advanced", advanced=True))
        self.assertEqual(basic["extra_requirements"], [])
        self.assertFalse(basic["advanced_requested"])
        self.assertEqual(advanced["extra_requirements"], ["requirements-causal.txt"])
        self.assertTrue(advanced["advanced_requested"])
        self.assertEqual(advanced["causal_design"], "difference_in_differences")


if __name__ == "__main__":
    unittest.main()
