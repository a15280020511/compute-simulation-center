from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from dynamic_family_router import DynamicFamilyRoutingError, resolve_dynamic_family
from dynamic_indirect_intelligence_planner import run_dynamic_indirect_intelligence_ticket
from tool_registry import managed_runtime_plan, requirement_files_for_ticket


def dynamic_pipeline() -> dict[str, object]:
    return {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "indirect intelligence dynamic family test",
        "upstream_refs": [],
    }


def indirect_ticket(*, task_id: str, mode: str = "indirect_intelligence_analysis") -> dict[str, object]:
    return {
        "task_id": task_id,
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": mode,
            "hypothesis": "技术A已经进入实际应用",
            "evidence": [
                {
                    "evref": "ev-1",
                    "analysis_class": "DIRECT",
                    "stance": "support",
                    "reliability": 0.8,
                }
            ],
        },
        "pipeline": dynamic_pipeline(),
    }


def fusion_stub(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "indirect_intelligence_analysis",
        "inference_id": "inf-test",
        "analysis_class": "INFERRED",
        "prior_probability": 0.5,
        "posterior_probability": 0.8,
        "confidence": 0.75,
        "inference_not_fact": True,
        "scope_extrapolation_allowed": False,
        "expert_semantic_synthesis_required_for_publication": True,
        "governance_release_gate_required": True,
        "network_used": False,
        "external_data_fetches": 0,
        "model_calls": 0,
        "automatic_parallel_execution": False,
        "stage_plan": {
            "selected_stages": ["contradiction_check"],
            "selection_engine": "ortools-cp-sat",
            "graph_engine": "networkx",
            "solver_status": "OPTIMAL",
            "serial_execution": True,
            "automatic_parallel_execution": False,
        },
        "stage_results": {
            "contradiction_check": {
                "claims": [{"claim": inputs["hypothesis"], "classification": "supported"}]
            }
        },
    }


class DynamicIndirectIntelligenceFamilyTests(unittest.TestCase):
    def test_router_and_runtime_plan_are_mode_specific(self) -> None:
        ticket = indirect_ticket(task_id="indirect-dynamic")
        self.assertEqual(resolve_dynamic_family(ticket), "indirect-intelligence")
        requirements = [Path(item).name for item in requirement_files_for_ticket(ticket)]
        self.assertEqual(len(requirements), 11)
        self.assertEqual(requirements[0], "requirements-ortools.txt")
        self.assertIn("requirements-intelligence-splink.txt", requirements)
        self.assertIn("requirements-bayesian-network.txt", requirements)
        self.assertIn("requirements-intelligence-problog.txt", requirements)
        runtime = managed_runtime_plan(ticket)
        self.assertEqual(runtime["dynamic_family"], "indirect-intelligence")
        self.assertEqual(
            runtime["dynamic_entry_contract"],
            "finance_decision_analysis:indirect_intelligence_analysis",
        )
        self.assertEqual(runtime["mode"], "indirect_intelligence_analysis")
        self.assertEqual(runtime["python_version"], "3.12")
        self.assertEqual(len(runtime["requirements"]), 11)
        self.assertEqual(runtime["network_policy"], "deny")
        self.assertFalse(runtime["automatic_parallel_execution"])

    def test_dynamic_adapter_delegates_to_single_fusion_engine(self) -> None:
        ticket = indirect_ticket(task_id="indirect-adapter")
        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_indirect_intelligence_ticket(
                ticket,
                Path(directory),
                {"finance_decision_analysis": fusion_stub},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["dynamic_family"], "indirect-intelligence")
            self.assertEqual(result["results"]["stage_order"], ["contradiction_check"])
            self.assertEqual(result["results"]["analysis_class"], "INFERRED")
            self.assertTrue(result["results"]["inference_not_fact"])
            self.assertFalse(result["results"]["scope_extrapolation_allowed"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            state = json.loads(
                (Path(directory) / "compute-dynamic-pipeline-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["status"], "PASS")
            self.assertEqual(state["family"], "indirect-intelligence")
            self.assertEqual(len(state["stages"]), 1)
            self.assertIn("output_sha256", state["stages"][0])

    def test_router_does_not_hijack_other_finance_modes(self) -> None:
        ticket = indirect_ticket(task_id="other-finance", mode="portfolio_optimization")
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    def test_router_rejects_missing_structured_evidence(self) -> None:
        ticket = indirect_ticket(task_id="missing-evidence")
        ticket["inputs"]["evidence"] = []
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)


if __name__ == "__main__":
    unittest.main()
