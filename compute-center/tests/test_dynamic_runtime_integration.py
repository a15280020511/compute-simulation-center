from __future__ import annotations

import copy
import unittest
from pathlib import Path

from dynamic_pipeline_planner import is_dynamic_pipeline_ticket, plan_dynamic_pipeline
from tool_registry import managed_runtime_plan, requirement_files_for_ticket


BASE_TICKET = {
    "task_id": "dynamic-runtime-test-0001",
    "operation": "scenario_compare",
    "inputs": {
        "model": {
            "intercept": 10.0,
            "coefficients": {"demand": 2.0, "cost": -1.0},
        },
        "scenarios": [
            {"name": "weak", "values": {"demand": 1.0, "cost": 4.0}},
            {"name": "base", "values": {"demand": 2.0, "cost": 3.0}},
            {"name": "strong", "values": {"demand": 4.0, "cost": 1.0}},
        ],
    },
    "pipeline": {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "structured dynamic orchestration integration test",
        "upstream_refs": [],
    },
    "quality_profile": {
        "decision_class": "exploratory",
        "probabilistic_claim": True,
        "publication_policy": "status_only",
    },
}


class DynamicRuntimeIntegrationTests(unittest.TestCase):
    def test_exact_dynamic_contract_is_detected(self) -> None:
        self.assertTrue(is_dynamic_pipeline_ticket(BASE_TICKET))
        wrong = copy.deepcopy(BASE_TICKET)
        wrong["pipeline"]["stage_id"] = "pipeline"
        self.assertFalse(is_dynamic_pipeline_ticket(wrong))

    def test_dynamic_ticket_resolves_only_pinned_ortools_bundle(self) -> None:
        requirements = requirement_files_for_ticket(BASE_TICKET)
        self.assertEqual(len(requirements), 1)
        self.assertEqual(Path(requirements[0]).name, "requirements-ortools.txt")
        plan = managed_runtime_plan(BASE_TICKET)
        self.assertEqual(plan["capability_pack"], "dynamic-orchestration")
        self.assertEqual(plan["maturity"], "controlled-preview")
        self.assertEqual(plan["network_policy"], "deny")
        self.assertEqual(plan["selection_engine"], "ortools-cp-sat")
        self.assertEqual(plan["graph_engine"], "networkx")
        self.assertFalse(plan["automatic_parallel_execution"])
        self.assertFalse(plan["dynamic_operation_discovery_allowed"])
        self.assertFalse(plan["arbitrary_code_allowed"])
        self.assertFalse(plan["arbitrary_requirements_allowed"])

    def test_ordinary_core_ticket_does_not_gain_ortools_dependency(self) -> None:
        ordinary = copy.deepcopy(BASE_TICKET)
        ordinary.pop("pipeline")
        requirements = requirement_files_for_ticket(ordinary)
        self.assertNotIn("requirements-ortools.txt", [Path(item).name for item in requirements])
        self.assertFalse(is_dynamic_pipeline_ticket(ordinary))

    def test_dynamic_plan_is_policy_optimal_and_controlled_preview(self) -> None:
        plan = plan_dynamic_pipeline(BASE_TICKET)
        self.assertEqual(plan["id"], "dynamic-auto-v1")
        self.assertEqual(plan["maturity"], "controlled-preview")
        self.assertEqual(
            plan["stage_order"],
            ["scenarios", "sensitivity", "risk_simulation"],
        )
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])


if __name__ == "__main__":
    unittest.main()
