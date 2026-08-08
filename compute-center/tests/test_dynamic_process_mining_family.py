from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from dynamic_process_mining_planner import (
    DynamicProcessMiningError,
    plan_dynamic_process_mining,
    run_dynamic_process_mining_ticket,
)
from large_scale_data_intelligence_operations import large_scale_data_intelligence


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "process-mining dynamic-family test",
    "upstream_refs": [],
}


def process_ticket(*, decision_class: str = "exploratory", context=None):
    if context is None:
        context = {}
    return {
        "task_id": "dynamic-process-mining-test",
        "objective": "Objective prose must never select process-mining validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "pm4py_directly_follows",
            "cases": [
                {"case_id": "c1", "activities": ["A", "B", "C"]},
                {"case_id": "c2", "activities": ["A", "B", "D"]},
                {"case_id": "c3", "activities": ["A", "E", "D"]},
                {"case_id": "c4", "activities": ["A", "B", "C"]},
            ],
            "process_context": dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {
            "decision_class": decision_class,
            "publication_policy": "status_only",
        },
    }


ALL_SIGNALS = {
    "graph_summary_requested": True,
    "topology_consistency_requested": True,
    "expected_case_count": 4,
    "case_count_tolerance": 0,
    "expected_event_count": 12,
    "event_count_tolerance": 0,
    "expected_activity_count": 5,
    "activity_count_tolerance": 0,
    "expected_dfg_edge_count": 5,
    "dfg_edge_count_tolerance": 0,
}


class DynamicProcessMiningFamilyTests(unittest.TestCase):
    def test_router_and_minimal_runtime(self) -> None:
        value = process_ticket()
        self.assertEqual(resolve_dynamic_family(value), "process-mining")
        metadata = family_runtime_metadata(value)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(
            metadata["requirements"],
            ["requirements-ortools.txt", "requirements-global-pm4py.txt"],
        )
        self.assertEqual(
            metadata["entry_contract"],
            "finance_decision_analysis:pm4py_directly_follows",
        )

    def test_exploratory_ticket_selects_primary_only(self) -> None:
        plan = plan_dynamic_process_mining(process_ticket())
        self.assertEqual(plan["stage_order"], ["directly_follows_discovery"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

    def test_graph_summary_request_selects_graph_only(self) -> None:
        plan = plan_dynamic_process_mining(
            process_ticket(context={"graph_summary_requested": True})
        )
        self.assertEqual(
            plan["stage_order"],
            ["directly_follows_discovery", "workflow_graph_summary"],
        )

    def test_topology_consistency_requires_graph_summary(self) -> None:
        plan = plan_dynamic_process_mining(
            process_ticket(context={"topology_consistency_requested": True})
        )
        self.assertEqual(
            plan["stage_order"],
            [
                "directly_follows_discovery",
                "workflow_graph_summary",
                "topology_consistency_audit",
            ],
        )
        self.assertTrue(plan["optimization"]["required_by_node"]["workflow_graph_summary"])
        self.assertTrue(plan["optimization"]["required_by_node"]["topology_consistency_audit"])

    def test_process_targets_select_direct_audit(self) -> None:
        plan = plan_dynamic_process_mining(
            process_ticket(context={"expected_activity_count": 5})
        )
        self.assertEqual(
            plan["stage_order"],
            ["directly_follows_discovery", "process_target_audit"],
        )
        self.assertEqual(plan["planning_features"]["process_target_count"], 1)

    def test_all_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_process_mining(process_ticket(context=ALL_SIGNALS))
        self.assertEqual(
            plan["stage_order"],
            [
                "directly_follows_discovery",
                "workflow_graph_summary",
                "topology_consistency_audit",
                "process_target_audit",
            ],
        )
        self.assertEqual(plan["optimization"]["objective_value"], 510)
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["passed"])
        self.assertTrue(cross["unique_optimum"])

    def test_dangling_tolerance_fails_closed(self) -> None:
        with self.assertRaises(DynamicProcessMiningError):
            plan_dynamic_process_mining(
                process_ticket(context={"activity_count_tolerance": 1})
            )

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicProcessMiningError):
            plan_dynamic_process_mining(
                process_ticket(context={"run_every_process_tool": True})
            )

    def test_event_limit_fails_closed(self) -> None:
        value = process_ticket()
        value["inputs"]["cases"] = [
            {"case_id": f"c{index}", "activities": ["A"] * 200}
            for index in range(51)
        ]
        with self.assertRaises((DynamicProcessMiningError, ValueError)):
            plan_dynamic_process_mining(value)

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        value = process_ticket()
        value["objective"] = "Run graph summary, PageRank, topology audits and all process tools."
        plan = plan_dynamic_process_mining(value)
        self.assertEqual(plan["stage_order"], ["directly_follows_discovery"])
        self.assertFalse(plan["objective_text_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("pm4py") is not None,
        "pm4py is a managed optional dependency; real execution is enforced by process-mining CI",
    )
    def test_real_cross_tool_pipeline(self) -> None:
        value = process_ticket(context=ALL_SIGNALS)
        root = Path(tempfile.mkdtemp(prefix="dynamic-process-mining-"))
        try:
            result = run_dynamic_process_mining_ticket(
                value,
                root,
                {
                    "finance_decision_analysis": finance_decision_analysis,
                    "large_scale_data_intelligence": large_scale_data_intelligence,
                },
            )
            expected = [
                "directly_follows_discovery",
                "workflow_graph_summary",
                "topology_consistency_audit",
                "process_target_audit",
            ]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["stage_order"], expected)
            primary = result["results"]["final_result"]
            self.assertEqual(primary["case_count"], 4)
            self.assertEqual(primary["event_count"], 12)
            self.assertEqual(primary["activity_count"], 5)
            self.assertEqual(len(primary["directly_follows_edges"]), 5)
            validation = result["results"]["validation_results"]
            graph = validation["workflow_graph_summary"]
            self.assertEqual(graph["node_count"], 5)
            self.assertEqual(graph["edge_count"], 5)
            self.assertEqual(graph["component_count"], 1)
            self.assertEqual(graph["largest_component_size"], 5)
            self.assertEqual(validation["topology_consistency_audit"]["status"], "PASS")
            self.assertEqual(validation["topology_consistency_audit"]["candidate_count"], 2)
            self.assertEqual(validation["process_target_audit"]["status"], "PASS")
            self.assertEqual(validation["process_target_audit"]["candidate_count"], 4)
            self.assertEqual(result["results"]["optimization"]["solver_status"], "OPTIMAL")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 510)
            self.assertTrue(result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"])
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(
        importlib.util.find_spec("pm4py") is not None,
        "pm4py is a managed optional dependency",
    )
    def test_process_target_failure_is_informative(self) -> None:
        value = process_ticket(context={"expected_activity_count": 99})
        root = Path(tempfile.mkdtemp(prefix="dynamic-process-target-fail-"))
        try:
            result = run_dynamic_process_mining_ticket(
                value,
                root,
                {
                    "finance_decision_analysis": finance_decision_analysis,
                    "large_scale_data_intelligence": large_scale_data_intelligence,
                },
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(
                result["results"]["validation_results"]["process_target_audit"]["status"],
                "FAIL",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
