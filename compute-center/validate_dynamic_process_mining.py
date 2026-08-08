#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_process_mining_planner import run_dynamic_process_mining_ticket
from large_scale_data_intelligence_operations import large_scale_data_intelligence


def main() -> None:
    ticket = {
        "task_id": "dynamic-process-mining-validator",
        "objective": "Validate repository-controlled process-mining orchestration without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "pm4py_directly_follows",
            "cases": [
                {"case_id": "c1", "activities": ["A", "B", "C"]},
                {"case_id": "c2", "activities": ["A", "B", "D"]},
                {"case_id": "c3", "activities": ["A", "E", "D"]},
                {"case_id": "c4", "activities": ["A", "B", "C"]},
            ],
            "process_context": {
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
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic process-mining validation",
            "upstream_refs": [],
        },
        "quality_profile": {
            "decision_class": "exploratory",
            "publication_policy": "status_only",
        },
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-process-mining-"))
    try:
        result = run_dynamic_process_mining_ticket(
            ticket,
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
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        graph = validation["workflow_graph_summary"]
        dfg_edges = primary["directly_follows_edges"]
        edge_counts = {
            (row["source"], row["target"]): row["count"]
            for row in dfg_edges
        }

        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 510
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert primary["case_count"] == 4
        assert primary["event_count"] == 12
        assert primary["activity_count"] == 5
        assert len(dfg_edges) == 5
        assert edge_counts == {
            ("A", "B"): 3,
            ("A", "E"): 1,
            ("B", "C"): 2,
            ("B", "D"): 1,
            ("E", "D"): 1,
        }
        assert graph["node_count"] == 5
        assert graph["edge_count"] == 5
        assert graph["component_count"] == 1
        assert graph["largest_component_size"] == 5
        assert validation["topology_consistency_audit"]["status"] == "PASS"
        assert validation["topology_consistency_audit"]["candidate_count"] == 2
        assert validation["process_target_audit"]["status"] == "PASS"
        assert validation["process_target_audit"]["candidate_count"] == 4
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True

        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": "OPTIMAL",
            "selector_objective": 510,
            "case_count": primary["case_count"],
            "event_count": primary["event_count"],
            "activity_count": primary["activity_count"],
            "dfg_edge_count": len(dfg_edges),
            "dfg_edge_counts": {f"{left}->{right}": count for (left, right), count in sorted(edge_counts.items())},
            "graph_node_count": graph["node_count"],
            "graph_edge_count": graph["edge_count"],
            "component_count": graph["component_count"],
            "largest_component_size": graph["largest_component_size"],
            "topology_consistency": validation["topology_consistency_audit"]["status"],
            "process_targets": validation["process_target_audit"]["status"],
            "top_ranked_node": graph["ranking"][0]["node"] if graph["ranking"] else None,
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
