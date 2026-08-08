#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_transport_routing_planner import run_dynamic_transport_routing_ticket


def main() -> None:
    ticket = {
        "task_id": "dynamic-transport-routing-validator",
        "objective": "Validate AequilibraE shortest path against an independently rebuilt NetworkX directed graph.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "aequilibrae_shortest_path",
            "links": [
                {"a_node": 1, "b_node": 2, "cost": 2.0},
                {"a_node": 2, "b_node": 5, "cost": 2.0},
                {"a_node": 1, "b_node": 3, "cost": 1.0},
                {"a_node": 3, "b_node": 4, "cost": 1.0},
                {"a_node": 4, "b_node": 5, "cost": 1.0},
                {"a_node": 2, "b_node": 4, "cost": 10.0},
                {"a_node": 1, "b_node": 5, "cost": 9.0},
                {"a_node": 1, "b_node": 3, "cost": 4.0},
            ],
            "origin": 1,
            "destination": 5,
            "transport_routing_context": {
                "cost_consistency_tolerance": 1e-9,
                "maximum_total_cost": 3.1,
                "cost_target_tolerance": 0.0,
                "maximum_link_count": 3,
                "link_count_tolerance": 0,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real AequilibraE versus NetworkX exact routing validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-transport-routing-"))
    try:
        result = run_dynamic_transport_routing_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis},
        )
        expected_order = [
            "aequilibrae_shortest_path",
            "networkx_exact_route_audit",
            "route_cost_target_audit",
            "route_hop_target_audit",
        ]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected_order
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 465
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert primary["path_nodes"] == [1, 3, 4, 5]
        assert primary["link_count"] == 3
        assert abs(primary["total_cost"] - 3.0) <= 1e-9
        assert validation["networkx_exact_route_audit"]["status"] == "PASS"
        assert validation["networkx_exact_route_audit"]["candidate_count"] == 2
        assert validation["route_cost_target_audit"]["status"] == "PASS"
        assert validation["route_hop_target_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected_order,
            "selector_status": "OPTIMAL",
            "selector_objective": 465,
            "path_nodes": primary["path_nodes"],
            "path_links": primary["path_links"],
            "total_cost": primary["total_cost"],
            "link_count": primary["link_count"],
            "exact_route_consistency": validation["networkx_exact_route_audit"]["status"],
            "cost_target": validation["route_cost_target_audit"]["status"],
            "hop_target": validation["route_hop_target_audit"]["status"],
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
