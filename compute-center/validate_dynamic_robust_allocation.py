#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_robust_allocation_planner import run_dynamic_robust_allocation_ticket


MATRIX = [
    [0.10, 0.02, 0.05],
    [0.00, 0.09, 0.05],
    [0.08, 0.03, 0.05],
    [0.02, 0.08, 0.05],
]


def main() -> None:
    ticket = {
        "task_id": "dynamic-robust-allocation-validator",
        "objective": "Validate repository-controlled robust allocation without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "rsome_robust_allocation",
            "scenario_returns": MATRIX,
            "asset_names": ["A", "B", "C"],
            "robust_allocation_context": {
                "independent_crosscheck_requested": True,
                "objective_consistency_tolerance": 1e-8,
                "feasibility_audit_requested": True,
                "feasibility_tolerance": 1e-8,
                "minimum_worst_case_return": 0.051,
                "minimum_mean_return": 0.052,
                "maximum_single_asset_weight": 0.58,
                "allocation_target_tolerance": 0.0,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real robust-allocation cross-solver validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-robust-allocation-"))
    try:
        result = run_dynamic_robust_allocation_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis},
        )
        expected_order = [
            "rsome_robust_allocation",
            "ortools_maximin_crosscheck",
            "worst_case_objective_consistency_audit",
            "allocation_feasibility_audit",
            "allocation_target_audit",
        ]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        cross = validation["ortools_maximin_crosscheck"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected_order
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 645
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert abs(primary["weights"][0] - 3.0 / 7.0) < 1e-6
        assert abs(primary["weights"][1] - 4.0 / 7.0) < 1e-6
        assert abs(primary["weights"][2]) < 1e-6
        assert abs(primary["worst_case_return"] - 9.0 / 175.0) < 1e-8
        assert abs(primary["mean_return"] - 37.0 / 700.0) < 1e-8
        assert cross["status"] == "optimal"
        assert cross["optimality_not_guaranteed"] is False
        assert abs(cross["objective_value"] - primary["worst_case_return"]) < 1e-8
        assert validation["worst_case_objective_consistency_audit"]["status"] == "PASS"
        assert validation["allocation_feasibility_audit"]["status"] == "PASS"
        assert validation["allocation_target_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected_order,
            "selector_status": "OPTIMAL",
            "selector_objective": 645,
            "rsome_weights": primary["weights"],
            "rsome_worst_case_return": primary["worst_case_return"],
            "rsome_mean_return": primary["mean_return"],
            "ortools_objective": cross["objective_value"],
            "objective_consistency": validation["worst_case_objective_consistency_audit"]["status"],
            "feasibility": validation["allocation_feasibility_audit"]["status"],
            "targets": validation["allocation_target_audit"]["status"],
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
