#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_assignment_optimization_planner import run_dynamic_assignment_optimization_ticket


def main() -> None:
    ticket = {
        "task_id": "dynamic-assignment-optimization-validator",
        "objective": "Validate OR-Tools SCIP assignment against an independently solved SciPy linear-sum assignment problem.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "assignment_optimization",
            "workers": ["A", "B", "C", "D"],
            "tasks": ["T1", "T2", "T3"],
            "costs": [[9.0, 2.0, 7.0], [6.0, 4.0, 3.0], [5.0, 8.0, 1.0], [7.0, 6.0, 9.0]],
            "maximize": False,
            "require_all_tasks": True,
            "assignment_optimization_context": {
                "exact_consistency_tolerance": 1e-9,
                "maximum_objective_value": 9.5,
                "objective_target_tolerance": 0.0
            }
        },
        "pipeline": {"pipeline_id": "dynamic-auto-v1", "stage_id": "dynamic", "sequence_reason": "real OR-Tools versus SciPy exact assignment validation", "upstream_refs": []},
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"}
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-assignment-"))
    try:
        result = run_dynamic_assignment_optimization_ticket(ticket, root, {"finance_decision_analysis": finance_decision_analysis})
        expected_order = ["assignment_optimization", "scipy_exact_assignment_audit", "objective_target_audit"]
        primary = result["results"]["final_result"]; validation = result["results"]["validation_results"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected_order
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 350
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert abs(primary["objective_value"] - 9.0) < 1e-9
        assert len(primary["assignments"]) == 3
        assert validation["scipy_exact_assignment_audit"]["status"] == "PASS"
        assert validation["scipy_exact_assignment_audit"]["candidate_count"] == 2
        assert validation["objective_target_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS", "stage_order": expected_order,
            "selector_status": "OPTIMAL", "selector_objective": 350,
            "objective_value": primary["objective_value"],
            "assignments": primary["assignments"],
            "exact_assignment_consistency": validation["scipy_exact_assignment_audit"]["status"],
            "objective_target": validation["objective_target_audit"]["status"],
            "branching": True, "network_used": False, "model_calls": 0
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
