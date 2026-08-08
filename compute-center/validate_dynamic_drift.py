#!/usr/bin/env python3
"""Real production-dispatch validation for the dynamic drift family."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_drift_planner import run_dynamic_drift_ticket


def main() -> None:
    reference = [[0.01 * (i % 5), float(i % 7)] for i in range(60)]
    current = [[5.0 + 0.01 * (i % 5), float(i % 7)] for i in range(60)]
    ticket = {
        "task_id": "dynamic-drift-validator",
        "objective": "Validate repository-controlled drift orchestration without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "evidently_data_drift",
            "reference": reference,
            "current": current,
            "variable_names": ["shifted", "stable"],
            "drift_share": 0.5,
            "screening_alpha": 0.05,
            "drift_context": {
                "adwin_cross_check_requested": True,
                "change_point_cross_check_requested": True,
                "expected_drift_share": 0.5,
                "drift_share_tolerance": 0.0,
                "adwin_delta": 0.002,
                "change_point_cost_model": "l2",
                "change_point_penalty": 5.0,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic drift validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-drift-"))
    try:
        result = run_dynamic_drift_ticket(ticket, root, {"finance_decision_analysis": finance_decision_analysis})
        expected = ["distribution_drift", "adwin_cross_check", "change_point_cross_check", "drift_share_audit"]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 395
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert primary["drifted_columns_screen"] == 1
        assert abs(primary["drift_share_screen"] - 0.5) <= 1e-12
        assert validation["adwin_cross_check"]["drift_count"] >= 1
        assert 60 in validation["change_point_cross_check"]["change_points"]
        assert validation["drift_share_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": result["results"]["optimization"]["solver_status"],
            "selector_objective": result["results"]["optimization"]["objective_value"],
            "drift_share_screen": primary["drift_share_screen"],
            "adwin_drift_indices": validation["adwin_cross_check"]["drift_indices"],
            "change_points": validation["change_point_cross_check"]["change_points"],
            "branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"],
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
