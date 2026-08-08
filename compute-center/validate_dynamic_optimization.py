#!/usr/bin/env python3
"""Real production-dispatch validation for the dynamic optimization family."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_optimization_planner import run_dynamic_optimization_ticket


def main() -> None:
    ticket = {
        "task_id": "dynamic-optimization-validator",
        "objective": "Validate repository-controlled optimization orchestration without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "mixed_integer_optimization",
            "variables": [
                {"name": "x", "type": "integer", "lower_bound": 0.0, "upper_bound": 4.0, "objective_coefficient": 3.0},
                {"name": "y", "type": "continuous", "lower_bound": 0.0, "upper_bound": 8.0, "objective_coefficient": 2.0}
            ],
            "constraints": [
                {"coefficients": {"x": 2.0, "y": 1.0}, "relation": "<=", "rhs": 8.0},
                {"coefficients": {"x": 1.0, "y": 2.0}, "relation": "<=", "rhs": 8.0}
            ],
            "maximize": true,
            "time_limit_seconds": 20,
            "optimization_context": {
                "independent_relaxation_crosscheck": true,
                "crosscheck_tolerance": 1e-7,
                "external_objective_value": 13.0,
                "external_objective_tolerance": 1e-7
            }
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic optimization validation",
            "upstream_refs": []
        },
        "quality_profile": {"decision_class": "formal", "publication_policy": "status_only"}
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-optimization-"))
    try:
        result = run_dynamic_optimization_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis},
        )
        expected = [
            "primary_optimization",
            "independent_relaxation",
            "relaxation_bound_audit",
            "external_objective_benchmark",
        ]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert abs(result["results"]["final_result"]["objective_value"] - 13.0) <= 1e-7
        assert result["results"]["validation_results"]["relaxation_bound_audit"]["status"] == "PASS"
        assert result["results"]["validation_results"]["external_objective_benchmark"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": result["results"]["optimization"]["solver_status"],
            "selector_objective": result["results"]["optimization"]["objective_value"],
            "primary_objective": result["results"]["final_result"]["objective_value"],
            "relaxation_objective": result["results"]["validation_results"]["independent_relaxation"]["objective_value"],
            "branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"]
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
