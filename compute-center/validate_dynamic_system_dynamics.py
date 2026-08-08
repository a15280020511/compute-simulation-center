#!/usr/bin/env python3
"""Real production-dispatch validation for the dynamic system-dynamics family."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_system_dynamics_planner import run_dynamic_system_dynamics_ticket
from institutional_operations import system_dynamics_simulation


def main() -> None:
    ticket = {
        "task_id": "dynamic-system-dynamics-validator",
        "objective": "Validate repository-controlled system-dynamics orchestration without objective-text routing.",
        "operation": "system_dynamics_simulation",
        "inputs": {
            "mode": "feedback_delay",
            "steps": 20,
            "dt": 1.0,
            "initial_state": 10.0,
            "exogenous_input": 2.0,
            "decay_rate": 0.0,
            "feedback_gain": 0.0,
            "delay_steps": 2,
            "system_dynamics_context": {
                "trajectory_summary_requested": True,
                "robustness_parameter": "exogenous_input",
                "perturbation_fraction": 0.1,
                "max_absolute_deviation": 4.1,
                "external_final_value": 50.0,
                "external_final_tolerance": 1e-9,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic system-dynamics validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-system-dynamics-"))
    try:
        result = run_dynamic_system_dynamics_ticket(
            ticket,
            root,
            {
                "system_dynamics_simulation": system_dynamics_simulation,
                "descriptive_statistics": descriptive_statistics,
                "finance_decision_analysis": finance_decision_analysis,
            },
        )
        expected = [
            "primary_simulation",
            "trajectory_statistics",
            "robustness_simulation",
            "robustness_audit",
            "external_final_benchmark",
        ]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert abs(result["results"]["final_result"]["final_state"] - 50.0) <= 1e-9
        assert abs(result["results"]["validation_results"]["trajectory_statistics"]["mean"] - 30.0) <= 1e-9
        assert abs(result["results"]["validation_results"]["robustness_simulation"]["final_state"] - 54.0) <= 1e-9
        assert result["results"]["validation_results"]["robustness_audit"]["status"] == "PASS"
        assert result["results"]["validation_results"]["external_final_benchmark"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": result["results"]["optimization"]["solver_status"],
            "selector_objective": result["results"]["optimization"]["objective_value"],
            "primary_final_state": result["results"]["final_result"]["final_state"],
            "robustness_final_state": result["results"]["validation_results"]["robustness_simulation"]["final_state"],
            "trajectory_mean": result["results"]["validation_results"]["trajectory_statistics"]["mean"],
            "branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"],
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
