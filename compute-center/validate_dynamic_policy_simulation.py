#!/usr/bin/env python3
"""Real production-dispatch validation for the dynamic policy-simulation family."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_policy_simulation_planner import run_dynamic_policy_simulation_ticket


def main() -> None:
    ticket = {
        "task_id": "dynamic-policy-simulation-validator",
        "objective": "Validate repository-controlled policy microsimulation orchestration without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "policy_microsimulation",
            "incomes": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "tax_brackets": [
                {"threshold": 50.0, "rate": 0.1},
                {"threshold": 100.0, "rate": 0.2},
            ],
            "universal_transfer": 5.0,
            "poverty_line": 40.0,
            "policy_context": {
                "distribution_profile_requested": True,
                "mean_consistency_requested": True,
                "mean_consistency_tolerance": 1e-12,
                "minimum_net_fiscal_balance": 20.0,
                "maximum_gini_after": 0.27,
                "maximum_poverty_rate_after": 0.30,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic policy-simulation validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-policy-simulation-"))
    try:
        result = run_dynamic_policy_simulation_ticket(
            ticket,
            root,
            {
                "finance_decision_analysis": finance_decision_analysis,
                "descriptive_statistics": descriptive_statistics,
            },
        )
        expected = ["policy_microsimulation", "disposable_distribution_statistics", "mean_consistency_audit", "policy_target_audit"]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 530
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert abs(primary["tax_revenue"] - 70.0) <= 1e-12
        assert abs(primary["transfer_cost"] - 50.0) <= 1e-12
        assert abs(primary["net_fiscal_balance"] - 20.0) <= 1e-12
        assert abs(primary["mean_disposable_income"] - 53.0) <= 1e-12
        assert abs(primary["gini_before"] - 0.3) <= 1e-12
        assert abs(primary["gini_after"] - 0.26226415094339606) <= 1e-12
        assert abs(primary["poverty_rate_after"] - 0.3) <= 1e-12
        assert abs(validation["disposable_distribution_statistics"]["mean"] - 53.0) <= 1e-12
        assert validation["mean_consistency_audit"]["status"] == "PASS"
        assert validation["policy_target_audit"]["status"] == "PASS"
        assert validation["policy_target_audit"]["candidate_count"] == 3
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": result["results"]["optimization"]["solver_status"],
            "selector_objective": result["results"]["optimization"]["objective_value"],
            "tax_revenue": primary["tax_revenue"],
            "net_fiscal_balance": primary["net_fiscal_balance"],
            "mean_disposable_income": primary["mean_disposable_income"],
            "gini_before": primary["gini_before"],
            "gini_after": primary["gini_after"],
            "poverty_rate_after": primary["poverty_rate_after"],
            "mean_consistency": validation["mean_consistency_audit"]["status"],
            "policy_targets": validation["policy_target_audit"]["status"],
            "branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"],
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
