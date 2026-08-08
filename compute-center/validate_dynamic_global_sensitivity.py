#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_global_sensitivity_planner import run_dynamic_global_sensitivity_ticket


def main() -> None:
    ticket = {
        "task_id": "dynamic-global-sensitivity-validator",
        "objective": "Validate SALib Sobol analysis against exact linear-uniform variance decomposition.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "sobol_sensitivity",
            "parameters": [
                {"name": "x", "minimum": -1.0, "maximum": 1.0},
                {"name": "y", "minimum": -1.0, "maximum": 1.0},
                {"name": "z", "minimum": -1.0, "maximum": 1.0},
            ],
            "model": {
                "intercept": 5.0,
                "linear": {"x": 1.0, "y": 2.0, "z": 3.0},
                "quadratic": {},
                "interactions": [],
            },
            "base_samples": 1024,
            "seed": 11,
            "global_sensitivity_context": {
                "exact_index_consistency_requested": True,
                "exact_moment_consistency_requested": True,
                "index_consistency_tolerance": 0.03,
                "moment_consistency_tolerance": 0.03,
                "minimum_total_order_by_parameter": {"z": 0.60},
                "target_tolerance": 0.0,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real SALib versus closed-form sensitivity validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-global-sensitivity-"))
    try:
        result = run_dynamic_global_sensitivity_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis},
        )
        expected_order = [
            "salib_sobol_sensitivity",
            "exact_index_consistency_audit",
            "exact_moment_consistency_audit",
            "sensitivity_target_audit",
        ]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        rows = {row["parameter"]: row for row in primary["ranking"]}
        exact = {"x": 1.0 / 14.0, "y": 4.0 / 14.0, "z": 9.0 / 14.0}
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected_order
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 475
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert primary["base_samples"] == 1024
        assert primary["evaluations"] == 5120
        assert [row["parameter"] for row in primary["ranking"]] == ["z", "y", "x"]
        for name, expected in exact.items():
            assert abs(rows[name]["first_order"] - expected) <= 0.03
            assert abs(rows[name]["total_order"] - expected) <= 0.03
        assert abs(primary["output_distribution"]["mean"] - 5.0) <= 0.03
        assert abs(primary["output_distribution"]["standard_deviation"] - math.sqrt(14.0 / 3.0)) <= 0.03
        assert validation["exact_index_consistency_audit"]["status"] == "PASS"
        assert validation["exact_index_consistency_audit"]["candidate_count"] == 6
        assert validation["exact_moment_consistency_audit"]["status"] == "PASS"
        assert validation["exact_moment_consistency_audit"]["candidate_count"] == 2
        assert validation["sensitivity_target_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected_order,
            "selector_status": "OPTIMAL",
            "selector_objective": 475,
            "evaluations": primary["evaluations"],
            "ranking": primary["ranking"],
            "output_distribution": primary["output_distribution"],
            "exact_indices": exact,
            "exact_mean": 5.0,
            "exact_standard_deviation": math.sqrt(14.0 / 3.0),
            "index_consistency": validation["exact_index_consistency_audit"]["status"],
            "moment_consistency": validation["exact_moment_consistency_audit"]["status"],
            "targets": validation["sensitivity_target_audit"]["status"],
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
