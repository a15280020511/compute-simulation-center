#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_calibration_planner import run_dynamic_calibration_ticket


def main() -> None:
    x = [0.5 * i for i in range(11)]
    noise = [0.005, -0.003, 0.004, -0.002, 0.003, -0.004, 0.002, -0.001, 0.003, -0.002, 0.001]
    y = [3.0 * math.exp(-0.4 * value) + 2.0 + noise[index] for index, value in enumerate(x)]
    ticket = {
        "task_id": "dynamic-calibration-validator",
        "objective": "Validate repository-controlled calibration orchestration without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "lmfit_exponential_calibration",
            "x": x,
            "y": y,
            "initial": {"amplitude": 2.5, "decay": 0.3, "offset": 1.5},
            "calibration_context": {
                "residual_profile_requested": True,
                "rmse_consistency_requested": True,
                "rmse_consistency_tolerance": 1e-10,
                "maximum_abs_residual_mean": 0.01,
                "expected_amplitude": 3.0,
                "amplitude_tolerance": 0.05,
                "expected_decay": 0.4,
                "decay_tolerance": 0.05,
                "expected_offset": 2.0,
                "offset_tolerance": 0.05,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic calibration validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-calibration-"))
    try:
        result = run_dynamic_calibration_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis, "descriptive_statistics": descriptive_statistics},
        )
        expected = ["exponential_calibration", "residual_statistics", "rmse_consistency_audit", "residual_bias_audit", "parameter_target_audit"]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        params = primary["parameters"]
        stats = validation["residual_statistics"]
        reconstructed_rmse = math.sqrt(stats["mean"] ** 2 + stats["standard_deviation_population"] ** 2)
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 745
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert abs(params["amplitude"]["value"] - 3.0) < 0.05
        assert abs(params["decay"]["value"] - 0.4) < 0.05
        assert abs(params["offset"]["value"] - 2.0) < 0.05
        assert primary["rmse"] > 0.0
        assert abs(reconstructed_rmse - primary["rmse"]) <= 1e-10
        assert validation["rmse_consistency_audit"]["status"] == "PASS"
        assert validation["residual_bias_audit"]["status"] == "PASS"
        assert validation["parameter_target_audit"]["status"] == "PASS"
        assert validation["parameter_target_audit"]["candidate_count"] == 3
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": "OPTIMAL",
            "selector_objective": 745,
            "amplitude": params["amplitude"]["value"],
            "decay": params["decay"]["value"],
            "offset": params["offset"]["value"],
            "rmse": primary["rmse"],
            "residual_mean": stats["mean"],
            "residual_population_std": stats["standard_deviation_population"],
            "reconstructed_rmse": reconstructed_rmse,
            "rmse_consistency": validation["rmse_consistency_audit"]["status"],
            "residual_bias": validation["residual_bias_audit"]["status"],
            "parameter_targets": validation["parameter_target_audit"]["status"],
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
