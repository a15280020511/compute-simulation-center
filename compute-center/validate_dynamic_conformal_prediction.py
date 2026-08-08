#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_conformal_prediction_planner import run_dynamic_conformal_prediction_ticket


def fixture():
    train_x = [[index / 10.0] for index in range(40)]
    noise = [-0.12, 0.08, -0.05, 0.10, -0.02, 0.04]
    train_y = [2.0 * row[0] + 1.0 + noise[index % len(noise)] for index, row in enumerate(train_x)]
    predict_x = [[4.0 + index / 10.0] for index in range(10)]
    heldout_noise = [0.05, -0.06, 0.08, -0.04, 0.03, -0.05, 0.07, -0.02, 0.04, -0.03]
    observed = [2.0 * row[0] + 1.0 + heldout_noise[index] for index, row in enumerate(predict_x)]
    return train_x, train_y, predict_x, observed


def main() -> None:
    train_x, train_y, predict_x, observed = fixture()
    ticket = {
        "task_id": "dynamic-conformal-prediction-validator",
        "objective": "Validate repository-controlled conformal prediction without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "mapie_conformal_interval",
            "train_x": train_x,
            "train_y": train_y,
            "predict_x": predict_x,
            "confidence": 0.9,
            "cv": 5,
            "seed": 7,
            "conformal_context": {
                "validation_observed": observed,
                "interval_validation_requested": True,
                "width_consistency_requested": True,
                "point_feedback_requested": True,
                "width_consistency_tolerance": 1e-12,
                "drift_ratio_threshold": 2.0,
                "minimum_empirical_coverage": 0.8,
                "maximum_average_interval_width": 1.0,
                "maximum_mean_interval_score": 2.0,
                "maximum_point_rmse": 0.2,
                "maximum_absolute_bias": 0.1,
                "target_tolerance": 0.0,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real conformal interval and held-out outcome validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-conformal-"))
    try:
        result = run_dynamic_conformal_prediction_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis},
        )
        expected = [
            "mapie_conformal_interval",
            "prediction_interval_validation",
            "interval_width_consistency_audit",
            "interval_target_audit",
            "realized_outcome_feedback",
            "point_target_audit",
        ]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        interval = validation["prediction_interval_validation"]
        feedback = validation["realized_outcome_feedback"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 785
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert len(primary["predictions"]) == 10
        assert len(primary["lower_bounds"]) == 10
        assert len(primary["upper_bounds"]) == 10
        assert abs(primary["mean_interval_width"] - interval["average_interval_width"]) <= 1e-12
        assert interval["empirical_coverage"] >= 0.8
        assert interval["average_interval_width"] <= 1.0
        assert interval["mean_interval_score"] <= 2.0
        assert feedback["rmse"] <= 0.2
        assert abs(feedback["bias"]) <= 0.1
        assert validation["interval_width_consistency_audit"]["status"] == "PASS"
        assert validation["interval_target_audit"]["status"] == "PASS"
        assert validation["point_target_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": "OPTIMAL",
            "selector_objective": 785,
            "mapie_mean_interval_width": primary["mean_interval_width"],
            "empirical_coverage": interval["empirical_coverage"],
            "coverage_error": interval["coverage_error"],
            "average_interval_width": interval["average_interval_width"],
            "median_interval_width": interval["median_interval_width"],
            "mean_interval_score": interval["mean_interval_score"],
            "point_rmse": feedback["rmse"],
            "point_bias": feedback["bias"],
            "feedback_status": feedback["feedback_status"],
            "width_consistency": validation["interval_width_consistency_audit"]["status"],
            "interval_targets": validation["interval_target_audit"]["status"],
            "point_targets": validation["point_target_audit"]["status"],
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
