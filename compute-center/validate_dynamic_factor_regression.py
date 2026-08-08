#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_factor_regression_planner import run_dynamic_factor_regression_ticket


def _series():
    market = [(index - 15) / 100.0 for index in range(30)]
    value = [(((index * 7) % 11) - 5) / 100.0 for index in range(30)]
    noise = [((index % 5) - 2) * 0.0001 for index in range(30)]
    asset = [0.002 + 1.2 * market[index] - 0.6 * value[index] + noise[index] for index in range(30)]
    return asset, market, value


def main() -> None:
    asset, market, value = _series()
    ticket = {
        "task_id": "dynamic-factor-regression-validator",
        "objective": "Validate statsmodels OLS against an independently rebuilt NumPy least-squares solution.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "factor_regression",
            "asset_returns": asset,
            "factors": {"market": market, "value": value},
            "include_intercept": True,
            "covariance_type": "HAC",
            "hac_lags": 3,
            "factor_regression_context": {
                "exact_consistency_tolerance": 1e-9,
                "minimum_r_squared": 0.99,
                "r_squared_target_tolerance": 0.0,
                "maximum_residual_volatility": 0.001,
                "residual_volatility_target_tolerance": 0.0
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real statsmodels versus NumPy exact factor-regression validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-factor-regression-"))
    try:
        result = run_dynamic_factor_regression_ticket(ticket, root, {"finance_decision_analysis": finance_decision_analysis})
        expected_order = ["factor_regression", "numpy_exact_regression_audit", "r_squared_target_audit", "residual_volatility_target_audit"]
        primary = result["results"]["final_result"]
        validation = result["results"]["validation_results"]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected_order
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["objective_value"] == 465
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["unique_optimum"] is True
        assert primary["r_squared"] > 0.99
        assert abs(primary["parameters"]["market"]["coefficient"] - 1.2) < 0.01
        assert abs(primary["parameters"]["value"]["coefficient"] + 0.6) < 0.01
        assert validation["numpy_exact_regression_audit"]["status"] == "PASS"
        assert validation["numpy_exact_regression_audit"]["candidate_count"] == 5
        assert validation["r_squared_target_audit"]["status"] == "PASS"
        assert validation["residual_volatility_target_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected_order,
            "selector_status": "OPTIMAL",
            "selector_objective": 465,
            "alpha": primary["parameters"]["alpha"]["coefficient"],
            "market_beta": primary["parameters"]["market"]["coefficient"],
            "value_beta": primary["parameters"]["value"]["coefficient"],
            "r_squared": primary["r_squared"],
            "residual_volatility": primary["residual_volatility"],
            "exact_regression_consistency": validation["numpy_exact_regression_audit"]["status"],
            "r_squared_target": validation["r_squared_target_audit"]["status"],
            "residual_target": validation["residual_volatility_target_audit"]["status"],
            "branching": True,
            "network_used": False,
            "model_calls": 0,
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
