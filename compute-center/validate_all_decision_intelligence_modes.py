#!/usr/bin/env python3
"""Execute every allowlisted Decision Intelligence V2 mode through production dispatch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_dispatch  # noqa: E402
from decision_intelligence_gateway import SUPPORTED_MODES  # noqa: E402


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fixtures() -> dict[str, dict[str, Any]]:
    prices_a = [100.0 * (1.001 ** i) * (1.0 + 0.01 * math.sin(i / 5.0)) for i in range(120)]
    prices_b = [90.0 * (1.0007 ** i) * (1.0 + 0.008 * math.cos(i / 7.0)) for i in range(120)]
    backtest_prices = [100.0 + 0.35 * i + (2.0 if i % 15 < 8 else -2.0) for i in range(160)]
    factor = [(value - 20) / 1000 for value in range(1, 61)]
    asset = [0.001 + 1.5 * value + 0.0001 * math.sin(index) for index, value in enumerate(factor)]
    forecast_data = [10.0 + 0.4 * index + (1.5 if index % 12 < 6 else -1.5) for index in range(72)]
    return {
        "performance_metrics": {
            "mode": "performance_metrics",
            "returns": [0.01, -0.005, 0.012, 0.004, -0.002, 0.009, 0.003, -0.004],
            "periods_per_year": 252,
            "risk_free_rate": 0.02,
            "confidence": 0.95,
        },
        "portfolio_optimization": {
            "mode": "portfolio_optimization",
            "prices": {"asset_a": prices_a, "asset_b": prices_b},
            "periods_per_year": 252,
            "risk_free_rate": 0.0,
            "objective": "min_volatility",
            "weight_bounds": [0.0, 1.0],
        },
        "investment_projection": {
            "mode": "investment_projection",
            "initial_principal": 100000,
            "monthly_contribution": 3000,
            "annual_return": 0.06,
            "annual_fee_rate": 0.005,
            "annual_inflation_rate": 0.02,
            "years": 10,
        },
        "business_unit_economics": {
            "mode": "business_unit_economics",
            "price_per_unit": 80,
            "variable_cost_per_unit": 45,
            "fixed_costs": 100000,
            "expected_units": 5000,
            "customer_acquisition_cost": 50,
            "gross_profit_per_customer": 20,
            "retention_months": 8,
        },
        "capital_budgeting": {
            "mode": "capital_budgeting",
            "cash_flows": [-100000, 30000, 35000, 40000, 45000],
            "discount_rate": 0.08,
        },
        "strategy_backtest": {
            "mode": "strategy_backtest",
            "prices": backtest_prices,
            "strategy": "moving_average_crossover",
            "fast_window": 5,
            "slow_window": 20,
            "fees": 0.001,
            "slippage": 0.001,
            "initial_cash": 100000,
        },
        "factor_regression": {
            "mode": "factor_regression",
            "asset_returns": asset,
            "factors": {"market": factor},
            "covariance_type": "HC1",
        },
        "walk_forward_backtest": {
            "mode": "walk_forward_backtest",
            "prices": backtest_prices,
            "fast_window": 5,
            "slow_window": 20,
            "validation_window": 20,
            "fee_rate": 0.001,
            "slippage_rate": 0.001,
        },
        "risk_parity_allocation": {
            "mode": "risk_parity_allocation",
            "returns_by_asset": {
                "asset_a": [0.01, -0.01, 0.02, 0.0, 0.015, -0.005, 0.01, 0.005, -0.002, 0.012, 0.006, -0.004],
                "asset_b": [0.003, 0.002, -0.001, 0.004, 0.003, 0.002, -0.002, 0.004, 0.001, 0.003, 0.002, 0.001],
            },
        },
        "portfolio_stress_test": {
            "mode": "portfolio_stress_test",
            "weights": {"equity": 0.6, "bond": 0.4},
            "scenarios": [
                {"name": "recession", "asset_shocks": {"equity": -0.3, "bond": 0.05}},
                {"name": "inflation", "asset_shocks": {"equity": -0.12, "bond": -0.1}},
            ],
        },
        "sarimax_forecast": {
            "mode": "sarimax_forecast",
            "data": forecast_data,
            "horizon": 3,
            "holdout": 6,
            "order": [1, 1, 0],
            "seasonal_order": [0, 0, 0, 0],
        },
        "exponential_smoothing_forecast": {
            "mode": "exponential_smoothing_forecast",
            "data": forecast_data,
            "horizon": 4,
            "holdout": 6,
            "trend": "add",
        },
        "vector_autoregression_forecast": {
            "mode": "vector_autoregression_forecast",
            "series": {
                "demand": [50 + index * 0.3 + (index % 4) for index in range(60)],
                "price": [20 + index * 0.1 + ((index + 1) % 3) for index in range(60)],
            },
            "horizon": 3,
            "holdout": 4,
            "max_lags": 2,
        },
        "sobol_sensitivity": {
            "mode": "sobol_sensitivity",
            "parameters": [
                {"name": "demand", "minimum": 80, "maximum": 120},
                {"name": "margin", "minimum": 5, "maximum": 15},
            ],
            "base_samples": 64,
            "seed": 7,
            "model": {
                "intercept": 0,
                "linear": {"demand": 1.0, "margin": 3.0},
                "interactions": [{"left": "demand", "right": "margin", "coefficient": 0.02}],
            },
        },
        "mixed_integer_optimization": {
            "mode": "mixed_integer_optimization",
            "variables": [
                {"name": "x", "type": "integer", "lower": 0, "upper": 10, "objective": 3},
                {"name": "y", "type": "integer", "lower": 0, "upper": 10, "objective": 2},
            ],
            "constraints": [{"coefficients": {"x": 1, "y": 1}, "relation": "<=", "rhs": 4}],
            "maximize": True,
            "time_limit_seconds": 5,
        },
        "assignment_optimization": {
            "mode": "assignment_optimization",
            "workers": ["w1", "w2"],
            "tasks": ["t1", "t2"],
            "costs": [[1, 8], [7, 2]],
        },
        "vehicle_routing": {
            "mode": "vehicle_routing",
            "distance_matrix": [[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]],
            "vehicle_count": 1,
            "depot": 0,
            "time_limit_seconds": 2,
        },
        "weighted_mcda": {
            "mode": "weighted_mcda",
            "criteria": [
                {"name": "benefit", "weight": 0.4, "direction": "benefit"},
                {"name": "cost", "weight": 0.6, "direction": "cost"},
            ],
            "alternatives": [
                {"name": "A", "values": {"benefit": 80, "cost": 60}},
                {"name": "B", "values": {"benefit": 70, "cost": 20}},
            ],
        },
        "minimax_regret": {
            "mode": "minimax_regret",
            "actions": ["A", "B"],
            "scenarios": ["low", "high"],
            "payoffs": [[10, 30], [18, 25]],
            "objective": "maximize",
        },
        "value_of_information": {
            "mode": "value_of_information",
            "actions": ["A", "B"],
            "scenarios": ["low", "high"],
            "probabilities": [0.5, 0.5],
            "payoffs": [[10, 30], [18, 25]],
        },
        "competing_hypotheses": {
            "mode": "competing_hypotheses",
            "hypotheses": ["H1", "H2"],
            "evidence": [
                {"id": "E1", "reliability": 0.9, "diagnosticity": 1.0, "ratings": {"H1": 2, "H2": -2}},
                {"id": "E2", "reliability": 0.5, "diagnosticity": 0.5, "ratings": {"H1": 0, "H2": 1}},
            ],
        },
        "indicators_and_warnings": {
            "mode": "indicators_and_warnings",
            "indicators": [
                {"name": "liquidity", "current": 20, "warning_threshold": 30, "critical_threshold": 15, "direction": "lower_is_worse", "reliability": 1, "importance": 1},
                {"name": "defaults", "current": 5, "warning_threshold": 3, "critical_threshold": 5, "direction": "higher_is_worse", "reliability": 1, "importance": 1},
            ],
        },
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="all-decision-intelligence-mode-artifacts")
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    ticket_root = root / "_tickets"
    ticket_root.mkdir()

    cases = fixtures()
    expected_modes = set(SUPPORTED_MODES)
    supplied_modes = set(cases)
    if expected_modes != supplied_modes:
        raise AssertionError(
            f"mode fixture mismatch: missing={sorted(expected_modes-supplied_modes)} extra={sorted(supplied_modes-expected_modes)}"
        )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started_all = time.perf_counter()
    for index, mode in enumerate(sorted(cases), 1):
        output = root / mode
        ticket = {
            "task_id": f"allmodes-{index:02d}-{mode}",
            "objective": f"Production execution probe for decision-intelligence mode {mode}",
            "operation": "finance_decision_analysis",
            "inputs": cases[mode],
        }
        ticket_path = ticket_root / f"{mode}.json"
        ticket_path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            return_code = compute_dispatch.main([
                "--ticket", str(ticket_path), "--output-dir", str(output)
            ])
            assert return_code == 0, f"dispatcher returned {return_code}"
            transfer = load(output / "compute-result.json")
            audit = load(output / "compute-audit.json")
            diagnostics = load(output / "compute-diagnostics.json")
            preflight = load(output / "compute-preflight.json")
            gap_plan = load(output / "compute-data-gap-plan.json")
            expert_request = load(output / "compute-expert-review-request.json")
            manifest = load(output / "artifact-manifest.json")
            manifest_paths = {row["path"] for row in manifest["files"]}
            required = {
                "compute-result.json", "compute-audit.json", "compute-diagnostics.json",
                "compute-preflight.json", "compute-data-gap-plan.json",
                "compute-expert-review-request.json", "compute-summary.md",
            }
            assert transfer["status"] == "success"
            assert transfer["operation"] == "finance_decision_analysis"
            assert transfer["results"]["mode"] == mode
            assert transfer["execution"]["network_used"] is False
            assert transfer["execution"]["model_calls"] == 0
            assert transfer["relay_contract"]["sole_relay"] == "gpts-usage-center"
            assert audit["status"] == "PASS"
            assert diagnostics["status"] == "PASS"
            assert preflight["execution_allowed"] is True
            assert gap_plan["center_direct_contact_allowed"] is False
            assert expert_request["direct_center_delivery_allowed"] is False
            assert required <= manifest_paths
            rows.append({
                "mode": mode,
                "status": "PASS",
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "package_sha256": transfer["package_sha256"],
                "preflight_status": preflight["status"],
            })
        except Exception as exc:  # noqa: BLE001 - aggregate complete evidence
            failures.append({"mode": mode, "error_type": type(exc).__name__, "message": str(exc)})
            rows.append({
                "mode": mode,
                "status": "FAIL",
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "error_type": type(exc).__name__,
                "message": str(exc),
            })

    summary = {
        "schema_version": "all-decision-intelligence-modes-validation-v1",
        "status": "PASS" if not failures else "FAIL",
        "mode_count_expected": len(expected_modes),
        "mode_count_executed": len(rows),
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started_all, 6),
        "modes": sorted(expected_modes),
        "rows": rows,
        "failures": failures,
        "network_used": False,
        "model_calls": 0,
        "sole_relay": "gpts-usage-center",
    }
    summary["summary_sha256"] = canonical_sha(summary)
    (root / "all-decision-intelligence-modes-validation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
