#!/usr/bin/env python3
"""Execute deterministic smoke and numerical-contract checks for think-tank packs."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from decision_intelligence_gateway import finance_decision_analysis


def _finite_tree(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AssertionError(f"non-finite output: {value}")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child)
        return
    if isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _run(mode: str, inputs: dict[str, Any], check: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    result = finance_decision_analysis({"mode": mode, **inputs})
    assert isinstance(result, dict)
    assert result.get("mode") == mode
    assert result.get("external_data_fetches") == 0
    assert result.get("brokerage_execution") is False
    assert result.get("arbitrary_code_allowed") is False
    _finite_tree(result)
    if check:
        check(result)
    return result


def validate_data() -> list[str]:
    completed = []
    _run("bounded_table_profile", {"records": [{"id": 1, "value": 2.0}, {"id": 2, "value": 4.0}, {"id": 3, "value": None}]})
    completed.append("bounded_table_profile")
    _run(
        "bounded_table_join",
        {"left": [{"id": 1, "left_value": 2.0}, {"id": 2, "left_value": 3.0}], "right": [{"id": 1, "right_value": 5.0}], "keys": ["id"], "how": "left"},
        lambda result: (_ for _ in ()).throw(AssertionError("join row count")) if result["row_count"] != 2 else None,
    )
    completed.append("bounded_table_join")
    _run(
        "schema_unit_validation",
        {
            "records": [{"amount": 1.0, "name": "a"}, {"amount": 2.0, "name": "b"}],
            "schema": {"amount": {"type": "number", "minimum": 0.0}, "name": {"type": "string"}},
            "unit_conversions": [{"value": 1.0, "source_unit": "kilometer", "target_unit": "meter"}],
        },
    )
    completed.append("schema_unit_validation")
    return completed


def validate_econometrics() -> list[str]:
    rng = np.random.default_rng(20260802)
    completed = []
    x = np.linspace(-2, 2, 60)
    y = 1.5 + 2.2 * x + rng.normal(0, 0.2, x.size)
    _run("robust_glm", {"x": [[float(v)] for v in x], "y": y.tolist(), "family": "gaussian", "cov_type": "HC3"})
    completed.append("robust_glm")

    entities = []
    times = []
    panel_x = []
    panel_y = []
    for entity in range(4):
        for period in range(12):
            value = period / 10 + entity * 0.05
            entities.append(f"e{entity}")
            times.append(period)
            panel_x.append([value])
            panel_y.append(1.0 + entity * 0.3 + 1.8 * value + rng.normal(0, 0.05))
    _run("panel_fixed_effects", {"x": panel_x, "y": panel_y, "entity": entities, "time": times, "entity_effects": True})
    completed.append("panel_fixed_effects")

    _run("survey_weighted_estimation", {"values": [1, 2, 3, 4, 5, 6], "weights": [1, 1, 2, 2, 3, 3]})
    completed.append("survey_weighted_estimation")
    _run("meta_analysis", {"effects": [0.1, 0.2, 0.15, 0.3], "variances": [0.02, 0.03, 0.02, 0.04]})
    completed.append("meta_analysis")

    durations = np.arange(1, 41, dtype=float) + rng.uniform(0.1, 1.0, 40)
    events = np.asarray([1 if i % 3 else 0 for i in range(40)])
    covariates = np.column_stack([np.linspace(0, 1, 40), rng.normal(size=40)])
    _run("survival_analysis", {"durations": durations.tolist(), "events": events.tolist(), "covariates": covariates.tolist(), "penalizer": 0.1})
    completed.append("survival_analysis")

    series = np.r_[rng.normal(0, 0.2, 30), rng.normal(3, 0.2, 30)]
    _run("change_point_detection", {"values": series.tolist(), "cost_model": "l2", "penalty": 5.0})
    completed.append("change_point_detection")

    groups = [f"g{i // 15}" for i in range(60)]
    mixed_x = np.linspace(-1, 1, 60)
    group_effect = np.asarray([int(group[1:]) * 0.4 for group in groups])
    mixed_y = 2.0 + group_effect + 1.3 * mixed_x + rng.normal(0, 0.15, 60)
    _run("mixed_effects_model", {"x": [[float(v)] for v in mixed_x], "y": mixed_y.tolist(), "groups": groups})
    completed.append("mixed_effects_model")

    _run("quantile_regression", {"x": [[float(v)] for v in x], "y": y.tolist(), "quantile": 0.5})
    completed.append("quantile_regression")

    cause = rng.normal(size=80)
    effect = np.zeros(80)
    for index in range(1, 80):
        effect[index] = 0.7 * cause[index - 1] + 0.2 * effect[index - 1] + rng.normal(0, 0.2)
    _run("granger_causality", {"cause": cause.tolist(), "effect": effect.tolist(), "max_lag": 3})
    completed.append("granger_causality")

    _run("power_analysis", {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "group_ratio": 1.0})
    completed.append("power_analysis")

    trend = np.arange(60) * 0.2 + np.sin(np.arange(60) * 2 * np.pi / 12) + rng.normal(0, 0.1, 60)
    _run("unobserved_components_forecast", {"values": trend.tolist(), "horizon": 4, "seasonal": 12})
    completed.append("unobserved_components_forecast")

    regimes = np.r_[rng.normal(-1, 0.2, 40), rng.normal(1, 0.2, 40)]
    _run("markov_regime_model", {"values": regimes.tolist(), "regimes": 2})
    completed.append("markov_regime_model")

    prices = np.linspace(5, 15, 40)
    quantities = 2_000 * prices ** -1.2 * np.exp(rng.normal(0, 0.02, 40))
    _run("price_elasticity", {"price": prices.tolist(), "quantity": quantities.tolist()})
    completed.append("price_elasticity")

    _run("customer_lifetime_value", {"period_margin": 80.0, "retention_rate": 0.85, "discount_rate": 0.01, "acquisition_cost": 120.0, "periods": 24})
    completed.append("customer_lifetime_value")

    features = np.r_[rng.normal(-1, 0.2, (30, 2)), rng.normal(1, 0.2, (30, 2))]
    _run("customer_segmentation", {"features": features.tolist(), "clusters": 2, "seed": 42})
    completed.append("customer_segmentation")

    churn_x = rng.normal(size=(100, 3))
    churn_score = churn_x[:, 0] - 0.5 * churn_x[:, 1]
    churn_y = (churn_score > np.median(churn_score)).astype(int)
    _run("churn_probability", {"features": churn_x.tolist(), "churned": churn_y.tolist(), "seed": 42})
    completed.append("churn_probability")

    channels = np.abs(rng.normal(100, 20, (80, 3)))
    outcome = 20 + channels @ np.asarray([0.4, 0.2, 0.1]) + rng.normal(0, 2, 80)
    _run("marketing_mix_regression", {"channels": channels.tolist(), "outcome": outcome.tolist(), "alphas": [0.1, 1.0, 10.0]})
    completed.append("marketing_mix_regression")

    _run("inventory_policy", {"annual_demand": 10000.0, "order_cost": 100.0, "holding_cost_per_unit": 5.0, "lead_time_demand_mean": 500.0, "lead_time_demand_sd": 50.0, "service_level": 0.95})
    completed.append("inventory_policy")

    _run("input_output_shock", {"technical_coefficients": [[0.1, 0.2], [0.05, 0.1]], "final_demand": [100.0, 80.0], "demand_shock": [10.0, -5.0]})
    completed.append("input_output_shock")

    choice_x = rng.normal(size=(120, 2))
    utilities = np.column_stack([0.2 + choice_x[:, 0], -0.1 + choice_x[:, 1], np.zeros(120)])
    choices = np.argmax(utilities + rng.gumbel(size=utilities.shape), axis=1)
    _run("consumer_choice_logit", {"features": choice_x.tolist(), "choices": choices.tolist()})
    completed.append("consumer_choice_logit")

    _run("process_capability", {"values": rng.normal(10, 0.15, 100).tolist(), "lower_specification": 9.0, "upper_specification": 11.0})
    completed.append("process_capability")
    return completed


def validate_finance() -> list[str]:
    rng = np.random.default_rng(44)
    returns = rng.normal([0.001, 0.0008, 0.0005], [0.01, 0.008, 0.006], size=(200, 3))
    completed = []
    _run("cvar_portfolio", {"returns": returns.tolist(), "alpha": 0.95, "minimum_expected_return": -0.01})
    completed.append("cvar_portfolio")
    _run("drawdown_constrained_portfolio", {"returns": returns.tolist(), "maximum_drawdown": 0.5, "minimum_expected_return": -0.01})
    completed.append("drawdown_constrained_portfolio")
    _run("financial_ratio_analysis", {"revenue": 1000.0, "cost_of_goods_sold": 600.0, "operating_income": 180.0, "net_income": 120.0, "total_assets": 800.0, "total_equity": 400.0, "current_assets": 300.0, "current_liabilities": 150.0, "total_debt": 200.0})
    completed.append("financial_ratio_analysis")
    factors = rng.normal(size=(120, 3)) * 0.01
    portfolio = 0.0005 + factors @ np.asarray([0.8, -0.2, 0.4]) + rng.normal(0, 0.003, 120)
    _run("factor_attribution", {"portfolio_returns": portfolio.tolist(), "factor_returns": factors.tolist()})
    completed.append("factor_attribution")
    return completed


def validate_decision() -> list[str]:
    completed = []
    _run("multiobjective_pareto", {"objective_coefficients": [[1.0, 0.0], [0.0, 1.0]], "lower_bounds": [0.0, 0.0], "upper_bounds": [1.0, 1.0], "population": 30, "generations": 20, "seed": 42})
    completed.append("multiobjective_pareto")
    _run("bounded_hyperparameter_search", {"parameters": [{"name": "x", "minimum": -2.0, "maximum": 2.0, "target": 0.5, "weight": 1.0}, {"name": "y", "minimum": 0.0, "maximum": 4.0, "target": 2.0, "weight": 2.0}], "trials": 30, "seed": 42})
    completed.append("bounded_hyperparameter_search")
    _run("algebraic_resource_optimization", {"objective": [3.0, 2.0], "constraint_matrix": [[1.0, 1.0], [2.0, 1.0]], "constraint_bounds": [4.0, 6.0], "maximize": True})
    completed.append("algebraic_resource_optimization")
    _run("strategic_sandbox", {"actors": ["a", "b", "c"], "payoff_matrix": [[1.0, 0.2, -0.1], [0.0, 1.2, 0.1], [0.2, -0.2, 1.0]], "initial_resources": [1.0, 1.0, 1.0], "periods": 20, "adaptation_rate": 0.2, "shock_standard_deviation": 0.01, "seed": 42})
    completed.append("strategic_sandbox")
    _run("influence_diagram", {"actions": ["invest", "wait"], "states": ["growth", "recession"], "state_probabilities": [0.7, 0.3], "utilities": [[10.0, -8.0], [3.0, 2.0]]})
    completed.append("influence_diagram")
    _run("policy_microsimulation", {"incomes": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], "tax_brackets": [{"threshold": 3000, "rate": 0.05}, {"threshold": 8000, "rate": 0.10}], "universal_transfer": 200, "poverty_line": 2500})
    completed.append("policy_microsimulation")
    return completed


def validate_bayesian() -> list[str]:
    rng = np.random.default_rng(55)
    values = np.r_[rng.normal(0.0, 0.3, 20), rng.normal(1.0, 0.3, 20)]
    _run("hierarchical_bayesian_mean", {"values": values.tolist(), "groups": ["a"] * 20 + ["b"] * 20, "draws": 100, "tune": 100, "seed": 42})
    return ["hierarchical_bayesian_mean"]


def validate_geospatial() -> list[str]:
    completed = []
    values = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    zones = [[0, 0, 1], [0, 1, 1], [2, 2, 2]]
    _run("raster_zonal_statistics", {"values": values, "zones": zones, "crs": "EPSG:4326"})
    completed.append("raster_zonal_statistics")
    _run("raster_change_detection", {"before": values, "after": [[1.0, 2.5, 3.0], [2.0, 4.0, 4.0], [3.0, 4.0, 6.0]], "threshold": 0.4})
    completed.append("raster_change_detection")
    _run("spatial_autocorrelation", {"values": [1.0, 1.2, 4.8, 5.0, 2.0], "neighbors": {"0": [1, 4], "1": [0, 2], "2": [1, 3], "3": [2, 4], "4": [3, 0]}, "permutations": 99, "seed": 42})
    completed.append("spatial_autocorrelation")
    return completed


VALIDATORS = {
    "data": validate_data,
    "econometrics": validate_econometrics,
    "finance": validate_finance,
    "decision": validate_decision,
    "bayesian": validate_bayesian,
    "geospatial": validate_geospatial,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=sorted(VALIDATORS), required=True)
    args = parser.parse_args()
    completed = VALIDATORS[args.pack]()
    print(json.dumps({"status": "PASS", "pack": args.pack, "completed_modes": completed, "mode_count": len(completed), "network_used": False, "model_calls": 0}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
