#!/usr/bin/env python3
"""Execute deterministic smoke checks for all Exa institutional toolkit packs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from decision_intelligence_gateway import finance_decision_analysis


def finite_tree(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float, np.number)):
        if not math.isfinite(float(value)):
            raise AssertionError(f"non-finite output: {value}")
        return
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            finite_tree(child)


def run(mode: str, **inputs: Any) -> dict[str, Any]:
    result = finance_decision_analysis({"mode": mode, **inputs})
    assert result["mode"] == mode
    assert result["external_data_fetches"] == 0
    assert result["brokerage_execution"] is False
    assert result["arbitrary_code_allowed"] is False
    finite_tree(result)
    return result


def economics() -> list[str]:
    rng = np.random.default_rng(1101)
    n = 160
    x = rng.normal(size=(n, 3))
    fe = [f"g{i % 8}" for i in range(n)]
    fe_effect = np.asarray([int(item[1:]) * 0.2 for item in fe])
    y = 1.0 + x @ np.asarray([1.2, -0.4, 0.7]) + fe_effect + rng.normal(0, 0.2, n)
    run("high_dimensional_fixed_effects", x=x.tolist(), y=y.tolist(), fixed_effect=fe)
    treatment = 0.6 * x[:, 0] + rng.normal(0, 0.5, n)
    outcome = 2.0 * treatment + x[:, 1] - 0.5 * x[:, 2] + rng.normal(0, 0.5, n)
    run("double_machine_learning", x=x.tolist(), y=outcome.tolist(), treatment=treatment.tolist(), seed=42)
    run("heterogeneous_treatment_effects", x=x.tolist(), y=outcome.tolist(), treatment=treatment.tolist(), seed=42)
    sem_data = np.column_stack([x[:, 0], x[:, 1], 0.7 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(0, 0.2, n)])
    run(
        "structural_equation_model",
        variable_names=["x1", "x2", "y"],
        data=sem_data.tolist(),
        regressions=[{"dependent": "y", "predictors": ["x1", "x2"]}],
    )
    characteristics = [[1.0, 0.2], [1.2, 0.3], [0.8, 0.6], [1.1, 0.5], [0.9, 0.4], [1.3, 0.7]]
    run(
        "blp_demand_instruments",
        characteristics=characteristics,
        market_ids=["m1", "m1", "m1", "m2", "m2", "m2"],
        firm_ids=["a", "a", "b", "a", "c", "c"],
    )
    return [
        "high_dimensional_fixed_effects",
        "double_machine_learning",
        "heterogeneous_treatment_effects",
        "structural_equation_model",
        "blp_demand_instruments",
    ]


def forecasting() -> list[str]:
    rng = np.random.default_rng(1202)
    t = np.arange(48)
    series = np.vstack([
        10 + 0.2 * t + np.sin(t * 2 * np.pi / 12),
        5 + 0.1 * t + 0.5 * np.cos(t * 2 * np.pi / 12),
    ])
    run("scalable_statistical_forecast", series=series.tolist(), horizon=4, model="naive", season_length=12)
    summing = np.asarray([[1, 1], [1, 0], [0, 1]], dtype=float)
    base = np.asarray([[18, 19, 20], [10, 11, 12], [8, 8, 8]], dtype=float)
    result = run("hierarchical_forecast_reconciliation", summing_matrix=summing.tolist(), base_forecasts=base.tolist())
    assert result["maximum_coherence_error"] < 1e-9
    returns = rng.normal(0, 0.01, 400)
    run("garch_volatility", returns=returns.tolist(), horizon=3)
    features = rng.normal(size=(100, 3))
    features[-3:] += 8
    run("anomaly_detection", features=features.tolist(), contamination=0.05)
    values = rng.gumbel(loc=10, scale=2, size=1_200)
    run("extreme_value_analysis", values=values.tolist(), block_days=30, return_period=10)
    observations = rng.normal(size=(3, 20))
    forecasts = observations + rng.normal(0, 0.2, observations.shape)
    run("probabilistic_forecast_verification", observations=observations.tolist(), forecasts=forecasts.tolist())
    return [
        "scalable_statistical_forecast",
        "hierarchical_forecast_reconciliation",
        "garch_volatility",
        "anomaly_detection",
        "extreme_value_analysis",
        "probabilistic_forecast_verification",
    ]


def decision() -> list[str]:
    run(
        "deep_uncertainty_exploration",
        parameters=[
            {"name": "growth", "minimum": -1.0, "maximum": 3.0, "coefficient": 2.0},
            {"name": "cost", "minimum": 0.0, "maximum": 4.0, "coefficient": -1.0},
        ],
        scenarios=20,
        seed=42,
    )
    run(
        "comprehensive_mcda",
        decision_matrix=[[8, 3, 7], [6, 2, 9], [7, 4, 8]],
        weights=[0.4, 0.3, 0.3],
        criteria_types=[1, -1, 1],
        method="topsis",
    )
    result = run(
        "matrix_game_equilibrium",
        row_payoffs=[[1, -1], [-1, 1]],
        column_payoffs=[[-1, 1], [1, -1]],
    )
    assert result["equilibrium_count"] >= 1
    return ["deep_uncertainty_exploration", "comprehensive_mcda", "matrix_game_equilibrium"]


def spatial() -> list[str]:
    run(
        "geospatial_join",
        points=[{"id": "p1", "x": 0.5, "y": 0.5}, {"id": "p2", "x": 2.0, "y": 2.0}],
        regions=[{"id": "r1", "minx": 0, "miny": 0, "maxx": 1, "maxy": 1}],
        crs="EPSG:3857",
    )
    coordinates = []
    x = []
    y = []
    for i in range(5):
        for j in range(5):
            coordinates.append([float(i), float(j)])
            x.append([float(i + j)])
            y.append(1.0 + 0.8 * (i + j) + 0.05 * i * j)
    run("geographically_weighted_regression", coordinates=coordinates, x=x, y=y, bandwidth=10.0)
    run(
        "urban_morphology_metrics",
        buildings=[
            {"id": "b1", "minx": 0, "miny": 0, "maxx": 10, "maxy": 10},
            {"id": "b2", "minx": 20, "miny": 0, "maxx": 35, "maxy": 8},
        ],
        crs="EPSG:3857",
    )
    grid_n = 5
    neighbors = {}
    coords = [(i, j) for i in range(grid_n) for j in range(grid_n)]
    for idx, (i, j) in enumerate(coords):
        rows = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_n and 0 <= nj < grid_n:
                rows.append(ni * grid_n + nj)
        neighbors[str(idx)] = rows
    sx = [[float(i + j)] for i, j in coords]
    sy = [1.0 + 0.4 * (i + j) for i, j in coords]
    run("spatial_lag_regression", x=sx, y=sy, neighbors=neighbors)
    run(
        "facility_location",
        cost_matrix=[[1, 5, 7], [2, 3, 6], [8, 2, 1], [6, 4, 2]],
        demand_weights=[10, 20, 15, 5],
        p_facilities=2,
    )
    run(
        "trajectory_analysis",
        points=[
            {"trajectory_id": "t1", "timestamp": "2026-01-01T00:00:00", "x": 0, "y": 0},
            {"trajectory_id": "t1", "timestamp": "2026-01-01T00:10:00", "x": 100, "y": 0},
            {"trajectory_id": "t1", "timestamp": "2026-01-01T00:20:00", "x": 100, "y": 100},
        ],
        crs="EPSG:3857",
    )
    run(
        "spatial_segregation",
        areas=[
            {"id": "a", "group_population": 80, "total_population": 100, "minx": 0, "miny": 0, "maxx": 1, "maxy": 1},
            {"id": "b", "group_population": 20, "total_population": 100, "minx": 1, "miny": 0, "maxx": 2, "maxy": 1},
            {"id": "c", "group_population": 50, "total_population": 100, "minx": 2, "miny": 0, "maxx": 3, "maxy": 1},
        ],
        crs="EPSG:3857",
    )
    return [
        "geospatial_join",
        "geographically_weighted_regression",
        "urban_morphology_metrics",
        "spatial_lag_regression",
        "facility_location",
        "trajectory_analysis",
        "spatial_segregation",
    ]


def energy() -> list[str]:
    run(
        "energy_system_dispatch",
        load=[40, 50, 45],
        generators=[
            {"name": "base", "capacity": 40, "marginal_cost": 10, "availability": [1, 1, 1]},
            {"name": "peak", "capacity": 30, "marginal_cost": 30, "availability": [1, 1, 1]},
        ],
    )
    run("power_flow_analysis", load_mw=1.0, load_mvar=0.2, line_length_km=2.0, voltage_kv=20.0)
    return ["energy_system_dispatch", "power_flow_analysis"]


def climate_health() -> list[str]:
    run("water_network_resilience", demand_m3s=0.002, duration_hours=4, reservoir_head=80.0)
    temperatures = (30 + 8 * np.sin(np.arange(365) * 2 * np.pi / 365)).tolist()
    run("climate_threshold_index", daily_max_temperature_c=temperatures, threshold_c=35.0)
    run(
        "epidemic_scenario",
        population=300,
        duration_days=30,
        beta=0.04,
        infectious_duration=6,
        initial_prevalence=0.02,
        seed=42,
    )
    return ["water_network_resilience", "climate_threshold_index", "epidemic_scenario"]


def finance() -> list[str]:
    run(
        "european_option_pricing",
        option_type="call",
        spot=100,
        strike=100,
        volatility=0.2,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        maturity_days=365,
    )
    rng = np.random.default_rng(1404)
    correlated = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=200)
    from scipy.stats import norm
    uniforms = norm.cdf(correlated)
    run("copula_dependence_fit", uniform_data=uniforms.tolist())
    return ["european_option_pricing", "copula_dependence_fit"]


def knowledge() -> list[str]:
    run(
        "deterministic_record_linkage",
        records=[
            {"unique_id": 1, "city": "Fuzhou", "name": "Zhang Jie"},
            {"unique_id": 2, "city": "Fuzhou", "name": "Zhang Jie"},
            {"unique_id": 3, "city": "Xiamen", "name": "Li Ming"},
        ],
        id_column="unique_id",
        exact_fields=["city"],
        fuzzy_fields=["name"],
    )
    run("fuzzy_entity_matching", queries=["Fuzhou Baolong"], choices=["Fuzhou Baolong Plaza", "Xiamen Baolong", "Fuzhou Wanda"], limit=2)
    data_turtle = """
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:a ex:age 20 .
    """
    shapes_turtle = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:AgeShape a sh:NodeShape ;
          sh:targetSubjectsOf ex:age ;
          sh:property [ sh:path ex:age ; sh:datatype xsd:integer ; sh:minInclusive 0 ] .
    """
    result = run("shacl_graph_validation", data_turtle=data_turtle, shapes_turtle=shapes_turtle)
    assert result["conforms"] is True
    run("minhash_similarity", documents=[["a", "b", "c"], ["a", "b", "d"], ["x", "y"]], permutations=64)
    return ["deterministic_record_linkage", "fuzzy_entity_matching", "shacl_graph_validation", "minhash_similarity"]


def engineering() -> list[str]:
    run("control_system_response", numerator=[1.0], denominator=[1.0, 2.0, 1.0], duration=10, points=200)
    run("reliability_weibull_fit", failures=[5, 7, 9, 12, 15, 18, 20, 22], right_censored=[25, 30])
    run(
        "multi_echelon_inventory",
        echelon_holding_cost=[2, 2, 3],
        lead_time=[2, 1, 1],
        stockout_cost=37.12,
        demand_mean=5,
        demand_standard_deviation=1,
    )
    run(
        "queueing_network_simulation",
        arrival_rates=[0.2],
        service_rates=[0.5],
        servers=[1],
        routing=[[0.0]],
        duration=200,
        seed=42,
    )
    run(
        "job_shop_schedule",
        jobs=[
            [{"machine": 0, "duration": 2}, {"machine": 1, "duration": 1}],
            [{"machine": 1, "duration": 2}, {"machine": 0, "duration": 1}],
        ],
        time_limit_seconds=5,
    )
    return [
        "control_system_response",
        "reliability_weibull_fit",
        "multi_echelon_inventory",
        "queueing_network_simulation",
        "job_shop_schedule",
    ]


def assurance() -> list[str]:
    y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    groups = ["a"] * 6 + ["b"] * 6
    run("fairness_metric_audit", y_true=y_true, y_pred=y_pred, sensitive_features=groups)
    labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    probabilities = []
    for index, label in enumerate(labels):
        if index in {3, 14}:
            probabilities.append([0.9, 0.1] if label == 1 else [0.1, 0.9])
        else:
            probabilities.append([0.85, 0.15] if label == 0 else [0.15, 0.85])
    run("label_issue_detection", labels=labels, predicted_probabilities=probabilities)
    rng = np.random.default_rng(1505)
    x = rng.normal(size=(80, 3))
    y = 1 + x @ np.asarray([2.0, -1.0, 0.5]) + rng.normal(0, 0.05, 80)
    run("linear_model_explanation", x=x.tolist(), y=y.tolist(), explain_rows=10)
    run("synthetic_tabular_generation", column_names=["a", "b", "c"], data=x.tolist(), rows=20, seed=42)
    return ["fairness_metric_audit", "label_issue_detection", "linear_model_explanation", "synthetic_tabular_generation"]


VALIDATORS = {
    "economics": economics,
    "forecasting": forecasting,
    "decision": decision,
    "spatial": spatial,
    "energy": energy,
    "climate-health": climate_health,
    "finance": finance,
    "knowledge": knowledge,
    "engineering": engineering,
    "assurance": assurance,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=sorted(VALIDATORS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed = VALIDATORS[args.pack]()
    result = {
        "status": "PASS",
        "pack": args.pack,
        "completed_modes": completed,
        "mode_count": len(completed),
        "network_used": False,
        "external_data_fetches": 0,
        "model_calls": 0,
        "brokerage_execution": False,
        "arbitrary_code_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
