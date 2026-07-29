#!/usr/bin/env python3
"""Run deterministic numerical golden-oracle benchmarks."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_operations import markov_simulation  # noqa: E402
from compute_runner import (  # noqa: E402
    break_even_analysis,
    constrained_optimization,
    descriptive_statistics,
    monte_carlo,
    scenario_compare,
    sensitivity_analysis,
)
from professional_operations import bayesian_inference, econometric_analysis, gis_spatial_analysis  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "golden" / "manifest.json"


def close(actual: float, expected: float, tolerance: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"actual={actual} expected={expected} tolerance={tolerance}")


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []

    result = break_even_analysis({"fixed_cost": 1000, "unit_price": 20, "variable_cost": 12, "target_profit": 200})
    close(result["break_even_units"], 150.0, 1e-12)
    close(result["contribution_margin_per_unit"], 8.0, 1e-12)
    rows.append({"id": "golden-break-even-001", "status": "PASS"})

    result = descriptive_statistics({"data": [1, 2, 3, 4, 5, 8, 13]})
    close(result["mean"], 36 / 7, 1e-12)
    close(result["median"], 4.0, 1e-12)
    rows.append({"id": "golden-descriptive-001", "status": "PASS"})

    result = constrained_optimization({
        "objective": [3, 2],
        "maximize": True,
        "variable_names": ["x", "y"],
        "A_ub": [[1, 1], [1, 0], [0, 1]],
        "b_ub": [4, 2, 3],
        "bounds": [[0, None], [0, None]],
    })
    close(result["objective_value"], 10.0, 1e-9)
    close(result["solution"]["x"], 2.0, 1e-9)
    close(result["solution"]["y"], 2.0, 1e-9)
    rows.append({"id": "golden-linear-program-001", "status": "PASS"})

    result = sensitivity_analysis({
        "variables": [{"name": "a", "low": 1, "base": 2, "high": 5}, {"name": "b", "low": 4, "base": 5, "high": 6}],
        "model": {"intercept": 1, "coefficients": {"a": 3, "b": -1}},
    })
    close(result["baseline_score"], 2.0, 1e-12)
    if result["ranking"][0]["variable"] != "a":
        raise AssertionError("variable a must rank first")
    rows.append({"id": "golden-sensitivity-001", "status": "PASS"})

    result = scenario_compare({
        "model": {"intercept": 0, "coefficients": {"benefit": 1, "cost": -1}},
        "scenarios": [{"name": "A", "values": {"benefit": 10, "cost": 8}}, {"name": "B", "values": {"benefit": 9, "cost": 3}}],
    })
    if result["best_scenario"] != "B":
        raise AssertionError("scenario B must be best")
    rows.append({"id": "golden-scenario-001", "status": "PASS"})

    result = monte_carlo({
        "iterations": 1000,
        "seed": 20260729,
        "variables": [{"name": "x", "distribution": "constant", "value": 7}],
        "model": {"intercept": 3, "coefficients": {"x": 2}},
        "threshold": 18,
    })
    close(result["mean"], 17.0, 1e-12)
    close(result["standard_deviation"], 0.0, 1e-12)
    close(result["probability_below_threshold"], 1.0, 1e-12)
    rows.append({"id": "golden-monte-carlo-constant-001", "status": "PASS"})

    result = bayesian_inference({"mode": "beta_binomial", "prior_alpha": 1, "prior_beta": 1, "successes": 8, "trials": 10})
    close(result["posterior"]["alpha"], 9.0, 1e-12)
    close(result["posterior"]["beta"], 3.0, 1e-12)
    close(result["posterior"]["mean"], 0.75, 1e-12)
    rows.append({"id": "golden-beta-binomial-001", "status": "PASS"})

    result = econometric_analysis({"mode": "ols", "x": [[0], [1], [2], [3], [4]], "y": [1, 3, 5, 7, 9], "covariance_type": "HC1"})
    coefficients = {row["name"]: row["estimate"] for row in result["coefficients"]}
    close(coefficients["intercept"], 1.0, 1e-9)
    close(coefficients["x1"], 2.0, 1e-9)
    close(result["r_squared"], 1.0, 1e-9)
    rows.append({"id": "golden-ols-001", "status": "PASS"})

    result = gis_spatial_analysis({"mode": "geodesic_distance_matrix", "points": [{"id": "a", "longitude": 0, "latitude": 0}, {"id": "b", "longitude": 1, "latitude": 0}]})
    close(result["distance_matrix"][0][1], 111319.49079327357, 0.2)
    rows.append({"id": "golden-geodesic-001", "status": "PASS"})

    result = markov_simulation({"transition_matrix": [[0.9, 0.1], [0.2, 0.8]], "initial_distribution": [1, 0], "steps": 1})
    close(result["final_distribution"][0], 0.9, 1e-12)
    close(result["final_distribution"][1], 0.1, 1e-12)
    rows.append({"id": "golden-markov-001", "status": "PASS"})

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {row["id"] for row in manifest["cases"]}
    observed = {row["id"] for row in rows}
    if expected != observed:
        raise AssertionError(f"manifest mismatch missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    return {"schema_version": "golden-benchmark-result-v1", "status": "PASS", "passed": len(rows), "failed": 0, "rows": rows}


if __name__ == "__main__":
    output = run()
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("golden-benchmark-result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
