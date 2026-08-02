#!/usr/bin/env python3
"""Numerical smoke and contract validation for globally discovered think-tank modes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from think_tank_global_operations import HANDLERS


def fixtures() -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(17)
    econ_x = rng.normal(size=(80, 2))
    econ_t = 0.4 * econ_x[:, 0] + rng.normal(scale=0.3, size=80)
    econ_y = 2.0 * econ_t + 0.5 * econ_x[:, 1] + rng.normal(scale=0.1, size=80)
    spatial_n = 30
    spatial_x = np.linspace(-1.0, 1.0, spatial_n).reshape(-1, 1)
    spatial_y = 1.0 + 2.0 * spatial_x[:, 0] + 0.05 * np.sin(np.arange(spatial_n))
    neighbors = {str(i): [(i - 1) % spatial_n, (i + 1) % spatial_n] for i in range(spatial_n)}
    train_x = np.linspace(-2.0, 2.0, 60).reshape(-1, 1)
    train_y = 3.0 * train_x[:, 0] + 0.2 * np.sin(np.arange(60))
    return {
        "openturns_reliability_probability": {
            "mean": 0.0, "standard_deviation": 1.0, "threshold": -1.6448536269514729, "tail": "lower"
        },
        "control_step_response": {
            "numerator": [1.0], "denominator": [1.0, 1.0], "time_end": 8.0, "points": 81
        },
        "pm4py_directly_follows": {
            "cases": [
                {"case_id": "a", "activities": ["start", "review", "done"]},
                {"case_id": "b", "activities": ["start", "done"]},
            ]
        },
        "smt_rbf_surrogate": {
            "train_x": np.linspace(0.0, 1.0, 8).reshape(-1, 1).tolist(),
            "train_y": np.sin(np.linspace(0.0, 1.0, 8) * np.pi).tolist(),
            "predict_x": [[0.25], [0.5], [0.75]],
            "d0": 0.2,
        },
        "econml_linear_dml": {
            "x": econ_x.tolist(), "treatment": econ_t.tolist(), "outcome": econ_y.tolist(), "folds": 3, "seed": 17
        },
        "spreg_spatial_lag": {
            "x": spatial_x.tolist(), "y": spatial_y.tolist(), "neighbors": neighbors
        },
        "arch_garch_forecast": {
            "returns": rng.normal(scale=1.0, size=250).tolist(), "p": 1, "q": 1, "horizon": 3
        },
        "mapie_conformal_interval": {
            "train_x": train_x.tolist(), "train_y": train_y.tolist(), "predict_x": [[0.0], [1.0]],
            "confidence": 0.9, "cv": 3, "seed": 5
        },
        "pydoe3_latin_hypercube": {"factors": 4, "samples": 16, "criterion": "maximin", "seed": 7},
        "lmfit_exponential_calibration": {
            "x": np.linspace(0.0, 5.0, 50).tolist(),
            "y": (1.5 * np.exp(-0.7 * np.linspace(0.0, 5.0, 50)) + 0.2).tolist(),
            "initial_amplitude": 1.0, "initial_decay": 0.5, "initial_offset": 0.0,
        },
        "skgstat_variogram": {
            "coordinates": [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]],
            "values": [1.0, 1.2, 1.1, 1.5, 2.0, 2.2], "model": "spherical", "n_lags": 3,
        },
        "rsome_robust_allocation": {
            "asset_names": ["a", "b", "c"],
            "scenario_returns": [[0.08, 0.03, -0.02], [-0.04, 0.07, 0.01], [0.02, -0.01, 0.06], [0.01, 0.02, 0.02]],
        },
        "aequilibrae_shortest_path": {
            "links": [
                {"a_node": 1, "b_node": 2, "cost": 1.0},
                {"a_node": 2, "b_node": 4, "cost": 1.0},
                {"a_node": 1, "b_node": 3, "cost": 1.5},
                {"a_node": 3, "b_node": 4, "cost": 2.0},
            ],
            "origin": 1, "destination": 4,
        },
        "epydemix_sir_simulation": {
            "population": 2_000, "initial_infected": 20, "transmission_rate": 0.25,
            "recovery_rate": 0.1, "days": 20, "simulations": 3, "seed": 11,
        },
        "pysd_stock_flow_scenario": {
            "initial_stock": 100.0, "constant_inflow": 10.0, "decay_rate": 0.05,
            "final_time": 10.0, "time_step": 0.5,
        },
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(HANDLERS), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cases = fixtures()
    if set(cases) != set(HANDLERS):
        raise AssertionError(
            f"fixture mismatch: missing={sorted(set(HANDLERS)-set(cases))} extra={sorted(set(cases)-set(HANDLERS))}"
        )
    result = HANDLERS[args.mode](cases[args.mode])
    if result.get("mode") != args.mode or not finite_tree(result):
        raise AssertionError("mode result contract or finite-value contract failed")
    payload = {
        "status": "PASS",
        "mode": args.mode,
        "network_used": False,
        "arbitrary_code_used": False,
        "result": result,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
