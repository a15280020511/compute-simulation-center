#!/usr/bin/env python3
"""Install-time and numerical smoke validation for each isolated sector-model mode."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sector_model_operations import MODES, sector_model_analysis


def fixtures() -> dict[str, dict[str, Any]]:
    x = [[float(i), float((i % 5) - 2)] for i in range(40)]
    d = [float(i % 2) for i in range(40)]
    y = [2.0 * d[i] + 0.3 * x[i][0] - 0.2 * x[i][1] + ((i % 3) - 1) * 0.01 for i in range(40)]
    return {
        "doubleml_plr": {"mode": "doubleml_plr", "x": x, "y": y, "treatment": d, "folds": 2, "seed": 7},
        "quantecon_markov_chain": {"mode": "quantecon_markov_chain", "transition_matrix": [[0.9, 0.1], [0.2, 0.8]], "initial_distribution": [1.0, 0.0], "steps": 50},
        "nash_bimatrix_equilibria": {"mode": "nash_bimatrix_equilibria", "row_payoffs": [[3, 0], [5, 1]], "column_payoffs": [[3, 5], [0, 1]]},
        "ema_robust_regret": {"mode": "ema_robust_regret", "alternative_names": ["A", "B", "C"], "outcomes": [[8, 6, 7, 5], [7, 7, 6, 7], [9, 4, 8, 4]], "maximize": True},
        "pypsa_linear_power_flow": {"mode": "pypsa_linear_power_flow", "load_mw": 50.0, "voltage_kv": 110.0, "line_reactance": 0.1, "line_resistance": 0.01},
        "pandapower_ac_power_flow": {"mode": "pandapower_ac_power_flow", "load_mw": 2.0, "load_mvar": 0.4, "voltage_kv": 20.0, "line_length_km": 1.0},
        "wntr_hydraulic_snapshot": {"mode": "wntr_hydraulic_snapshot", "reservoir_head_m": 100.0, "junction_elevation_m": 10.0, "demand_m3_s": 0.01, "pipe_length_m": 1000.0, "pipe_diameter_m": 0.3, "pipe_roughness": 100.0},
        "pywr_resource_allocation": {"mode": "pywr_resource_allocation", "supply_capacity": 10.0, "demand_capacity": 6.0, "supply_cost": 1.0, "demand_value": 5.0},
        "gstools_random_field": {"mode": "gstools_random_field", "nx": 8, "ny": 6, "seed": 11, "variance": 1.0, "length_scale": 3.0, "mean": 0.0},
        "pykrige_interpolation": {"mode": "pykrige_interpolation", "x": [0, 1, 0, 1], "y": [0, 0, 1, 1], "z": [1, 2, 2, 3], "predict_x": [0.5], "predict_y": [0.5], "variogram_model": "linear"},
        "brightway_matrix_lca": {"mode": "brightway_matrix_lca", "technology_matrix": [[1.0, -0.2], [0.0, 1.0]], "biosphere_matrix": [[1.0, 0.5], [0.1, 2.0]], "demand": [1.0, 0.0], "characterization_factors": [2.0, 3.0]},
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
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    case_map = fixtures()
    if set(case_map) != MODES:
        raise AssertionError(f"fixture mismatch: missing={sorted(MODES-set(case_map))} extra={sorted(set(case_map)-MODES)}")
    result = sector_model_analysis(case_map[args.mode])
    assert result["mode"] == args.mode
    assert result["network_used"] is False
    assert result["arbitrary_code_used"] is False
    assert result["maturity"] == "controlled-preview"
    assert finite_tree(result)
    payload = {"status": "PASS", "mode": args.mode, "result": result}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
