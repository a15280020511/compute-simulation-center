#!/usr/bin/env python3
"""Execute one fixed offline fixture for each institutional expansion mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from institutional_expansion_operations import HANDLERS


def _causal_data(seed: int = 7, rows: int = 240) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    y = 0.8 * x + rng.normal(scale=0.35, size=rows)
    z = -0.5 * y + rng.normal(scale=0.35, size=rows)
    return np.column_stack([x, y, z]).tolist()


def fixture(mode: str) -> dict:
    rng = np.random.default_rng(42)
    if mode in {"causal_pc_discovery", "causal_fci_discovery"}:
        return {"data": _causal_data(), "variable_names": ["x", "y", "z"], "alpha": 0.05}
    if mode == "causal_graph_stability_bootstrap":
        return {"data": _causal_data(), "variable_names": ["x", "y", "z"], "alpha": 0.05, "bootstraps": 8, "seed": 9, "stability_threshold": 0.5}
    if mode == "tigramite_pcmci_discovery":
        rows = 300
        x = rng.normal(size=rows)
        y = rng.normal(scale=0.4, size=rows)
        for index in range(1, rows):
            y[index] += 0.7 * x[index - 1]
        return {"data": np.column_stack([x, y]).tolist(), "variable_names": ["x", "y"], "tau_max": 2, "alpha": 0.05}
    if mode == "evidently_data_drift":
        return {
            "reference": rng.normal(size=(180, 2)).tolist(),
            "current": (rng.normal(size=(180, 2)) + np.array([1.2, 0.0])).tolist(),
            "variable_names": ["shifted", "stable"],
        }
    if mode == "river_adwin_drift":
        return {"values": ([0.0] * 100) + ([1.0] * 100), "delta": 0.01}
    if mode == "skfolio_walk_forward_portfolio":
        returns = rng.normal(loc=[0.0004, 0.0002, 0.0001], scale=[0.01, 0.006, 0.004], size=(180, 3))
        return {"returns": returns.tolist(), "train_size": 80, "test_size": 20, "transaction_cost_bps": 8.0}
    if mode == "black_litterman_allocation":
        return {
            "covariance": [[0.04, 0.01, 0.005], [0.01, 0.03, 0.004], [0.005, 0.004, 0.02]],
            "prior_returns": [0.07, 0.05, 0.035],
            "view_returns": [0.025],
            "pick_matrix": [[1.0, -1.0, 0.0]],
            "tau": 0.05,
        }
    if mode == "deflated_sharpe_gate":
        returns = rng.normal(loc=0.001, scale=0.01, size=400)
        return {"returns": returns.tolist(), "strategy_trials": 20, "trial_sharpe_standard_deviation": 0.15}
    if mode == "transaction_cost_capacity":
        returns = rng.normal(loc=0.0002, scale=0.01, size=(80, 2))
        weights = np.tile([0.6, 0.4], (80, 1))
        weights[40:, :] = [0.4, 0.6]
        return {
            "asset_returns": returns.tolist(),
            "weights": weights.tolist(),
            "capital": 1_000_000.0,
            "average_daily_volume": [50_000_000.0, 40_000_000.0],
            "commission_bps": 2.0,
            "spread_bps": 4.0,
            "impact_coefficient": 0.02,
        }
    raise KeyError(mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(HANDLERS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = HANDLERS[args.mode](fixture(args.mode))
    if result.get("mode") != args.mode:
        raise RuntimeError("handler returned the wrong mode")
    receipt = {
        "status": "PASS",
        "mode": args.mode,
        "network_used": bool(result.get("network_used", False)),
        "model_calls": 0,
        "brokerage_execution": bool(result.get("brokerage_execution", False)),
        "arbitrary_code_used": False,
        "result": result,
    }
    if receipt["network_used"] or receipt["brokerage_execution"]:
        raise RuntimeError("institutional expansion violated runtime boundaries")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
