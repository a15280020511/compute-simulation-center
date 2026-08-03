#!/usr/bin/env python3
"""Execute one fixed offline fixture for each uncertainty/factor/accuracy mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty_factor_accuracy_operations import HANDLERS


def fixture(mode: str) -> dict[str, Any]:
    rng = np.random.default_rng(101)
    if mode == "joint_random_sample":
        return {
            "variables": [
                {"name": "demand", "distribution": "lognormal", "parameters": {"mu": 4.0, "sigma": 0.2}},
                {"name": "loss", "distribution": "student_t", "parameters": {"degrees_of_freedom": 5.0, "location": 0.0, "scale": 1.0}},
                {"name": "events", "distribution": "zero_inflated_poisson", "parameters": {"rate": 3.0, "zero_probability": 0.25}},
            ],
            "sample_count": 1000,
            "seed": 11,
            "dependence": {
                "method": "t_copula",
                "degrees_of_freedom": 6.0,
                "correlation_matrix": [[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]],
            },
        }
    if mode == "distribution_fit_select":
        return {
            "observations": rng.normal(2.0, 0.7, size=400).tolist(),
            "candidates": ["normal", "student_t", "gev"],
        }
    if mode == "variable_role_validate":
        return {
            "variables": [
                {"name": "income", "role": "exogenous", "dependencies": []},
                {"name": "policy", "role": "treatment", "dependencies": ["income"], "manipulable": True},
                {"name": "mediator", "role": "mediator", "dependencies": ["policy"]},
                {"name": "outcome", "role": "outcome", "dependencies": ["income", "policy", "mediator"]},
            ]
        }
    if mode == "probabilistic_accuracy":
        return {
            "actual": [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            "probabilities": [0.05, 0.15, 0.85, 0.75, 0.95, 0.35, 0.65, 0.25, 0.55, 0.45],
            "bins": 5,
        }
    if mode == "forecast_accuracy":
        actual = np.arange(1.0, 31.0)
        predicted = actual + np.sin(actual) * 0.2
        return {
            "actual": actual.tolist(),
            "predicted": predicted.tolist(),
            "baseline_predicted": (actual + 2.0).tolist(),
            "lower": (predicted - 1.0).tolist(),
            "upper": (predicted + 1.0).tolist(),
            "interval_alpha": 0.1,
            "quantile_predictions": {
                "0.1": (predicted - 0.8).tolist(),
                "0.5": predicted.tolist(),
                "0.9": (predicted + 0.8).tolist(),
            },
        }
    if mode == "bayesian_linear_calibration":
        x = np.arange(60, dtype=float)
        return {
            "features": x[:, None].tolist(),
            "observations": (2.0 + 3.0 * x).tolist(),
            "prediction_features": [[61.0], [62.0]],
        }
    if mode == "reliability_analysis":
        samples = rng.normal(0.5, 1.0, size=2000)
        return {
            "method": "monte_carlo",
            "limit_state_values": samples.tolist(),
            "factors": np.column_stack([samples, samples**2]).tolist(),
            "factor_names": ["linear", "quadratic"],
        }
    if mode == "factor_information_analysis":
        periods, assets = 20, 30
        signal = rng.normal(size=(periods, assets))
        noise = rng.normal(size=(periods, assets))
        returns = 0.35 * signal + 0.05 * noise
        return {
            "forward_returns": returns.tolist(),
            "factors": {"signal": signal.tolist(), "noise": noise.tolist()},
            "quantiles": 5,
            "regimes": ["calm" if index < 10 else "stress" for index in range(periods)],
        }
    if mode == "factor_selection_diagnostics":
        matrix = rng.normal(size=(400, 6))
        matrix[:, 5] = matrix[:, 0] * 0.99 + rng.normal(scale=0.01, size=400)
        target = 1.2 * matrix[:, 0] - 0.8 * matrix[:, 1] + rng.normal(scale=0.4, size=400)
        return {
            "factor_matrix": matrix.tolist(),
            "target": target.tolist(),
            "factor_names": ["value", "quality", "size", "momentum", "liquidity", "value_clone"],
            "lasso_penalty": 10.0,
        }
    if mode == "factor_overfit_diagnostics":
        returns = rng.normal(0.0, 0.01, size=(120, 10))
        returns[:, 0] += 0.0008
        return {
            "strategy_returns": returns.tolist(),
            "blocks": 6,
            "bootstraps": 100,
            "seed": 13,
        }
    if mode == "cross_validation_plan":
        return {
            "rows": 100,
            "strategy": "rolling",
            "splits": 4,
            "minimum_train_size": 40,
            "test_size": 10,
        }
    raise ValueError(f"unsupported fixture mode: {mode}")


def validate(mode: str) -> dict[str, Any]:
    handler = HANDLERS[mode]
    first = handler(fixture(mode))
    second = handler(fixture(mode))
    if first != second:
        raise AssertionError(f"{mode} is not deterministic for its fixed fixture")
    if first.get("network_used") is not False:
        raise AssertionError("network_used must be false")
    if first.get("model_calls") != 0:
        raise AssertionError("model_calls must equal zero")
    if first.get("brokerage_execution") is not False:
        raise AssertionError("brokerage_execution must be false")
    if first.get("arbitrary_code_allowed") is not False:
        raise AssertionError("arbitrary_code_allowed must be false")
    return {
        "schema_version": "uncertainty-factor-accuracy-validation-v1",
        "status": "PASS",
        "mode": mode,
        "network_used": False,
        "model_calls": 0,
        "brokerage_execution": False,
        "arbitrary_code_used": False,
        "deterministic_fixture": True,
        "result": first,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(HANDLERS))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    receipt = validate(arguments.mode)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
