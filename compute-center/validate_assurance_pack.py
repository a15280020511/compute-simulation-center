#!/usr/bin/env python3
"""Execute deterministic truth fixtures for every assurance capability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assurance_operations import HANDLERS

CHECKS = [
    "code_verification",
    "numerical_benchmark",
    "input_data_quality",
    "assumption_register",
    "sensitivity_analysis",
    "uncertainty_analysis",
    "external_validation",
    "independent_review",
    "realized_outcome_feedback",
]

FIXTURES: dict[str, dict[str, Any]] = {
    "probabilistic_forecast_scoring": {
        "probabilities": [[0.8, 0.2], [0.25, 0.75], [0.6, 0.4]],
        "outcomes": [0, 1, 0],
    },
    "calibration_diagnostics": {
        "probabilities": [0.05, 0.2, 0.35, 0.7, 0.8, 0.95],
        "outcomes": [0, 0, 0, 1, 1, 1],
        "bins": 3,
    },
    "prediction_interval_validation": {
        "lower": [0.0, 1.0, 2.0, 3.0],
        "upper": [2.0, 3.0, 4.0, 5.0],
        "observed": [1.0, 2.0, 4.5, 4.0],
        "alpha": 0.1,
    },
    "realized_outcome_feedback": {
        "predicted": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "observed": [10.2, 10.8, 12.1, 12.6, 14.6, 16.0],
        "drift_ratio_threshold": 4.0,
    },
    "benchmark_comparison": {
        "candidates": [
            {"name": "truth_case", "observed": 1.01, "benchmark": 1.0, "tolerance": 0.02},
            {"name": "minimum_quality", "observed": 0.92, "benchmark": 0.9, "tolerance": 0.0, "direction": "minimum"},
        ]
    },
    "cross_model_agreement": {
        "model_names": ["engine_a", "engine_b", "engine_c"],
        "outputs": [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [0.9, 1.9, 3.2, 4.2]],
        "minimum_rank_correlation": 0.8,
    },
    "vva_acceptance_gate": {
        "checks": {name: "PASS" for name in CHECKS},
        "high_risk": True,
    },
    "bounded_linear_kalman_filter": {
        "transition_matrix": [[1.0]],
        "observation_matrix": [[1.0]],
        "process_covariance": [[0.05]],
        "observation_covariance": [[0.2]],
        "initial_covariance": [[1.0]],
        "initial_state": [0.0],
        "observations": [[1.0], [1.5], [2.0]],
    },
}


def validate(mode: str) -> dict[str, Any]:
    handler = HANDLERS.get(mode)
    if handler is None or mode not in FIXTURES:
        raise RuntimeError(f"unknown assurance mode: {mode}")
    result = handler(FIXTURES[mode])
    if result.get("mode") != mode:
        raise RuntimeError("mode receipt mismatch")
    for key, expected in {
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_used": False,
        "live_feed_used": False,
        "individual_or_target_tracking_allowed": False,
        "decision_support_only": True,
    }.items():
        if result.get(key) != expected:
            raise RuntimeError(f"governance receipt mismatch for {mode}: {key}")
    return {
        "status": "PASS",
        "mode": mode,
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_used": False,
        "live_feed_used": False,
        "individual_or_target_tracking_allowed": False,
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(HANDLERS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode:
        payload: dict[str, Any] = validate(args.mode)
    else:
        rows = [validate(mode) for mode in sorted(HANDLERS)]
        payload = {"status": "PASS", "mode_count": len(rows), "rows": rows}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
