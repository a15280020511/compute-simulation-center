#!/usr/bin/env python3
"""Deterministic adapters for the state-estimation dynamic capability family."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from pipeline_adapters import ADAPTERS, PipelineAdapterError


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineAdapterError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineAdapterError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineAdapterError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _matrix(value: Any, name: str) -> list[list[float]]:
    rows = _sequence(value, name)
    if not rows:
        raise PipelineAdapterError(f"{name} must not be empty")
    converted: list[list[float]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"{name}[{row_index}]")
        if not row:
            raise PipelineAdapterError(f"{name}[{row_index}] must not be empty")
        values = [_finite(item, f"{name}[{row_index}][]") for item in row]
        width = len(values) if width is None else width
        if len(values) != width:
            raise PipelineAdapterError(f"{name} rows must have equal width")
        converted.append(values)
    return converted


def _state_result(stage_results: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _mapping(stage_results.get("state_estimation"), "stage results.state_estimation")
    if result.get("mode") != "bounded_linear_kalman_filter":
        raise PipelineAdapterError("state_estimation stage returned the wrong mode")
    if result.get("fixed_offline_generic_state_estimation") is not True:
        raise PipelineAdapterError("state_estimation stage did not preserve fixed offline semantics")
    return result


def state_ticket_to_kalman(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "bounded_linear_kalman_filter":
        raise PipelineAdapterError("state-estimation entry mode must be bounded_linear_kalman_filter")
    result = {"mode": "bounded_linear_kalman_filter"}
    for name in (
        "transition_matrix",
        "observation_matrix",
        "process_covariance",
        "observation_covariance",
        "initial_covariance",
        "initial_state",
        "observations",
    ):
        if name not in initial_inputs:
            raise PipelineAdapterError(f"state-estimation entry requires {name}")
        result[name] = _clone(initial_inputs[name])
    return result


def state_estimate_to_realized_feedback(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    estimation = _state_result(stage_results)
    observation_matrix = _matrix(initial_inputs.get("observation_matrix"), "ticket inputs.observation_matrix")
    if len(observation_matrix) != 1:
        raise PipelineAdapterError("realized feedback currently requires scalar observations")
    observations = _matrix(initial_inputs.get("observations"), "ticket inputs.observations")
    if len(observations) < 4 or any(len(row) != 1 for row in observations):
        raise PipelineAdapterError("realized feedback requires at least four scalar observations")
    states_raw = _sequence(estimation.get("filtered_states"), "state estimation.filtered_states")
    if len(states_raw) != len(observations):
        raise PipelineAdapterError("filtered state count must match observation count")
    h = observation_matrix[0]
    predicted: list[float] = []
    for index, raw_state in enumerate(states_raw):
        state = [_finite(item, f"filtered_states[{index}][]") for item in _sequence(raw_state, f"filtered_states[{index}]")]
        if len(state) != len(h):
            raise PipelineAdapterError("filtered state dimension does not match observation matrix")
        predicted.append(float(sum(weight * value for weight, value in zip(h, state, strict=True))))
    observed = [row[0] for row in observations]
    threshold = _finite(initial_inputs.get("drift_ratio_threshold", 1.5), "ticket inputs.drift_ratio_threshold")
    if threshold <= 0:
        raise PipelineAdapterError("drift_ratio_threshold must be positive")
    return {
        "mode": "realized_outcome_feedback",
        "predicted": predicted,
        "observed": observed,
        "drift_ratio_threshold": threshold,
    }


def state_estimate_to_benchmark(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    estimation = _state_result(stage_results)
    final_state = [
        _finite(item, "state estimation.final_state[]")
        for item in _sequence(estimation.get("final_state"), "state estimation.final_state")
    ]
    benchmark = [
        _finite(item, "ticket inputs.benchmark_state[]")
        for item in _sequence(initial_inputs.get("benchmark_state"), "ticket inputs.benchmark_state")
    ]
    if len(final_state) != len(benchmark):
        raise PipelineAdapterError("benchmark_state length must match final state dimension")
    raw_tolerance = initial_inputs.get("benchmark_tolerance")
    if isinstance(raw_tolerance, Sequence) and not isinstance(raw_tolerance, (str, bytes)):
        tolerance = [_finite(item, "ticket inputs.benchmark_tolerance[]") for item in raw_tolerance]
        if len(tolerance) != len(final_state):
            raise PipelineAdapterError("benchmark_tolerance array must match final state dimension")
    else:
        scalar = _finite(raw_tolerance, "ticket inputs.benchmark_tolerance")
        tolerance = [scalar] * len(final_state)
    if any(item < 0 for item in tolerance):
        raise PipelineAdapterError("benchmark_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": f"state_{index}",
                "observed": observed,
                "benchmark": expected,
                "tolerance": allowed,
                "direction": "absolute",
            }
            for index, (observed, expected, allowed) in enumerate(
                zip(final_state, benchmark, tolerance, strict=True)
            )
        ],
    }


STATE_ESTIMATION_ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "state_ticket_to_kalman": state_ticket_to_kalman,
    "state_estimate_to_realized_feedback": state_estimate_to_realized_feedback,
    "state_estimate_to_benchmark": state_estimate_to_benchmark,
}


def install_state_estimation_adapters() -> None:
    for name, handler in STATE_ESTIMATION_ADAPTERS.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"conflicting pipeline adapter registration: {name}")
        ADAPTERS[name] = handler
