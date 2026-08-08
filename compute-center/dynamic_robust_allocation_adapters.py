#!/usr/bin/env python3
"""Deterministic adapters for the dynamic robust-allocation family."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pipeline_adapters import ADAPTERS, PipelineAdapterError


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


def _nonnegative(value: Any, name: str, default: float) -> float:
    result = _finite(default if value is None else value, name)
    if result < 0:
        raise PipelineAdapterError(f"{name} must be non-negative")
    return result


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("robust_allocation_context")
    return {} if raw is None else _mapping(raw, "inputs.robust_allocation_context")


def _matrix(initial_inputs: Mapping[str, Any]) -> list[list[float]]:
    rows = _sequence(initial_inputs.get("scenario_returns"), "inputs.scenario_returns")
    if not 2 <= len(rows) <= 500:
        raise PipelineAdapterError("scenario_returns must contain 2 to 500 rows")
    converted: list[list[float]] = []
    width: int | None = None
    for i, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"inputs.scenario_returns[{i}]")
        if width is None:
            width = len(row)
        if len(row) != width or width is None or not 2 <= width <= 50:
            raise PipelineAdapterError("scenario_returns must be rectangular with 2 to 50 columns")
        converted.append([_finite(value, f"scenario_returns[{i}][]") for value in row])
    return converted


def allocation_ticket_to_rsome(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "rsome_robust_allocation":
        raise PipelineAdapterError("robust-allocation family requires rsome_robust_allocation entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "robust_allocation_context"}


def allocation_to_ortools_crosscheck(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage_results, stage
    matrix = _matrix(initial_inputs)
    asset_count = len(matrix[0])
    flat = [value for row in matrix for value in row]
    floor_lower = min(flat)
    floor_upper = max(flat)
    variables = [
        {
            "name": f"w_{index}",
            "type": "continuous",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "objective_coefficient": 0.0,
        }
        for index in range(asset_count)
    ]
    variables.append(
        {
            "name": "floor",
            "type": "continuous",
            "lower_bound": floor_lower,
            "upper_bound": floor_upper,
            "objective_coefficient": 1.0,
        }
    )
    constraints = [
        {
            "coefficients": {f"w_{index}": 1.0 for index in range(asset_count)},
            "relation": "==",
            "rhs": 1.0,
        }
    ]
    for row in matrix:
        coefficients = {f"w_{index}": float(value) for index, value in enumerate(row)}
        coefficients["floor"] = -1.0
        constraints.append({"coefficients": coefficients, "relation": ">=", "rhs": 0.0})
    return {
        "mode": "mixed_integer_optimization",
        "variables": variables,
        "constraints": constraints,
        "maximize": True,
        "time_limit_seconds": 20,
    }


def allocation_crosscheck_to_objective_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("rsome_robust_allocation"), "stage_results.rsome_robust_allocation")
    cross = _mapping(stage_results.get("ortools_maximin_crosscheck"), "stage_results.ortools_maximin_crosscheck")
    context = _context(initial_inputs)
    tolerance = _nonnegative(context.get("objective_consistency_tolerance"), "robust_allocation_context.objective_consistency_tolerance", 1e-8)
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "rsome-ortools-worst-case-objective-consistency",
                "observed": _finite(cross.get("objective_value"), "ortools objective_value"),
                "benchmark": _finite(primary.get("worst_case_return"), "rsome worst_case_return"),
                "tolerance": tolerance,
                "direction": "absolute",
            }
        ],
    }


def allocation_to_feasibility_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("rsome_robust_allocation"), "stage_results.rsome_robust_allocation")
    matrix = _matrix(initial_inputs)
    raw_weights = _sequence(primary.get("weights"), "rsome weights")
    if len(raw_weights) != len(matrix[0]):
        raise PipelineAdapterError("RSOME weight count does not match scenario columns")
    weights = [_finite(value, "rsome weight") for value in raw_weights]
    context = _context(initial_inputs)
    tolerance = _nonnegative(context.get("feasibility_tolerance"), "robust_allocation_context.feasibility_tolerance", 1e-8)
    realized = [sum(value * weight for value, weight in zip(row, weights, strict=True)) for row in matrix]
    recomputed_worst = min(realized)
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "allocation-weight-sum",
                "observed": sum(weights),
                "benchmark": 1.0,
                "tolerance": tolerance,
                "direction": "absolute",
            },
            {
                "name": "allocation-minimum-weight",
                "observed": min(weights),
                "benchmark": 0.0,
                "tolerance": tolerance,
                "direction": "minimum",
            },
            {
                "name": "allocation-recomputed-worst-case",
                "observed": recomputed_worst,
                "benchmark": _finite(primary.get("worst_case_return"), "rsome worst_case_return"),
                "tolerance": tolerance,
                "direction": "absolute",
            },
        ],
    }


def allocation_to_target_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("rsome_robust_allocation"), "stage_results.rsome_robust_allocation")
    context = _context(initial_inputs)
    tolerance = _nonnegative(context.get("allocation_target_tolerance"), "robust_allocation_context.allocation_target_tolerance", 0.0)
    weights = [_finite(value, "rsome weight") for value in _sequence(primary.get("weights"), "rsome weights")]
    specs = (
        ("minimum_worst_case_return", _finite(primary.get("worst_case_return"), "worst_case_return"), "minimum"),
        ("minimum_mean_return", _finite(primary.get("mean_return"), "mean_return"), "minimum"),
        ("maximum_single_asset_weight", max(weights), "maximum"),
    )
    candidates = []
    for name, observed, direction in specs:
        if name not in context:
            continue
        candidates.append(
            {
                "name": f"allocation-target-{name}",
                "observed": observed,
                "benchmark": _finite(context[name], f"robust_allocation_context.{name}"),
                "tolerance": tolerance,
                "direction": direction,
            }
        )
    if not candidates:
        raise PipelineAdapterError("allocation_target_audit requires at least one explicit target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_robust_allocation_adapters() -> None:
    adapters = {
        "allocation_ticket_to_rsome": allocation_ticket_to_rsome,
        "allocation_to_ortools_crosscheck": allocation_to_ortools_crosscheck,
        "allocation_crosscheck_to_objective_audit": allocation_crosscheck_to_objective_audit,
        "allocation_to_feasibility_audit": allocation_to_feasibility_audit,
        "allocation_to_target_audit": allocation_to_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"robust-allocation adapter name collision: {name}")
        ADAPTERS[name] = handler
