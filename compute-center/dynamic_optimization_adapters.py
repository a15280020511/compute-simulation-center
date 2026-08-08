#!/usr/bin/env python3
"""Deterministic adapters for the dynamic linear-optimization capability family."""
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
        raise PipelineAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _problem(initial_inputs: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], bool]:
    if str(initial_inputs.get("mode") or "") != "mixed_integer_optimization":
        raise PipelineAdapterError("optimization family requires mixed_integer_optimization entry mode")
    variables = [_mapping(row, "inputs.variables[]") for row in _sequence(initial_inputs.get("variables"), "inputs.variables")]
    constraints = [_mapping(row, "inputs.constraints[]") for row in _sequence(initial_inputs.get("constraints", []), "inputs.constraints")]
    return variables, constraints, bool(initial_inputs.get("maximize", True))


def optimization_ticket_to_primary(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    variables, constraints, maximize = _problem(initial_inputs)
    result = {
        "mode": "mixed_integer_optimization",
        "variables": [dict(row) for row in variables],
        "constraints": [dict(row) for row in constraints],
        "maximize": maximize,
        "time_limit_seconds": int(initial_inputs.get("time_limit_seconds", 20)),
    }
    return result


def optimization_problem_to_highs_relaxation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    variables, constraints, maximize = _problem(initial_inputs)
    names: list[str] = []
    objective: list[float] = []
    upper_bounds: list[float | None] = []
    for index, row in enumerate(variables):
        name = str(row.get("name") or "")
        if not name or name in names:
            raise PipelineAdapterError("optimization variable names must be non-empty and unique")
        lower = _finite(row.get("lower_bound", row.get("lower", 0.0)), f"variables[{index}].lower_bound")
        if abs(lower) > 1e-12:
            raise PipelineAdapterError("independent LP relaxation currently requires zero lower bounds")
        kind = str(row.get("type") or "continuous")
        if kind not in {"continuous", "integer", "binary"}:
            raise PipelineAdapterError("unsupported optimization variable type")
        upper_raw = row.get("upper_bound", row.get("upper"))
        if kind == "binary" and upper_raw is None:
            upper_raw = 1.0
        upper = None if upper_raw is None else _finite(upper_raw, f"variables[{index}].upper_bound")
        if upper is not None and upper < 0:
            raise PipelineAdapterError("upper bounds must be non-negative for the LP relaxation adapter")
        names.append(name)
        objective.append(_finite(row.get("objective_coefficient", row.get("objective", 0.0)), f"variables[{index}].objective_coefficient"))
        upper_bounds.append(upper)

    name_to_index = {name: index for index, name in enumerate(names)}
    matrix: list[list[float]] = []
    bounds: list[float] = []

    def add_row(coefficients: Mapping[str, Any], rhs: float, multiplier: float = 1.0) -> None:
        unknown = set(str(name) for name in coefficients) - set(names)
        if unknown:
            raise PipelineAdapterError(f"constraint references unknown variables: {sorted(unknown)}")
        row = [0.0] * len(names)
        for raw_name, raw_value in coefficients.items():
            name = str(raw_name)
            row[name_to_index[name]] = multiplier * _finite(raw_value, f"constraint.{name}")
        matrix.append(row)
        bounds.append(multiplier * rhs)

    for index, raw in enumerate(constraints):
        coefficients = _mapping(raw.get("coefficients"), f"constraints[{index}].coefficients")
        rhs = _finite(raw.get("rhs"), f"constraints[{index}].rhs")
        relation = str(raw.get("relation") or "<=")
        if relation == "<=":
            add_row(coefficients, rhs)
        elif relation == ">=":
            add_row(coefficients, rhs, -1.0)
        elif relation == "==":
            add_row(coefficients, rhs)
            add_row(coefficients, rhs, -1.0)
        else:
            raise PipelineAdapterError("constraint relation must be <=, >=, or ==")

    for index, upper in enumerate(upper_bounds):
        if upper is None:
            continue
        row = [0.0] * len(names)
        row[index] = 1.0
        matrix.append(row)
        bounds.append(upper)

    if not matrix:
        # algebraic_resource_optimization requires a non-empty matrix; a very loose
        # deterministic row preserves the original feasible region for bounded tickets.
        matrix.append([0.0] * len(names))
        bounds.append(0.0)

    return {
        "mode": "algebraic_resource_optimization",
        "objective": objective,
        "constraint_matrix": matrix,
        "constraint_bounds": bounds,
        "maximize": maximize,
        "solver_engine": "highs",
    }


def optimization_primary_relaxation_to_bound_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("primary_optimization"), "stage_results.primary_optimization")
    relaxation = _mapping(stage_results.get("independent_relaxation"), "stage_results.independent_relaxation")
    variables, _, maximize = _problem(initial_inputs)
    continuous_only = all(str(row.get("type") or "continuous") == "continuous" for row in variables)
    context = initial_inputs.get("optimization_context")
    if context is None:
        context = {}
    context = _mapping(context, "inputs.optimization_context")
    tolerance = _finite(context.get("crosscheck_tolerance", 1e-7), "optimization_context.crosscheck_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("crosscheck_tolerance must be non-negative")
    direction = "absolute" if continuous_only else ("maximum" if maximize else "minimum")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "primary-vs-independent-lp-relaxation",
                "observed": _finite(primary.get("objective_value"), "primary.objective_value"),
                "benchmark": _finite(relaxation.get("objective_value"), "relaxation.objective_value"),
                "tolerance": tolerance,
                "direction": direction,
            }
        ],
    }


def optimization_primary_to_external_benchmark(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("primary_optimization"), "stage_results.primary_optimization")
    context = _mapping(initial_inputs.get("optimization_context"), "inputs.optimization_context")
    expected = _finite(context.get("external_objective_value"), "optimization_context.external_objective_value")
    tolerance = _finite(context.get("external_objective_tolerance"), "optimization_context.external_objective_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("external_objective_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "external-objective-benchmark",
                "observed": _finite(primary.get("objective_value"), "primary.objective_value"),
                "benchmark": expected,
                "tolerance": tolerance,
                "direction": "absolute",
            }
        ],
    }


def install_optimization_adapters() -> None:
    adapters = {
        "optimization_ticket_to_primary": optimization_ticket_to_primary,
        "optimization_problem_to_highs_relaxation": optimization_problem_to_highs_relaxation,
        "optimization_primary_relaxation_to_bound_audit": optimization_primary_relaxation_to_bound_audit,
        "optimization_primary_to_external_benchmark": optimization_primary_to_external_benchmark,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"optimization adapter name collision: {name}")
        ADAPTERS[name] = handler
