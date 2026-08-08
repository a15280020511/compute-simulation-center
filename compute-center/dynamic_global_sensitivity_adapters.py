#!/usr/bin/env python3
"""Deterministic adapters and closed-form checks for global sensitivity."""
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


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("global_sensitivity_context")
    return {} if raw is None else _mapping(raw, "inputs.global_sensitivity_context")


def _linear_spec(initial_inputs: Mapping[str, Any]) -> tuple[list[tuple[str, float, float, float]], float]:
    parameters = _sequence(initial_inputs.get("parameters"), "inputs.parameters")
    model = _mapping(initial_inputs.get("model"), "inputs.model")
    linear = _mapping(model.get("linear", {}), "inputs.model.linear")
    quadratic = _mapping(model.get("quadratic", {}), "inputs.model.quadratic")
    interactions = _sequence(model.get("interactions", []), "inputs.model.interactions")
    if quadratic or interactions:
        raise PipelineAdapterError("dynamic global-sensitivity v1 admits only linear additive fixed models")
    intercept = _finite(model.get("intercept", 0.0), "inputs.model.intercept")
    rows: list[tuple[str, float, float, float]] = []
    names: set[str] = set()
    for index, raw in enumerate(parameters):
        parameter = _mapping(raw, f"inputs.parameters[{index}]")
        name = str(parameter.get("name") or "").strip()
        low = _finite(parameter.get("minimum"), f"parameter[{name}].minimum")
        high = _finite(parameter.get("maximum"), f"parameter[{name}].maximum")
        if not name or name in names or not low < high:
            raise PipelineAdapterError("parameter names must be unique and minimum lower than maximum")
        names.add(name)
        coefficient = _finite(linear.get(name, 0.0), f"inputs.model.linear[{name}]")
        rows.append((name, low, high, coefficient))
    unknown = set(str(name) for name in linear) - names
    if unknown:
        raise PipelineAdapterError(f"linear model references unknown parameters: {sorted(unknown)}")
    total_variance = sum((coefficient * (high - low)) ** 2 / 12.0 for _, low, high, coefficient in rows)
    if total_variance <= 1e-18:
        raise PipelineAdapterError("linear sensitivity model must have positive output variance")
    return rows, intercept


def _exact_indices(initial_inputs: Mapping[str, Any]) -> dict[str, float]:
    rows, _ = _linear_spec(initial_inputs)
    contributions = {
        name: (coefficient * (high - low)) ** 2 / 12.0
        for name, low, high, coefficient in rows
    }
    total = sum(contributions.values())
    return {name: value / total for name, value in contributions.items()}


def sensitivity_ticket_to_salib(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "sobol_sensitivity":
        raise PipelineAdapterError("global-sensitivity family requires sobol_sensitivity entry mode")
    _linear_spec(initial_inputs)
    return {key: value for key, value in initial_inputs.items() if key != "global_sensitivity_context"}


def sensitivity_to_exact_index_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("salib_sobol_sensitivity"), "stage_results.salib_sobol_sensitivity")
    ranking = _sequence(primary.get("ranking"), "sobol ranking")
    observed: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(ranking):
        row = _mapping(raw, f"ranking[{index}]")
        parameter = str(row.get("parameter") or "")
        if not parameter or parameter in observed:
            raise PipelineAdapterError("SALib ranking contains invalid parameter names")
        observed[parameter] = row
    exact = _exact_indices(initial_inputs)
    if set(observed) != set(exact):
        raise PipelineAdapterError("SALib ranking parameters do not match ticket parameters")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("index_consistency_tolerance", 0.03), "global_sensitivity_context.index_consistency_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("index_consistency_tolerance must be non-negative")
    candidates = []
    for parameter in sorted(exact):
        candidates.append({
            "name": f"sobol-first-order-{parameter}",
            "observed": _finite(observed[parameter].get("first_order"), f"ranking[{parameter}].first_order"),
            "benchmark": exact[parameter],
            "tolerance": tolerance,
            "direction": "absolute",
        })
        candidates.append({
            "name": f"sobol-total-order-{parameter}",
            "observed": _finite(observed[parameter].get("total_order"), f"ranking[{parameter}].total_order"),
            "benchmark": exact[parameter],
            "tolerance": tolerance,
            "direction": "absolute",
        })
    return {"mode": "benchmark_comparison", "candidates": candidates}


def sensitivity_to_exact_moment_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("salib_sobol_sensitivity"), "stage_results.salib_sobol_sensitivity")
    distribution = _mapping(primary.get("output_distribution"), "sobol output_distribution")
    rows, intercept = _linear_spec(initial_inputs)
    exact_mean = intercept + sum(coefficient * (low + high) / 2.0 for _, low, high, coefficient in rows)
    exact_variance = sum((coefficient * (high - low)) ** 2 / 12.0 for _, low, high, coefficient in rows)
    exact_std = math.sqrt(exact_variance)
    context = _context(initial_inputs)
    tolerance = _finite(context.get("moment_consistency_tolerance", 0.03), "global_sensitivity_context.moment_consistency_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("moment_consistency_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "sobol-output-mean-consistency",
                "observed": _finite(distribution.get("mean"), "output_distribution.mean"),
                "benchmark": exact_mean,
                "tolerance": tolerance,
                "direction": "absolute",
            },
            {
                "name": "sobol-output-standard-deviation-consistency",
                "observed": _finite(distribution.get("standard_deviation"), "output_distribution.standard_deviation"),
                "benchmark": exact_std,
                "tolerance": tolerance,
                "direction": "absolute",
            },
        ],
    }


def sensitivity_to_target_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("salib_sobol_sensitivity"), "stage_results.salib_sobol_sensitivity")
    ranking = _sequence(primary.get("ranking"), "sobol ranking")
    observed = {
        str(_mapping(row, "ranking row").get("parameter") or ""): _mapping(row, "ranking row")
        for row in ranking
    }
    context = _context(initial_inputs)
    targets = _mapping(context.get("minimum_total_order_by_parameter"), "global_sensitivity_context.minimum_total_order_by_parameter")
    tolerance = _finite(context.get("target_tolerance", 0.0), "global_sensitivity_context.target_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("target_tolerance must be non-negative")
    candidates = []
    for parameter, raw_target in sorted(targets.items()):
        name = str(parameter)
        if name not in observed:
            raise PipelineAdapterError(f"sensitivity target references unknown parameter: {name}")
        target = _finite(raw_target, f"minimum_total_order_by_parameter[{name}]")
        candidates.append({
            "name": f"sensitivity-minimum-total-order-{name}",
            "observed": _finite(observed[name].get("total_order"), f"ranking[{name}].total_order"),
            "benchmark": target,
            "tolerance": tolerance,
            "direction": "minimum",
        })
    if not candidates:
        raise PipelineAdapterError("sensitivity_target_audit requires at least one explicit target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_global_sensitivity_adapters() -> None:
    adapters = {
        "sensitivity_ticket_to_salib": sensitivity_ticket_to_salib,
        "sensitivity_to_exact_index_audit": sensitivity_to_exact_index_audit,
        "sensitivity_to_exact_moment_audit": sensitivity_to_exact_moment_audit,
        "sensitivity_to_target_audit": sensitivity_to_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"global-sensitivity adapter name collision: {name}")
        ADAPTERS[name] = handler
