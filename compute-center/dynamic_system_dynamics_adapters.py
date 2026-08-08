#!/usr/bin/env python3
"""Deterministic adapters for the dynamic system-dynamics capability family."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pipeline_adapters import ADAPTERS, PipelineAdapterError


MODES = {
    "stock_flow",
    "feedback_delay",
    "policy_switch",
    "coupled_capacity",
    "resource_depletion",
    "adoption_saturation",
}
ROBUSTNESS_PARAMETERS = {
    "feedback_delay": {"exogenous_input", "decay_rate", "feedback_gain"},
    "policy_switch": {"growth_rate_before", "growth_rate_after"},
    "coupled_capacity": {"demand_growth", "capacity_addition"},
    "resource_depletion": {"regeneration_rate", "extraction"},
    "adoption_saturation": {"innovation_rate", "imitation_rate"},
    "stock_flow": set(),
}


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


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    value = initial_inputs.get("system_dynamics_context")
    return {} if value is None else _mapping(value, "inputs.system_dynamics_context")


def _mode(initial_inputs: Mapping[str, Any]) -> str:
    mode = str(initial_inputs.get("mode") or "")
    if mode not in MODES:
        raise PipelineAdapterError("unsupported system-dynamics mode")
    return mode


def _target_descriptor(initial_inputs: Mapping[str, Any]) -> tuple[str, str | None]:
    mode = _mode(initial_inputs)
    context = _context(initial_inputs)
    if mode == "stock_flow":
        stocks = [_mapping(row, "inputs.stocks[]") for row in _sequence(initial_inputs.get("stocks"), "inputs.stocks")]
        requested = str(context.get("target_stock_name") or stocks[0].get("name") or "")
        names = [str(row.get("name") or "") for row in stocks]
        if requested not in names:
            raise PipelineAdapterError("target_stock_name must match an admitted stock")
        return "stock_final", requested
    defaults = {
        "feedback_delay": "final_state",
        "policy_switch": "final_state",
        "coupled_capacity": "final_backlog",
        "resource_depletion": "final_stock",
        "adoption_saturation": "final_adoption",
    }
    allowed = {
        "feedback_delay": {"final_state"},
        "policy_switch": {"final_state"},
        "coupled_capacity": {"final_demand", "final_capacity", "final_backlog"},
        "resource_depletion": {"final_stock"},
        "adoption_saturation": {"final_adoption"},
    }
    metric = str(context.get("target_metric") or defaults[mode])
    if metric not in allowed[mode]:
        raise PipelineAdapterError(f"target_metric is not admitted for mode {mode}")
    return metric, None


def _target_value(initial_inputs: Mapping[str, Any], result: Mapping[str, Any]) -> float:
    metric, stock_name = _target_descriptor(initial_inputs)
    if metric == "stock_final":
        rows = [_mapping(row, "result.stocks[]") for row in _sequence(result.get("stocks"), "result.stocks")]
        for row in rows:
            if str(row.get("name") or "") == stock_name:
                return _finite(row.get("final"), f"result.stocks[{stock_name}].final")
        raise PipelineAdapterError("target stock is missing from system-dynamics result")
    return _finite(result.get(metric), f"result.{metric}")


def _trajectory(initial_inputs: Mapping[str, Any], result: Mapping[str, Any]) -> list[float]:
    mode = _mode(initial_inputs)
    metric, stock_name = _target_descriptor(initial_inputs)
    rows = [_mapping(row, "result.history[]") for row in _sequence(result.get("history"), "result.history")]
    values: list[float] = []
    if mode == "stock_flow":
        stocks = [_mapping(row, "inputs.stocks[]") for row in _sequence(initial_inputs.get("stocks"), "inputs.stocks")]
        names = [str(row.get("name") or "") for row in stocks]
        index = names.index(str(stock_name))
        for row in rows:
            raw = _sequence(row.get("stocks"), "result.history[].stocks")
            if index >= len(raw):
                raise PipelineAdapterError("stock trajectory width changed across history")
            values.append(_finite(raw[index], "result.history[].stocks[]"))
        return values
    history_key = {
        "feedback_delay": "state",
        "policy_switch": "state",
        "coupled_capacity": {
            "final_demand": "demand",
            "final_capacity": "capacity",
            "final_backlog": "backlog",
        }[metric],
        "resource_depletion": "stock",
        "adoption_saturation": "adoption",
    }[mode]
    for row in rows:
        values.append(_finite(row.get(history_key), f"result.history[].{history_key}"))
    if not values:
        raise PipelineAdapterError("system-dynamics trajectory is empty")
    return values


def system_dynamics_ticket_to_primary(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    _mode(initial_inputs)
    return {key: value for key, value in initial_inputs.items() if key != "system_dynamics_context"}


def system_dynamics_primary_to_trajectory_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("primary_simulation"), "stage_results.primary_simulation")
    return {"data": _trajectory(initial_inputs, primary)}


def system_dynamics_ticket_to_robustness_simulation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    mode = _mode(initial_inputs)
    context = _context(initial_inputs)
    parameter = str(context.get("robustness_parameter") or "")
    if parameter not in ROBUSTNESS_PARAMETERS[mode]:
        raise PipelineAdapterError(f"robustness_parameter is not admitted for mode {mode}")
    fraction = _finite(context.get("perturbation_fraction"), "system_dynamics_context.perturbation_fraction")
    if not -0.5 <= fraction <= 1.0 or abs(fraction) < 1e-12:
        raise PipelineAdapterError("perturbation_fraction must be in [-0.5,1.0] and non-zero")
    base = _finite(initial_inputs.get(parameter), f"inputs.{parameter}")
    if abs(base) < 1e-15:
        raise PipelineAdapterError("robustness perturbation requires a non-zero baseline parameter")
    perturbed = base * (1.0 + fraction)
    derived = {key: value for key, value in initial_inputs.items() if key != "system_dynamics_context"}
    derived[parameter] = perturbed
    return derived


def system_dynamics_primary_robustness_to_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("primary_simulation"), "stage_results.primary_simulation")
    perturbed = _mapping(stage_results.get("robustness_simulation"), "stage_results.robustness_simulation")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("max_absolute_deviation"), "system_dynamics_context.max_absolute_deviation")
    if tolerance < 0:
        raise PipelineAdapterError("max_absolute_deviation must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "system-dynamics-robustness-bound",
            "observed": _target_value(initial_inputs, perturbed),
            "benchmark": _target_value(initial_inputs, primary),
            "tolerance": tolerance,
            "direction": "absolute",
        }],
    }


def system_dynamics_primary_to_external_benchmark(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("primary_simulation"), "stage_results.primary_simulation")
    context = _context(initial_inputs)
    benchmark = _finite(context.get("external_final_value"), "system_dynamics_context.external_final_value")
    tolerance = _finite(context.get("external_final_tolerance"), "system_dynamics_context.external_final_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("external_final_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "system-dynamics-external-final-benchmark",
            "observed": _target_value(initial_inputs, primary),
            "benchmark": benchmark,
            "tolerance": tolerance,
            "direction": "absolute",
        }],
    }


def install_system_dynamics_adapters() -> None:
    adapters = {
        "system_dynamics_ticket_to_primary": system_dynamics_ticket_to_primary,
        "system_dynamics_primary_to_trajectory_statistics": system_dynamics_primary_to_trajectory_statistics,
        "system_dynamics_ticket_to_robustness_simulation": system_dynamics_ticket_to_robustness_simulation,
        "system_dynamics_primary_robustness_to_audit": system_dynamics_primary_robustness_to_audit,
        "system_dynamics_primary_to_external_benchmark": system_dynamics_primary_to_external_benchmark,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"system-dynamics adapter name collision: {name}")
        ADAPTERS[name] = handler
