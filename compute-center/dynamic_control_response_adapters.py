#!/usr/bin/env python3
"""Deterministic adapters for the dynamic control-response family."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

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


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("control_context")
    return {} if raw is None else _mapping(raw, "inputs.control_context")


def control_ticket_to_step_response(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "control_step_response":
        raise PipelineAdapterError("control-response family requires control_step_response entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "control_context"}


def control_response_to_tail_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("control_step_response"), "stage_results.control_step_response")
    response = [_finite(item, "control_step_response.response[]") for item in _sequence(primary.get("response"), "control_step_response.response")]
    context = _context(initial_inputs)
    fraction = _finite(context.get("tail_fraction", 0.2), "control_context.tail_fraction")
    if not 0.05 <= fraction <= 0.5:
        raise PipelineAdapterError("tail_fraction must be between 0.05 and 0.5")
    count = max(10, int(math.ceil(len(response) * fraction)))
    count = min(count, len(response))
    return {"data": response[-count:]}


def tail_statistics_to_stability_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    stats = _mapping(stage_results.get("tail_response_statistics"), "stage_results.tail_response_statistics")
    observed = _finite(stats.get("standard_deviation_population"), "tail_response_statistics.standard_deviation_population")
    context = _context(initial_inputs)
    threshold = _finite(context.get("maximum_tail_standard_deviation"), "control_context.maximum_tail_standard_deviation")
    tolerance = _finite(context.get("tail_standard_deviation_tolerance", 0.0), "control_context.tail_standard_deviation_tolerance")
    if threshold < 0 or tolerance < 0:
        raise PipelineAdapterError("tail stability threshold/tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "control-tail-standard-deviation",
            "observed": observed,
            "benchmark": threshold,
            "tolerance": tolerance,
            "direction": "maximum",
        }],
    }


def _independent_dc_gain(initial_inputs: Mapping[str, Any]) -> tuple[float, list[complex]]:
    numerator = np.asarray([_finite(item, "inputs.numerator[]") for item in _sequence(initial_inputs.get("numerator"), "inputs.numerator")], dtype=float)
    denominator = np.asarray([_finite(item, "inputs.denominator[]") for item in _sequence(initial_inputs.get("denominator"), "inputs.denominator")], dtype=float)
    if denominator.size < 2 or abs(float(denominator[0])) < 1e-15:
        raise PipelineAdapterError("denominator is invalid for independent pole analysis")
    poles = [complex(item) for item in np.roots(denominator)]
    if any(not (math.isfinite(item.real) and math.isfinite(item.imag)) for item in poles):
        raise PipelineAdapterError("independent pole calculation returned non-finite values")
    if any(item.real >= -1e-12 for item in poles):
        raise PipelineAdapterError("DC-gain consistency requires all continuous-time poles to have negative real part")
    if abs(float(denominator[-1])) < 1e-15:
        raise PipelineAdapterError("DC-gain consistency is undefined for a zero denominator constant term")
    return float(numerator[-1] / denominator[-1]), poles


def control_response_to_dc_gain_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("control_step_response"), "stage_results.control_step_response")
    observed = _finite(primary.get("final_value"), "control_step_response.final_value")
    dc_gain, _ = _independent_dc_gain(initial_inputs)
    context = _context(initial_inputs)
    tolerance = _finite(context.get("dc_gain_tolerance", 1e-3), "control_context.dc_gain_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("dc_gain_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "control-final-value-vs-independent-dc-gain",
            "observed": observed,
            "benchmark": dc_gain,
            "tolerance": tolerance,
            "direction": "absolute",
        }],
    }


def control_response_to_target_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("control_step_response"), "stage_results.control_step_response")
    context = _context(initial_inputs)
    specs = (
        ("overshoot_percent", "maximum_overshoot_percent", "overshoot_tolerance", "maximum"),
        ("final_value", "minimum_final_value", "final_value_tolerance", "minimum"),
        ("final_value", "maximum_final_value", "final_value_tolerance", "maximum"),
    )
    candidates = []
    for observed_name, target_name, tolerance_name, direction in specs:
        if target_name not in context:
            continue
        observed = _finite(primary.get(observed_name), f"control_step_response.{observed_name}")
        target = _finite(context.get(target_name), f"control_context.{target_name}")
        tolerance = _finite(context.get(tolerance_name, 0.0), f"control_context.{tolerance_name}")
        if tolerance < 0:
            raise PipelineAdapterError(f"{tolerance_name} must be non-negative")
        if observed_name == "overshoot_percent" and target < 0:
            raise PipelineAdapterError("maximum_overshoot_percent must be non-negative")
        candidates.append({
            "name": f"control-target-{target_name}",
            "observed": observed,
            "benchmark": target,
            "tolerance": tolerance,
            "direction": direction,
        })
    if not candidates:
        raise PipelineAdapterError("control_target_audit requires at least one explicit target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_control_response_adapters() -> None:
    adapters = {
        "control_ticket_to_step_response": control_ticket_to_step_response,
        "control_response_to_tail_statistics": control_response_to_tail_statistics,
        "tail_statistics_to_stability_audit": tail_statistics_to_stability_audit,
        "control_response_to_dc_gain_audit": control_response_to_dc_gain_audit,
        "control_response_to_target_audit": control_response_to_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"control-response adapter name collision: {name}")
        ADAPTERS[name] = handler
