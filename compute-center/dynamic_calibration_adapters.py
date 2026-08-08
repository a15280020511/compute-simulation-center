#!/usr/bin/env python3
"""Deterministic adapters for the dynamic calibration family."""
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
    raw = initial_inputs.get("calibration_context")
    return {} if raw is None else _mapping(raw, "inputs.calibration_context")


def _parameter_value(primary: Mapping[str, Any], name: str) -> float:
    params = _mapping(primary.get("parameters"), "calibration.parameters")
    row = _mapping(params.get(name), f"calibration.parameters.{name}")
    return _finite(row.get("value"), f"calibration.parameters.{name}.value")


def calibration_ticket_to_lmfit(initial_inputs, stage_results, stage):
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "lmfit_exponential_calibration":
        raise PipelineAdapterError("calibration family requires lmfit_exponential_calibration entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "calibration_context"}


def calibration_to_residual_statistics(initial_inputs, stage_results, stage):
    del stage
    primary = _mapping(stage_results.get("exponential_calibration"), "stage_results.exponential_calibration")
    x = np.asarray([_finite(v, "inputs.x[]") for v in _sequence(initial_inputs.get("x"), "inputs.x")], dtype=float)
    y = np.asarray([_finite(v, "inputs.y[]") for v in _sequence(initial_inputs.get("y"), "inputs.y")], dtype=float)
    if x.size != y.size or x.size < 5:
        raise PipelineAdapterError("x/y must align and contain at least five observations")
    amplitude = _parameter_value(primary, "amplitude")
    decay = _parameter_value(primary, "decay")
    offset = _parameter_value(primary, "offset")
    fitted = amplitude * np.exp(-decay * x) + offset
    residuals = y - fitted
    if not np.all(np.isfinite(residuals)):
        raise PipelineAdapterError("reconstructed residuals are non-finite")
    return {"data": residuals.tolist()}


def residual_statistics_to_rmse_audit(initial_inputs, stage_results, stage):
    del stage
    primary = _mapping(stage_results.get("exponential_calibration"), "stage_results.exponential_calibration")
    stats = _mapping(stage_results.get("residual_statistics"), "stage_results.residual_statistics")
    mean = _finite(stats.get("mean"), "residual_statistics.mean")
    std = _finite(stats.get("standard_deviation_population"), "residual_statistics.standard_deviation_population")
    reconstructed_rmse = math.sqrt(mean * mean + std * std)
    reported_rmse = _finite(primary.get("rmse"), "exponential_calibration.rmse")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("rmse_consistency_tolerance", 1e-10), "calibration_context.rmse_consistency_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("rmse_consistency_tolerance must be non-negative")
    return {"mode": "benchmark_comparison", "candidates": [{"name": "lmfit-rmse-vs-residual-statistics", "observed": reconstructed_rmse, "benchmark": reported_rmse, "tolerance": tolerance, "direction": "absolute"}]}


def residual_statistics_to_bias_audit(initial_inputs, stage_results, stage):
    del stage
    stats = _mapping(stage_results.get("residual_statistics"), "stage_results.residual_statistics")
    observed = _finite(stats.get("mean"), "residual_statistics.mean")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("maximum_abs_residual_mean"), "calibration_context.maximum_abs_residual_mean")
    if tolerance < 0:
        raise PipelineAdapterError("maximum_abs_residual_mean must be non-negative")
    return {"mode": "benchmark_comparison", "candidates": [{"name": "residual-mean-bias", "observed": observed, "benchmark": 0.0, "tolerance": tolerance, "direction": "absolute"}]}


def calibration_to_parameter_target_audit(initial_inputs, stage_results, stage):
    del stage
    primary = _mapping(stage_results.get("exponential_calibration"), "stage_results.exponential_calibration")
    context = _context(initial_inputs)
    specs = (
        ("amplitude", "expected_amplitude", "amplitude_tolerance"),
        ("decay", "expected_decay", "decay_tolerance"),
        ("offset", "expected_offset", "offset_tolerance"),
    )
    candidates = []
    for parameter_name, target_name, tolerance_name in specs:
        if target_name not in context:
            continue
        target = _finite(context.get(target_name), f"calibration_context.{target_name}")
        tolerance = _finite(context.get(tolerance_name, 0.0), f"calibration_context.{tolerance_name}")
        if tolerance < 0:
            raise PipelineAdapterError(f"{tolerance_name} must be non-negative")
        candidates.append({"name": f"calibration-target-{parameter_name}", "observed": _parameter_value(primary, parameter_name), "benchmark": target, "tolerance": tolerance, "direction": "absolute"})
    if not candidates:
        raise PipelineAdapterError("parameter_target_audit requires at least one explicit target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_calibration_adapters() -> None:
    adapters = {
        "calibration_ticket_to_lmfit": calibration_ticket_to_lmfit,
        "calibration_to_residual_statistics": calibration_to_residual_statistics,
        "residual_statistics_to_rmse_audit": residual_statistics_to_rmse_audit,
        "residual_statistics_to_bias_audit": residual_statistics_to_bias_audit,
        "calibration_to_parameter_target_audit": calibration_to_parameter_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"calibration adapter name collision: {name}")
        ADAPTERS[name] = handler
