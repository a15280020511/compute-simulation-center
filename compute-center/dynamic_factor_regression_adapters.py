#!/usr/bin/env python3
"""Deterministic adapters and independent NumPy checks for factor regression."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pipeline_adapters import ADAPTERS, PipelineAdapterError

PRIMARY_STAGE = "factor_regression"


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
    raw = initial_inputs.get("factor_regression_context")
    return {} if raw is None else _mapping(raw, "inputs.factor_regression_context")


def _design(initial_inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    asset = np.asarray([_finite(v, "inputs.asset_returns[]") for v in _sequence(initial_inputs.get("asset_returns"), "inputs.asset_returns")], dtype=float)
    if not 10 <= asset.size <= 20_000:
        raise PipelineAdapterError("asset_returns must contain 10 to 20000 values")
    raw_factors = _mapping(initial_inputs.get("factors"), "inputs.factors")
    if not 1 <= len(raw_factors) <= 20:
        raise PipelineAdapterError("factors must contain 1 to 20 series")
    names = [str(name) for name in raw_factors]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise PipelineAdapterError("factor names must be non-empty and unique")
    columns: list[np.ndarray] = []
    for name in names:
        values = np.asarray([_finite(v, f"inputs.factors[{name}][]") for v in _sequence(raw_factors[name], f"inputs.factors[{name}]")], dtype=float)
        if values.size != asset.size:
            raise PipelineAdapterError("all factor series must match asset return length")
        columns.append(values)
    x = np.column_stack(columns)
    include_intercept = bool(initial_inputs.get("include_intercept", True))
    if include_intercept:
        x = np.column_stack([np.ones(asset.size, dtype=float), x])
        parameter_names = ["alpha", *names]
    else:
        parameter_names = names
    if asset.size <= x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise PipelineAdapterError("dynamic factor regression requires an overdetermined full-rank design matrix")
    return asset, x, parameter_names


def _primary(stage_results: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    return _mapping(stage_results.get(PRIMARY_STAGE), f"stage_results.{PRIMARY_STAGE}")


def _independent_metrics(initial_inputs: Mapping[str, Any]) -> tuple[list[str], np.ndarray, float, float]:
    y, x, parameter_names = _design(initial_inputs)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    ssr = float(residual @ residual)
    include_intercept = bool(initial_inputs.get("include_intercept", True))
    tss = float(np.sum((y - np.mean(y)) ** 2)) if include_intercept else float(y @ y)
    if tss <= np.finfo(float).eps:
        raise PipelineAdapterError("dynamic factor regression requires non-constant asset returns")
    r_squared = float(1.0 - ssr / tss)
    residual_volatility = float(np.std(residual, ddof=max(1, len(parameter_names))))
    if not math.isfinite(residual_volatility):
        raise PipelineAdapterError("independent residual volatility is non-finite")
    return parameter_names, beta, r_squared, residual_volatility


def factor_ticket_to_statsmodels(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "factor_regression":
        raise PipelineAdapterError("factor-regression family requires factor_regression entry mode")
    _design(initial_inputs)
    return {key: value for key, value in initial_inputs.items() if key != "factor_regression_context"}


def factor_to_numpy_exact_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _primary(stage_results)
    parameter_names, beta, r_squared, residual_volatility = _independent_metrics(initial_inputs)
    parameters = _mapping(primary.get("parameters"), "primary.parameters")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("exact_consistency_tolerance", 1e-9), "factor_regression_context.exact_consistency_tolerance")
    if not 0 <= tolerance <= 1e-6:
        raise PipelineAdapterError("exact_consistency_tolerance must be between 0 and 1e-6")
    candidates: list[dict[str, Any]] = []
    for index, name in enumerate(parameter_names):
        row = _mapping(parameters.get(name), f"primary.parameters[{name}]")
        candidates.append({"name": f"coefficient:{name}", "observed": _finite(row.get("coefficient"), f"primary.parameters[{name}].coefficient"), "benchmark": float(beta[index]), "tolerance": tolerance, "direction": "absolute"})
    candidates.extend([
        {"name": "r_squared", "observed": _finite(primary.get("r_squared"), "primary.r_squared"), "benchmark": r_squared, "tolerance": tolerance, "direction": "absolute"},
        {"name": "residual_volatility", "observed": _finite(primary.get("residual_volatility"), "primary.residual_volatility"), "benchmark": residual_volatility, "tolerance": tolerance, "direction": "absolute"},
    ])
    return {"mode": "benchmark_comparison", "candidates": candidates}


def factor_to_r_squared_target_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _primary(stage_results)
    context = _context(initial_inputs)
    if "minimum_r_squared" not in context:
        raise PipelineAdapterError("r_squared_target_audit requires minimum_r_squared")
    target = _finite(context["minimum_r_squared"], "factor_regression_context.minimum_r_squared")
    tolerance = _finite(context.get("r_squared_target_tolerance", 0.0), "factor_regression_context.r_squared_target_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("r_squared_target_tolerance must be non-negative")
    return {"mode": "benchmark_comparison", "candidates": [{"name": "minimum-r-squared", "observed": _finite(primary.get("r_squared"), "primary.r_squared"), "benchmark": target, "tolerance": tolerance, "direction": "minimum"}]}


def factor_to_residual_target_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _primary(stage_results)
    context = _context(initial_inputs)
    if "maximum_residual_volatility" not in context:
        raise PipelineAdapterError("residual_volatility_target_audit requires maximum_residual_volatility")
    target = _finite(context["maximum_residual_volatility"], "factor_regression_context.maximum_residual_volatility")
    tolerance = _finite(context.get("residual_volatility_target_tolerance", 0.0), "factor_regression_context.residual_volatility_target_tolerance")
    if target < 0 or tolerance < 0:
        raise PipelineAdapterError("residual volatility target and tolerance must be non-negative")
    return {"mode": "benchmark_comparison", "candidates": [{"name": "maximum-residual-volatility", "observed": _finite(primary.get("residual_volatility"), "primary.residual_volatility"), "benchmark": target, "tolerance": tolerance, "direction": "maximum"}]}


def install_factor_regression_adapters() -> None:
    adapters = {
        "factor_ticket_to_statsmodels": factor_ticket_to_statsmodels,
        "factor_to_numpy_exact_audit": factor_to_numpy_exact_audit,
        "factor_to_r_squared_target_audit": factor_to_r_squared_target_audit,
        "factor_to_residual_target_audit": factor_to_residual_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"factor-regression adapter name collision: {name}")
        ADAPTERS[name] = handler
