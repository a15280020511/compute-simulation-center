#!/usr/bin/env python3
"""Deterministic adapters for the dynamic drift capability family."""
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


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("drift_context")
    return {} if raw is None else _mapping(raw, "inputs.drift_context")


def drift_ticket_to_evidently(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "evidently_data_drift":
        raise PipelineAdapterError("drift family requires evidently_data_drift entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "drift_context"}


def _selected_column(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
) -> tuple[int, str, list[float]]:
    primary = _mapping(stage_results.get("distribution_drift"), "stage_results.distribution_drift")
    columns = _sequence(primary.get("columns"), "distribution_drift.columns")
    reference = _sequence(initial_inputs.get("reference"), "inputs.reference")
    current = _sequence(initial_inputs.get("current"), "inputs.current")
    if not reference or not current:
        raise PipelineAdapterError("reference and current must be non-empty")
    first = _sequence(reference[0], "inputs.reference[0]")
    width = len(first)
    if width < 2 or len(columns) != width:
        raise PipelineAdapterError("drift column metadata must align with the source matrices")
    raw_names = initial_inputs.get("variable_names")
    names = [f"x{index}" for index in range(width)] if raw_names is None else [str(item) for item in _sequence(raw_names, "inputs.variable_names")]
    if len(names) != width:
        raise PipelineAdapterError("variable_names must align with the source matrices")
    best_index = 0
    best_stat = -1.0
    for index, raw in enumerate(columns):
        row = _mapping(raw, f"distribution_drift.columns[{index}]")
        name = str(row.get("column") or "")
        if name != names[index]:
            raise PipelineAdapterError("distribution drift column order does not match source data")
        statistic = _finite(row.get("ks_statistic"), f"distribution_drift.columns[{index}].ks_statistic")
        if statistic > best_stat:
            best_stat = statistic
            best_index = index
    values: list[float] = []
    for group_name, matrix in (("reference", reference), ("current", current)):
        for row_index, raw_row in enumerate(matrix):
            row = _sequence(raw_row, f"inputs.{group_name}[{row_index}]")
            if len(row) != width:
                raise PipelineAdapterError(f"inputs.{group_name} must be rectangular")
            values.append(_finite(row[best_index], f"inputs.{group_name}[{row_index}][{best_index}]"))
    return best_index, names[best_index], values


def evidently_to_adwin(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    _, _, values = _selected_column(initial_inputs, stage_results)
    context = _context(initial_inputs)
    delta = _finite(context.get("adwin_delta", 0.002), "drift_context.adwin_delta")
    if not 1e-8 <= delta <= 0.5:
        raise PipelineAdapterError("adwin_delta must be between 1e-8 and 0.5")
    return {"mode": "river_adwin_drift", "values": values, "delta": delta}


def evidently_to_change_point(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    _, _, values = _selected_column(initial_inputs, stage_results)
    context = _context(initial_inputs)
    model = str(context.get("change_point_cost_model") or "l2")
    if model not in {"l1", "l2", "rbf", "normal"}:
        raise PipelineAdapterError("change_point_cost_model is unsupported")
    penalty = _finite(context.get("change_point_penalty", 5.0), "drift_context.change_point_penalty")
    if penalty < 0:
        raise PipelineAdapterError("change_point_penalty must be non-negative")
    return {"mode": "change_point_detection", "values": values, "cost_model": model, "penalty": penalty}


def evidently_to_drift_share_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("distribution_drift"), "stage_results.distribution_drift")
    observed = _finite(primary.get("drift_share_screen"), "distribution_drift.drift_share_screen")
    context = _context(initial_inputs)
    expected = _finite(context.get("expected_drift_share"), "drift_context.expected_drift_share")
    tolerance = _finite(context.get("drift_share_tolerance", 0.0), "drift_context.drift_share_tolerance")
    if not 0.0 <= expected <= 1.0 or tolerance < 0:
        raise PipelineAdapterError("expected_drift_share must be [0,1] and tolerance non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "ks-screen-drift-share",
            "observed": observed,
            "benchmark": expected,
            "tolerance": tolerance,
            "direction": "absolute",
        }],
    }


def install_drift_adapters() -> None:
    adapters = {
        "drift_ticket_to_evidently": drift_ticket_to_evidently,
        "evidently_to_adwin": evidently_to_adwin,
        "evidently_to_change_point": evidently_to_change_point,
        "evidently_to_drift_share_audit": evidently_to_drift_share_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"drift adapter name collision: {name}")
        ADAPTERS[name] = handler
