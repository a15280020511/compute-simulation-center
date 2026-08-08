#!/usr/bin/env python3
"""Deterministic adapters for the dynamic conformal-prediction family."""
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
    raw = initial_inputs.get("conformal_context")
    return {} if raw is None else _mapping(raw, "inputs.conformal_context")


def _observed(initial_inputs: Mapping[str, Any]) -> list[float]:
    context = _context(initial_inputs)
    raw = _sequence(context.get("validation_observed"), "conformal_context.validation_observed")
    return [_finite(value, "conformal_context.validation_observed[]") for value in raw]


def conformal_ticket_to_mapie(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "mapie_conformal_interval":
        raise PipelineAdapterError("conformal-prediction family requires mapie_conformal_interval entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "conformal_context"}


def conformal_to_interval_validation(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("mapie_conformal_interval"), "stage_results.mapie_conformal_interval")
    lower = list(_sequence(primary.get("lower_bounds"), "mapie lower_bounds"))
    upper = list(_sequence(primary.get("upper_bounds"), "mapie upper_bounds"))
    observed = _observed(initial_inputs)
    if len(lower) != len(upper) or len(lower) != len(observed):
        raise PipelineAdapterError("MAPIE interval rows must match validation_observed")
    confidence = _finite(primary.get("confidence"), "mapie confidence")
    return {"mode": "prediction_interval_validation", "lower": lower, "upper": upper, "observed": observed, "alpha": 1.0 - confidence}


def conformal_validation_to_width_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("mapie_conformal_interval"), "stage_results.mapie_conformal_interval")
    validation = _mapping(stage_results.get("prediction_interval_validation"), "stage_results.prediction_interval_validation")
    context = _context(initial_inputs)
    tolerance = _nonnegative(context.get("width_consistency_tolerance"), "conformal_context.width_consistency_tolerance", 1e-12)
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "mapie-validation-mean-interval-width-consistency",
            "observed": _finite(validation.get("average_interval_width"), "interval validation average_interval_width"),
            "benchmark": _finite(primary.get("mean_interval_width"), "mapie mean_interval_width"),
            "tolerance": tolerance,
            "direction": "absolute",
        }],
    }


def conformal_validation_to_interval_targets(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    validation = _mapping(stage_results.get("prediction_interval_validation"), "stage_results.prediction_interval_validation")
    context = _context(initial_inputs)
    tolerance = _nonnegative(context.get("target_tolerance"), "conformal_context.target_tolerance", 0.0)
    specs = (
        ("minimum_empirical_coverage", "empirical_coverage", "minimum"),
        ("maximum_average_interval_width", "average_interval_width", "maximum"),
        ("maximum_mean_interval_score", "mean_interval_score", "maximum"),
    )
    candidates = []
    for target_name, observed_name, direction in specs:
        if target_name not in context:
            continue
        candidates.append({
            "name": f"conformal-target-{target_name}",
            "observed": _finite(validation.get(observed_name), observed_name),
            "benchmark": _finite(context[target_name], f"conformal_context.{target_name}"),
            "tolerance": tolerance,
            "direction": direction,
        })
    if not candidates:
        raise PipelineAdapterError("interval_target_audit requires at least one interval target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def conformal_to_realized_feedback(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("mapie_conformal_interval"), "stage_results.mapie_conformal_interval")
    predictions = list(_sequence(primary.get("predictions"), "mapie predictions"))
    observed = _observed(initial_inputs)
    if len(predictions) != len(observed):
        raise PipelineAdapterError("MAPIE prediction rows must match validation_observed")
    context = _context(initial_inputs)
    drift_ratio_threshold = _finite(context.get("drift_ratio_threshold", 1.5), "conformal_context.drift_ratio_threshold")
    if drift_ratio_threshold <= 0:
        raise PipelineAdapterError("drift_ratio_threshold must be positive")
    return {"mode": "realized_outcome_feedback", "predicted": predictions, "observed": observed, "drift_ratio_threshold": drift_ratio_threshold}


def conformal_feedback_to_point_targets(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    feedback = _mapping(stage_results.get("realized_outcome_feedback"), "stage_results.realized_outcome_feedback")
    context = _context(initial_inputs)
    tolerance = _nonnegative(context.get("target_tolerance"), "conformal_context.target_tolerance", 0.0)
    candidates = []
    if "maximum_point_rmse" in context:
        candidates.append({
            "name": "conformal-target-maximum-point-rmse",
            "observed": _finite(feedback.get("rmse"), "feedback rmse"),
            "benchmark": _finite(context["maximum_point_rmse"], "conformal_context.maximum_point_rmse"),
            "tolerance": tolerance,
            "direction": "maximum",
        })
    if "maximum_absolute_bias" in context:
        candidates.append({
            "name": "conformal-target-maximum-absolute-bias",
            "observed": abs(_finite(feedback.get("bias"), "feedback bias")),
            "benchmark": _finite(context["maximum_absolute_bias"], "conformal_context.maximum_absolute_bias"),
            "tolerance": tolerance,
            "direction": "maximum",
        })
    if not candidates:
        raise PipelineAdapterError("point_target_audit requires at least one point target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_conformal_prediction_adapters() -> None:
    adapters = {
        "conformal_ticket_to_mapie": conformal_ticket_to_mapie,
        "conformal_to_interval_validation": conformal_to_interval_validation,
        "conformal_validation_to_width_audit": conformal_validation_to_width_audit,
        "conformal_validation_to_interval_targets": conformal_validation_to_interval_targets,
        "conformal_to_realized_feedback": conformal_to_realized_feedback,
        "conformal_feedback_to_point_targets": conformal_feedback_to_point_targets,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"conformal-prediction adapter name collision: {name}")
        ADAPTERS[name] = handler
