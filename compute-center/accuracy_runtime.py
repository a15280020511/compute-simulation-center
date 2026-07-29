#!/usr/bin/env python3
"""Executable accuracy controls for registered compute operations.

This module connects previously isolated calibration, replication, model-comparison and
residual-diagnostic utilities to the production governance wrapper. It accepts only data
paths and numeric observations; no ticket-supplied code, imports or formulas are evaluated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from calibration_engine import CalibrationError, calibrate
from model_comparison import compare_models
from model_ensemble import ensemble_predictions
from residual_diagnostics import diagnose_residuals


class AccuracyRuntimeError(ValueError):
    pass


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parts(path: str) -> list[str]:
    rows = [part for part in str(path).split(".") if part]
    if not rows:
        raise AccuracyRuntimeError("data path must not be empty")
    return rows


def _deep_get(value: Any, path: str) -> Any:
    current = value
    for part in _parts(path):
        if isinstance(current, Mapping):
            if part not in current:
                raise AccuracyRuntimeError(f"path not found: {path}")
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise AccuracyRuntimeError(f"path index outside range: {path}")
            current = current[index]
        else:
            raise AccuracyRuntimeError(f"path cannot be resolved: {path}")
    return current


def _deep_set(value: dict[str, Any], path: str, replacement: float) -> None:
    current: Any = value
    parts = _parts(path)
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                raise AccuracyRuntimeError(f"parameter path not found: {path}")
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise AccuracyRuntimeError(f"parameter path cannot be resolved: {path}")
    final = parts[-1]
    if isinstance(current, dict) and final in current:
        current[final] = float(replacement)
    elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = float(replacement)
    else:
        raise AccuracyRuntimeError(f"parameter path not found: {path}")


def _finite_vector(value: Any, name: str) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    if not values or not all(math.isfinite(item) for item in values):
        raise AccuracyRuntimeError(f"{name} must contain finite numeric values")
    return values


def _extract(result: Mapping[str, Any], paths: Sequence[str]) -> list[float]:
    values: list[float] = []
    for path in paths:
        values.extend(_finite_vector(_deep_get(result, str(path)), f"result path {path}"))
    return values


def _parameter_paths(profile: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in profile.get("parameters") or []:
        if not isinstance(row, Mapping):
            raise AccuracyRuntimeError("calibration parameters must be objects")
        name = str(row.get("name") or "")
        path = str(row.get("input_path") or f"parameters.{name}")
        result[name] = path
    return result


def execute_calibration(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    inputs: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile is None:
        return dict(inputs), {
            "schema_version": "compute-calibration-assurance-v2",
            "requested": False,
            "execution_status": "NOT_REQUESTED",
        }
    observations_raw = profile.get("observations")
    result_paths = profile.get("result_paths")
    if not isinstance(observations_raw, list) or not observations_raw:
        raise AccuracyRuntimeError("calibration_profile.observations is required for executable calibration")
    if not isinstance(result_paths, list) or not result_paths:
        raise AccuracyRuntimeError("calibration_profile.result_paths is required for executable calibration")
    observations = _finite_vector(observations_raw, "calibration observations")
    parameter_paths = _parameter_paths(profile)

    def model(parameters: Mapping[str, float]) -> Sequence[float]:
        candidate = copy.deepcopy(dict(inputs))
        for name, value in parameters.items():
            _deep_set(candidate, parameter_paths[name], value)
        raw = handler(candidate)
        predicted = _extract(raw, [str(item) for item in result_paths])
        if len(predicted) != len(observations):
            raise AccuracyRuntimeError(
                f"calibration prediction length {len(predicted)} does not match observation length {len(observations)}"
            )
        return predicted

    starts = max(1, int(profile.get("multi_start", 1)))
    seed = int(profile.get("seed", 0))
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    for index in range(starts):
        run_profile = copy.deepcopy(dict(profile))
        run_profile.pop("observations", None)
        run_profile.pop("weights", None)
        run_profile.pop("result_paths", None)
        run_profile.pop("validation_observations", None)
        run_profile.pop("input_snapshot_sha256", None)
        rows = run_profile.get("parameters") or []
        for row in rows:
            row.pop("input_path", None)
            if index:
                low = float(row["minimum"])
                high = float(row["maximum"])
                row["initial"] = float(rng.uniform(low, high)) if high > low else low
        try:
            result = calibrate(
                model,
                observations,
                run_profile,
                weights=profile.get("weights"),
            )
            result["start_index"] = index
            candidates.append(result)
        except (CalibrationError, AccuracyRuntimeError, ValueError) as exc:
            candidates.append({"start_index": index, "success": False, "message": str(exc), "objective_value": math.inf})
    successful = [row for row in candidates if row.get("success") and math.isfinite(float(row.get("objective_value", math.inf)))]
    if not successful:
        raise AccuracyRuntimeError("all calibration starts failed")
    selected = min(successful, key=lambda row: (float(row["objective_value"]), int(row["start_index"])))
    calibrated_inputs = copy.deepcopy(dict(inputs))
    for name, value in selected["parameters"].items():
        _deep_set(calibrated_inputs, parameter_paths[name], float(value))

    validation = None
    validation_raw = profile.get("validation_observations")
    if isinstance(validation_raw, list) and validation_raw:
        validation_actual = np.asarray(_finite_vector(validation_raw, "validation observations"), dtype=float)
        validation_prediction = np.asarray(model(selected["parameters"]), dtype=float)
        if validation_prediction.shape != validation_actual.shape:
            raise AccuracyRuntimeError("validation observations must match extracted calibrated predictions")
        error = validation_prediction - validation_actual
        validation = {
            "count": int(error.size),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }

    assurance = {
        "schema_version": "compute-calibration-assurance-v2",
        "requested": True,
        "profile_validated": True,
        "execution_status": "EXECUTED",
        "backend": selected["backend"],
        "multi_start_requested": starts,
        "multi_start_executed": len(candidates),
        "successful_start_count": len(successful),
        "selected_start_index": selected["start_index"],
        "result": selected,
        "validation": validation,
        "input_snapshot_sha256": profile.get("input_snapshot_sha256"),
        "candidate_results": candidates,
    }
    assurance["assurance_sha256"] = _sha(assurance)
    return calibrated_inputs, assurance


def execute_experiment(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    inputs: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if profile is None:
        return {
            "schema_version": "compute-experiment-execution-v1",
            "requested": False,
            "status": "NOT_REQUESTED",
        }
    replications = int(profile.get("replications", 1))
    base_seed = profile.get("base_seed")
    result_paths = profile.get("result_paths")
    if replications > 1 and base_seed is None:
        raise AccuracyRuntimeError("experiment_profile.base_seed is required when replications exceed one")
    if replications > 1 and (not isinstance(result_paths, list) or not result_paths):
        raise AccuracyRuntimeError("experiment_profile.result_paths is required for replicated accuracy evidence")
    seed_path = str(profile.get("seed_path") or "seed")
    rows: list[list[float]] = []
    seeds: list[int | None] = []
    for index in range(replications):
        candidate = copy.deepcopy(dict(inputs))
        seed = None if base_seed is None else (int(base_seed) + index) % (2**32)
        if seed is not None:
            _deep_set(candidate, seed_path, seed)
        result = handler(candidate)
        rows.append(_extract(result, [str(item) for item in result_paths]) if result_paths else [])
        seeds.append(seed)
    matrix = np.asarray(rows, dtype=float) if rows and rows[0] else np.empty((replications, 0))
    metrics: list[dict[str, Any]] = []
    if matrix.size:
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            mcse = sd / math.sqrt(values.size) if values.size else 0.0
            metrics.append({
                "metric_index": column,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": sd,
                "monte_carlo_standard_error": mcse,
                "confidence_interval_95": [float(np.mean(values) - 1.96 * mcse), float(np.mean(values) + 1.96 * mcse)],
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })
    target = profile.get("precision_target")
    precision_pass = True if target is None else bool(metrics and all(row["monte_carlo_standard_error"] <= float(target) for row in metrics))
    report = {
        "schema_version": "compute-experiment-execution-v1",
        "requested": True,
        "status": "PASS" if precision_pass else "FAIL",
        "requested_replications": replications,
        "executed_replications": len(rows),
        "seeds": seeds,
        "result_paths": list(result_paths or []),
        "metrics": metrics,
        "precision_target": target,
        "precision_rule_passed": precision_pass,
        "stopping_rule": profile.get("stopping_rule"),
    }
    report["execution_sha256"] = _sha(report)
    return report


def execute_validation(result: Mapping[str, Any], profile: Mapping[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        return {
            "schema_version": "compute-validation-result-v1",
            "requested": False,
            "status": "NOT_REQUESTED",
        }
    actual_raw = profile.get("actual_values")
    result_paths = profile.get("result_paths")
    if not isinstance(actual_raw, list) or not actual_raw or not isinstance(result_paths, list) or not result_paths:
        return {
            "schema_version": "compute-validation-result-v1",
            "requested": True,
            "status": "NOT_EXECUTED",
            "reason": "actual_values and result_paths are required for executable validation",
        }
    actual = _finite_vector(actual_raw, "validation actual_values")
    predicted = _extract(result, [str(item) for item in result_paths])
    if len(actual) != len(predicted):
        raise AccuracyRuntimeError("validation actual and prediction lengths do not match")
    residual = diagnose_residuals(actual, predicted, time_index=profile.get("time_index"), groups=profile.get("groups"))

    predictions: dict[str, Sequence[float]] = {"candidate": predicted}
    baseline = profile.get("baseline_predictions")
    if isinstance(baseline, list):
        predictions[str(profile.get("baseline_model_id") or "baseline")] = baseline
    alternatives = profile.get("alternative_predictions")
    if isinstance(alternatives, Mapping):
        predictions.update({str(key): value for key, value in alternatives.items() if isinstance(value, list)})
    comparison = None
    if len(predictions) > 1:
        baseline_id = str(profile.get("baseline_model_id") or "baseline")
        if baseline_id not in predictions:
            raise AccuracyRuntimeError("baseline_model_id does not match supplied baseline_predictions")
        comparison = compare_models(
            actual,
            predictions,
            baseline_model_id=baseline_id,
            complexity=profile.get("model_complexity"),
            minimum_improvement_over_baseline=float(profile.get("minimum_improvement_over_baseline", 0.0)),
        )

    ensemble = None
    if profile.get("ensemble_method") and len(predictions) > 1:
        errors = {name: float(np.sqrt(np.mean((np.asarray(values, dtype=float) - np.asarray(actual, dtype=float)) ** 2))) for name, values in predictions.items()}
        ensemble = ensemble_predictions(
            predictions,
            method=str(profile["ensemble_method"]),
            validation_errors=errors,
            maximum_weight=float(profile.get("maximum_ensemble_weight", 0.7)),
        )
    thresholds = profile.get("metric_thresholds") if isinstance(profile.get("metric_thresholds"), Mapping) else {}
    rmse = float(residual["diagnostics"]["rmse"])
    mae = float(residual["diagnostics"]["mae"])
    threshold_failures = []
    if thresholds.get("rmse_maximum") is not None and rmse > float(thresholds["rmse_maximum"]):
        threshold_failures.append("RMSE_THRESHOLD_FAILED")
    if thresholds.get("mae_maximum") is not None and mae > float(thresholds["mae_maximum"]):
        threshold_failures.append("MAE_THRESHOLD_FAILED")
    if comparison and comparison.get("selected_model_id") != "candidate":
        threshold_failures.append("CANDIDATE_DID_NOT_BEAT_BASELINE")
    status = "FAIL" if threshold_failures else "WARN" if residual["status"] == "WARN" else "PASS"
    report = {
        "schema_version": "compute-validation-result-v1",
        "requested": True,
        "status": status,
        "strategy": profile.get("strategy"),
        "residual_diagnostics": residual,
        "model_comparison": comparison,
        "ensemble": ensemble,
        "threshold_failures": threshold_failures,
        "source_snapshot_sha256": profile.get("source_snapshot_sha256"),
    }
    report["validation_sha256"] = _sha(report)
    return report


def derive_evidence_maturity(
    model: Mapping[str, Any],
    calibration: Mapping[str, Any],
    validation: Mapping[str, Any],
    experiment: Mapping[str, Any],
    quality_profile: Mapping[str, Any],
) -> dict[str, Any]:
    engineering = str(model.get("maturity") or "experimental")
    benchmark_ids = list(model.get("benchmark_ids") or []) + list(quality_profile.get("benchmark_ids") or [])
    calibration_needed = bool(model.get("calibration_supported"))
    calibration_ok = not calibration_needed or calibration.get("execution_status") == "EXECUTED"
    validation_ok = validation.get("status") == "PASS"
    replication_ok = experiment.get("status") in {"PASS", "NOT_REQUESTED"}
    feedback_present = bool(quality_profile.get("operational_feedback_evidence_sha256"))
    technical_review = bool(quality_profile.get("technical_review_evidence_sha256"))
    if validation_ok and calibration_ok and replication_ok and benchmark_ids and feedback_present and technical_review:
        evidence = "decision-grade"
    elif validation_ok and calibration_ok and replication_ok and benchmark_ids:
        evidence = "production"
    elif benchmark_ids or validation.get("requested"):
        evidence = "controlled-preview"
    else:
        evidence = "experimental"
    return {
        "schema_version": "compute-maturity-assessment-v1",
        "engineering_maturity": engineering,
        "evidence_maturity": evidence,
        "calibration_required": calibration_needed,
        "calibration_evidence_passed": calibration_ok,
        "validation_evidence_passed": validation_ok,
        "replication_evidence_passed": replication_ok,
        "benchmark_ids": sorted(set(str(item) for item in benchmark_ids)),
        "operational_feedback_evidence_present": feedback_present,
        "technical_review_evidence_present": technical_review,
    }
