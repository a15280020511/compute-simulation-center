#!/usr/bin/env python3
"""Executable accuracy controls for registered compute operations.

Only fixed data paths and numeric observations are accepted. No ticket-supplied
code, import, expression or formula is evaluated.
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
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
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
        elif (
            isinstance(current, Sequence)
            and not isinstance(current, (str, bytes))
            and part.isdigit()
        ):
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
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
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
        values.extend(
            _finite_vector(_deep_get(result, str(path)), f"result path {path}")
        )
    return values


def _parameter_paths(profile: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in profile.get("parameters") or []:
        if not isinstance(row, Mapping):
            raise AccuracyRuntimeError("calibration parameters must be objects")
        name = str(row.get("name") or "")
        result[name] = str(row.get("input_path") or f"parameters.{name}")
    return result


def _not_requested(schema_version: str, status_key: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "requested": False,
        status_key: status,
    }


def _calibration_model(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    inputs: Mapping[str, Any],
    parameter_paths: Mapping[str, str],
    result_paths: list[str],
    expected_length: int,
) -> Callable[[Mapping[str, float]], Sequence[float]]:
    def model(parameters: Mapping[str, float]) -> Sequence[float]:
        candidate = copy.deepcopy(dict(inputs))
        for name, value in parameters.items():
            _deep_set(candidate, parameter_paths[name], value)
        predicted = _extract(handler(candidate), result_paths)
        if len(predicted) != expected_length:
            raise AccuracyRuntimeError(
                f"calibration prediction length {len(predicted)} does not match "
                f"observation length {expected_length}"
            )
        return predicted

    return model


def _calibration_run_profile(
    profile: Mapping[str, Any], index: int, rng: np.random.Generator
) -> dict[str, Any]:
    run_profile = copy.deepcopy(dict(profile))
    for key in (
        "observations",
        "weights",
        "result_paths",
        "validation_observations",
        "input_snapshot_sha256",
    ):
        run_profile.pop(key, None)
    for row in run_profile.get("parameters") or []:
        row.pop("input_path", None)
        if index:
            low = float(row["minimum"])
            high = float(row["maximum"])
            row["initial"] = float(rng.uniform(low, high)) if high > low else low
    return run_profile


def _calibration_candidates(
    model: Callable[[Mapping[str, float]], Sequence[float]],
    observations: list[float],
    profile: Mapping[str, Any],
    starts: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(profile.get("seed", 0)))
    candidates: list[dict[str, Any]] = []
    for index in range(starts):
        try:
            result = calibrate(
                model,
                observations,
                _calibration_run_profile(profile, index, rng),
                weights=profile.get("weights"),
            )
            result["start_index"] = index
            candidates.append(result)
        except (CalibrationError, AccuracyRuntimeError, ValueError) as exc:
            candidates.append(
                {
                    "start_index": index,
                    "success": False,
                    "message": str(exc),
                    "objective_value": math.inf,
                }
            )
    return candidates


def _successful_candidates(candidates: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in candidates
        if row.get("success")
        and math.isfinite(float(row.get("objective_value", math.inf)))
    ]


def _calibration_validation(
    model: Callable[[Mapping[str, float]], Sequence[float]],
    parameters: Mapping[str, float],
    profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw = profile.get("validation_observations")
    if not isinstance(raw, list) or not raw:
        return None
    actual = np.asarray(_finite_vector(raw, "validation observations"), dtype=float)
    predicted = np.asarray(model(parameters), dtype=float)
    if predicted.shape != actual.shape:
        raise AccuracyRuntimeError(
            "validation observations must match extracted calibrated predictions"
        )
    error = predicted - actual
    return {
        "count": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def execute_calibration(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    inputs: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile is None:
        return dict(inputs), _not_requested(
            "compute-calibration-assurance-v2", "execution_status", "NOT_REQUESTED"
        )
    observations_raw = profile.get("observations")
    result_paths_raw = profile.get("result_paths")
    if not isinstance(observations_raw, list) or not observations_raw:
        raise AccuracyRuntimeError(
            "calibration_profile.observations is required for executable calibration"
        )
    if not isinstance(result_paths_raw, list) or not result_paths_raw:
        raise AccuracyRuntimeError(
            "calibration_profile.result_paths is required for executable calibration"
        )
    observations = _finite_vector(observations_raw, "calibration observations")
    result_paths = [str(item) for item in result_paths_raw]
    parameter_paths = _parameter_paths(profile)
    model = _calibration_model(
        handler,
        inputs,
        parameter_paths,
        result_paths,
        len(observations),
    )
    starts = max(1, int(profile.get("multi_start", 1)))
    candidates = _calibration_candidates(model, observations, profile, starts)
    successful = _successful_candidates(candidates)
    if not successful:
        raise AccuracyRuntimeError("all calibration starts failed")
    selected = min(
        successful,
        key=lambda row: (float(row["objective_value"]), int(row["start_index"])),
    )
    calibrated_inputs = copy.deepcopy(dict(inputs))
    for name, value in selected["parameters"].items():
        _deep_set(calibrated_inputs, parameter_paths[name], float(value))
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
        "validation": _calibration_validation(
            model, selected["parameters"], profile
        ),
        "input_snapshot_sha256": profile.get("input_snapshot_sha256"),
        "candidate_results": candidates,
    }
    assurance["assurance_sha256"] = _sha(assurance)
    return calibrated_inputs, assurance


def _experiment_rows(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    inputs: Mapping[str, Any],
    replications: int,
    base_seed: Any,
    seed_path: str,
    result_paths: list[str],
) -> tuple[list[list[float]], list[int | None]]:
    rows: list[list[float]] = []
    seeds: list[int | None] = []
    for index in range(replications):
        candidate = copy.deepcopy(dict(inputs))
        seed = None if base_seed is None else (int(base_seed) + index) % (2**32)
        if seed is not None:
            _deep_set(candidate, seed_path, seed)
        result = handler(candidate)
        rows.append(_extract(result, result_paths) if result_paths else [])
        seeds.append(seed)
    return rows, seeds


def _experiment_metrics(rows: list[list[float]], replications: int) -> list[dict[str, Any]]:
    matrix = (
        np.asarray(rows, dtype=float)
        if rows and rows[0]
        else np.empty((replications, 0))
    )
    metrics: list[dict[str, Any]] = []
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        mcse = standard_deviation / math.sqrt(values.size) if values.size else 0.0
        mean = float(np.mean(values))
        metrics.append(
            {
                "metric_index": column,
                "mean": mean,
                "median": float(np.median(values)),
                "standard_deviation": standard_deviation,
                "monte_carlo_standard_error": mcse,
                "confidence_interval_95": [mean - 1.96 * mcse, mean + 1.96 * mcse],
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return metrics


def execute_experiment(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    inputs: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if profile is None:
        return _not_requested(
            "compute-experiment-execution-v1", "status", "NOT_REQUESTED"
        )
    replications = int(profile.get("replications", 1))
    base_seed = profile.get("base_seed")
    result_paths_raw = profile.get("result_paths")
    if replications > 1 and base_seed is None:
        raise AccuracyRuntimeError(
            "experiment_profile.base_seed is required when replications exceed one"
        )
    if replications > 1 and (
        not isinstance(result_paths_raw, list) or not result_paths_raw
    ):
        raise AccuracyRuntimeError(
            "experiment_profile.result_paths is required for replicated accuracy evidence"
        )
    result_paths = [str(item) for item in result_paths_raw or []]
    rows, seeds = _experiment_rows(
        handler,
        inputs,
        replications,
        base_seed,
        str(profile.get("seed_path") or "seed"),
        result_paths,
    )
    metrics = _experiment_metrics(rows, replications)
    target = profile.get("precision_target")
    precision_pass = (
        True
        if target is None
        else bool(
            metrics
            and all(
                row["monte_carlo_standard_error"] <= float(target)
                for row in metrics
            )
        )
    )
    report = {
        "schema_version": "compute-experiment-execution-v1",
        "requested": True,
        "status": "PASS" if precision_pass else "FAIL",
        "requested_replications": replications,
        "executed_replications": len(rows),
        "seeds": seeds,
        "result_paths": result_paths,
        "metrics": metrics,
        "precision_target": target,
        "precision_rule_passed": precision_pass,
        "stopping_rule": profile.get("stopping_rule"),
    }
    report["execution_sha256"] = _sha(report)
    return report


def _validation_predictions(
    predicted: list[float], profile: Mapping[str, Any]
) -> dict[str, Sequence[float]]:
    predictions: dict[str, Sequence[float]] = {"candidate": predicted}
    baseline = profile.get("baseline_predictions")
    if isinstance(baseline, list):
        predictions[str(profile.get("baseline_model_id") or "baseline")] = baseline
    alternatives = profile.get("alternative_predictions")
    if isinstance(alternatives, Mapping):
        predictions.update(
            {
                str(key): value
                for key, value in alternatives.items()
                if isinstance(value, list)
            }
        )
    return predictions


def _model_comparison(
    actual: list[float],
    predictions: Mapping[str, Sequence[float]],
    profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    if len(predictions) <= 1:
        return None
    baseline_id = str(profile.get("baseline_model_id") or "baseline")
    if baseline_id not in predictions:
        raise AccuracyRuntimeError(
            "baseline_model_id does not match supplied baseline_predictions"
        )
    return compare_models(
        actual,
        predictions,
        baseline_model_id=baseline_id,
        complexity=profile.get("model_complexity"),
        minimum_improvement_over_baseline=float(
            profile.get("minimum_improvement_over_baseline", 0.0)
        ),
    )


def _ensemble_result(
    actual: list[float],
    predictions: Mapping[str, Sequence[float]],
    profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not profile.get("ensemble_method") or len(predictions) <= 1:
        return None
    actual_array = np.asarray(actual, dtype=float)
    errors = {
        name: float(
            np.sqrt(
                np.mean((np.asarray(values, dtype=float) - actual_array) ** 2)
            )
        )
        for name, values in predictions.items()
    }
    return ensemble_predictions(
        predictions,
        method=str(profile["ensemble_method"]),
        validation_errors=errors,
        maximum_weight=float(profile.get("maximum_ensemble_weight", 0.7)),
    )


def _threshold_failures(
    residual: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
) -> list[str]:
    thresholds = profile.get("metric_thresholds")
    thresholds = thresholds if isinstance(thresholds, Mapping) else {}
    diagnostics = residual["diagnostics"]
    failures: list[str] = []
    if (
        thresholds.get("rmse_maximum") is not None
        and float(diagnostics["rmse"]) > float(thresholds["rmse_maximum"])
    ):
        failures.append("RMSE_THRESHOLD_FAILED")
    if (
        thresholds.get("mae_maximum") is not None
        and float(diagnostics["mae"]) > float(thresholds["mae_maximum"])
    ):
        failures.append("MAE_THRESHOLD_FAILED")
    if comparison and comparison.get("selected_model_id") != "candidate":
        failures.append("CANDIDATE_DID_NOT_BEAT_BASELINE")
    return failures


def execute_validation(
    result: Mapping[str, Any], profile: Mapping[str, Any] | None
) -> dict[str, Any]:
    if profile is None:
        return _not_requested(
            "compute-validation-result-v1", "status", "NOT_REQUESTED"
        )
    actual_raw = profile.get("actual_values")
    result_paths_raw = profile.get("result_paths")
    if (
        not isinstance(actual_raw, list)
        or not actual_raw
        or not isinstance(result_paths_raw, list)
        or not result_paths_raw
    ):
        return {
            "schema_version": "compute-validation-result-v1",
            "requested": True,
            "status": "NOT_EXECUTED",
            "reason": (
                "actual_values and result_paths are required for executable validation"
            ),
        }
    actual = _finite_vector(actual_raw, "validation actual_values")
    predicted = _extract(result, [str(item) for item in result_paths_raw])
    if len(actual) != len(predicted):
        raise AccuracyRuntimeError(
            "validation actual and prediction lengths do not match"
        )
    residual = diagnose_residuals(
        actual,
        predicted,
        time_index=profile.get("time_index"),
        groups=profile.get("groups"),
    )
    predictions = _validation_predictions(predicted, profile)
    comparison = _model_comparison(actual, predictions, profile)
    failures = _threshold_failures(residual, comparison, profile)
    status = "FAIL" if failures else "WARN" if residual["status"] == "WARN" else "PASS"
    report = {
        "schema_version": "compute-validation-result-v1",
        "requested": True,
        "status": status,
        "strategy": profile.get("strategy"),
        "residual_diagnostics": residual,
        "model_comparison": comparison,
        "ensemble": _ensemble_result(actual, predictions, profile),
        "threshold_failures": failures,
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
    benchmark_ids = list(model.get("benchmark_ids") or []) + list(
        quality_profile.get("benchmark_ids") or []
    )
    calibration_needed = bool(model.get("calibration_supported"))
    calibration_ok = (
        not calibration_needed or calibration.get("execution_status") == "EXECUTED"
    )
    validation_ok = validation.get("status") == "PASS"
    replication_ok = experiment.get("status") in {"PASS", "NOT_REQUESTED"}
    feedback_present = bool(
        quality_profile.get("operational_feedback_evidence_sha256")
    )
    technical_review = bool(quality_profile.get("technical_review_evidence_sha256"))
    if (
        validation_ok
        and calibration_ok
        and replication_ok
        and benchmark_ids
        and feedback_present
        and technical_review
    ):
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
