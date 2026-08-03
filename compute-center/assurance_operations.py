#!/usr/bin/env python3
"""Bounded offline forecast assurance, VV&A, and generic state-estimation modes."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError

MAX_ROWS = 20_000
MAX_MODELS = 50
MAX_DIMENSION = 20
MAX_BINS = 50
EPSILON = 1e-12


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ComputeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ComputeError(f"{name} must be an integer") from exc
    if result != value or not minimum <= result <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return result


def _vector(value: Any, name: str, minimum: int = 1, maximum: int = MAX_ROWS) -> np.ndarray:
    rows = _sequence(value, name)
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} values")
    result = np.asarray([_finite(item, f"{name}[]") for item in rows], dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ComputeError(f"{name} must be a finite one-dimensional array")
    return result


def _matrix(value: Any, name: str, minimum_rows: int = 1, maximum_rows: int = MAX_ROWS, maximum_columns: int = MAX_MODELS) -> np.ndarray:
    rows = _sequence(value, name)
    if not minimum_rows <= len(rows) <= maximum_rows:
        raise ComputeError(f"{name} row count is outside the governed range")
    converted: list[np.ndarray] = []
    width: int | None = None
    for index, row in enumerate(rows):
        vector = _vector(row, f"{name}[{index}]", 1, maximum_columns)
        width = vector.size if width is None else width
        if vector.size != width:
            raise ComputeError(f"{name} rows must have equal length")
        converted.append(vector)
    result = np.asarray(converted, dtype=float)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ComputeError(f"{name} must be a finite matrix")
    return result


def _probability_vector(value: Any, name: str) -> np.ndarray:
    result = _vector(value, name)
    if np.any(result < 0) or np.any(result > 1):
        raise ComputeError(f"{name} must contain probabilities between 0 and 1")
    return result


def _binary_outcomes(value: Any, name: str, expected: int) -> np.ndarray:
    rows = _sequence(value, name)
    if len(rows) != expected:
        raise ComputeError(f"{name} length must be {expected}")
    converted = []
    for item in rows:
        if item not in (0, 1, False, True):
            raise ComputeError(f"{name} must contain only binary outcomes")
        converted.append(int(item))
    return np.asarray(converted, dtype=int)


def _base_result(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_used": False,
        "live_feed_used": False,
        "individual_or_target_tracking_allowed": False,
        "decision_support_only": True,
    }


def probabilistic_forecast_scoring(inputs: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = _matrix(inputs.get("probabilities"), "inputs.probabilities", maximum_columns=50)
    outcomes_raw = _sequence(inputs.get("outcomes"), "inputs.outcomes")
    rows, classes = probabilities.shape
    if classes < 2 or len(outcomes_raw) != rows:
        raise ComputeError("probabilities must have at least two classes and match outcomes")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ComputeError("probabilities must be between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ComputeError("each probability row must sum to 1")
    outcomes = np.asarray([_integer(item, "inputs.outcomes[]", 0, classes - 1) for item in outcomes_raw], dtype=int)
    observed = np.zeros_like(probabilities)
    observed[np.arange(rows), outcomes] = 1.0
    brier = float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))
    true_probabilities = probabilities[np.arange(rows), outcomes]
    log_loss = float(-np.mean(np.log(np.clip(true_probabilities, EPSILON, 1.0))))
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == outcomes))
    climatology = np.bincount(outcomes, minlength=classes).astype(float) / rows
    baseline = float(np.mean(np.sum((climatology - observed) ** 2, axis=1)))
    skill = None if baseline <= EPSILON else float(1.0 - brier / baseline)
    result = _base_result("probabilistic_forecast_scoring")
    result.update({"observations": rows, "classes": classes, "brier_score": brier, "log_loss": log_loss, "top_class_accuracy": accuracy, "climatology_brier_score": baseline, "brier_skill_score": skill, "lower_is_better": ["brier_score", "log_loss"]})
    return result


def calibration_diagnostics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = _probability_vector(inputs.get("probabilities"), "inputs.probabilities")
    outcomes = _binary_outcomes(inputs.get("outcomes"), "inputs.outcomes", probabilities.size)
    bins = _integer(inputs.get("bins", 10), "inputs.bins", 2, MAX_BINS)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(probabilities, edges[1:-1], right=True), bins - 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        mask = bucket == index
        count = int(mask.sum())
        if not count:
            continue
        mean_probability = float(probabilities[mask].mean())
        event_rate = float(outcomes[mask].mean())
        gap = abs(mean_probability - event_rate)
        ece += gap * count / probabilities.size
        mce = max(mce, gap)
        rows.append({"bin": index, "count": count, "mean_probability": mean_probability, "event_rate": event_rate, "calibration_gap": gap})
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones_like(logits), logits])
    beta = np.zeros(2, dtype=float)
    for _ in range(50):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -40, 40)))
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        gradient = design.T @ (outcomes - fitted)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    result = _base_result("calibration_diagnostics")
    result.update({"observations": int(probabilities.size), "bins_requested": bins, "bins_populated": len(rows), "expected_calibration_error": float(ece), "maximum_calibration_error": float(mce), "calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1]), "event_rate": float(outcomes.mean()), "reliability_table": rows})
    return result


def prediction_interval_validation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    lower = _vector(inputs.get("lower"), "inputs.lower")
    upper = _vector(inputs.get("upper"), "inputs.upper")
    observed = _vector(inputs.get("observed"), "inputs.observed")
    if not lower.size == upper.size == observed.size:
        raise ComputeError("lower, upper and observed arrays must have equal length")
    if np.any(lower > upper):
        raise ComputeError("lower bounds must not exceed upper bounds")
    alpha = _finite(inputs.get("alpha", 0.1), "inputs.alpha")
    if not 0 < alpha < 1:
        raise ComputeError("alpha must be between 0 and 1")
    inside = (observed >= lower) & (observed <= upper)
    width = upper - lower
    below = observed < lower
    above = observed > upper
    interval_score = width.copy()
    interval_score[below] += (2.0 / alpha) * (lower[below] - observed[below])
    interval_score[above] += (2.0 / alpha) * (observed[above] - upper[above])
    result = _base_result("prediction_interval_validation")
    result.update({"observations": int(observed.size), "nominal_coverage": float(1.0 - alpha), "empirical_coverage": float(inside.mean()), "coverage_error": float(inside.mean() - (1.0 - alpha)), "average_interval_width": float(width.mean()), "median_interval_width": float(np.median(width)), "mean_interval_score": float(interval_score.mean()), "below_interval_rate": float(below.mean()), "above_interval_rate": float(above.mean())})
    return result


def realized_outcome_feedback(inputs: Mapping[str, Any]) -> dict[str, Any]:
    predicted = _vector(inputs.get("predicted"), "inputs.predicted", minimum=4)
    observed = _vector(inputs.get("observed"), "inputs.observed", minimum=4)
    if predicted.size != observed.size:
        raise ComputeError("predicted and observed arrays must have equal length")
    errors = predicted - observed
    absolute = np.abs(errors)
    middle = predicted.size // 2
    early = errors[:middle]
    late = errors[middle:]
    early_mae = float(np.mean(np.abs(early)))
    late_mae = float(np.mean(np.abs(late)))
    drift_ratio = None if early_mae <= EPSILON else float(late_mae / early_mae)
    tolerance = _finite(inputs.get("drift_ratio_threshold", 1.5), "inputs.drift_ratio_threshold")
    if tolerance <= 0:
        raise ComputeError("drift_ratio_threshold must be positive")
    drift = bool(drift_ratio is not None and drift_ratio > tolerance)
    result = _base_result("realized_outcome_feedback")
    result.update({"observations": int(predicted.size), "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))), "bias": float(errors.mean()), "median_absolute_error": float(np.median(absolute)), "early_period_mae": early_mae, "late_period_mae": late_mae, "late_to_early_mae_ratio": drift_ratio, "drift_ratio_threshold": tolerance, "drift_flag": drift, "feedback_status": "REVIEW_REQUIRED" if drift else "WITHIN_THRESHOLD"})
    return result


def benchmark_comparison(inputs: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _sequence(inputs.get("candidates"), "inputs.candidates")
    if not 1 <= len(candidates) <= 200:
        raise ComputeError("candidates must contain 1 to 200 rows")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        row = _mapping(raw, f"inputs.candidates[{index}]")
        name = str(row.get("name") or "").strip()
        if not name or len(name) > 120:
            raise ComputeError("each candidate requires a bounded name")
        observed = _finite(row.get("observed"), f"inputs.candidates[{index}].observed")
        benchmark = _finite(row.get("benchmark"), f"inputs.candidates[{index}].benchmark")
        tolerance = _finite(row.get("tolerance"), f"inputs.candidates[{index}].tolerance")
        if tolerance < 0:
            raise ComputeError("benchmark tolerance must be non-negative")
        direction = str(row.get("direction") or "absolute").strip().lower()
        if direction == "absolute":
            error = abs(observed - benchmark)
            passed = error <= tolerance
        elif direction == "minimum":
            error = max(0.0, benchmark - observed)
            passed = observed + tolerance >= benchmark
        elif direction == "maximum":
            error = max(0.0, observed - benchmark)
            passed = observed - tolerance <= benchmark
        else:
            raise ComputeError("direction must be absolute, minimum or maximum")
        rows.append({"name": name, "observed": observed, "benchmark": benchmark, "tolerance": tolerance, "direction": direction, "error": float(error), "normalized_error": None if tolerance <= EPSILON else float(error / tolerance), "passed": bool(passed)})
    failed = [row["name"] for row in rows if not row["passed"]]
    result = _base_result("benchmark_comparison")
    result.update({"candidate_count": len(rows), "passed": len(rows) - len(failed), "failed": len(failed), "status": "PASS" if not failed else "FAIL", "failed_candidates": failed, "rows": rows})
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def cross_model_agreement(inputs: Mapping[str, Any]) -> dict[str, Any]:
    outputs = _matrix(inputs.get("outputs"), "inputs.outputs", minimum_rows=2, maximum_rows=MAX_MODELS, maximum_columns=MAX_ROWS)
    models, observations = outputs.shape
    names_raw = inputs.get("model_names")
    if names_raw is None:
        names = [f"model_{index + 1}" for index in range(models)]
    else:
        names_seq = _sequence(names_raw, "inputs.model_names")
        if len(names_seq) != models:
            raise ComputeError("model_names length must match outputs rows")
        names = [str(item).strip() for item in names_seq]
        if any(not item or len(item) > 120 for item in names) or len(set(names)) != len(names):
            raise ComputeError("model_names must be unique bounded strings")
    threshold = _finite(inputs.get("minimum_rank_correlation", 0.7), "inputs.minimum_rank_correlation")
    if not -1 <= threshold <= 1:
        raise ComputeError("minimum_rank_correlation must be between -1 and 1")
    pairwise: list[dict[str, Any]] = []
    low_pairs: list[str] = []
    for left in range(models):
        for right in range(left + 1, models):
            left_rank = _rank(outputs[left])
            right_rank = _rank(outputs[right])
            if np.std(left_rank) <= EPSILON or np.std(right_rank) <= EPSILON:
                correlation = 1.0 if np.allclose(outputs[left], outputs[right]) else 0.0
            else:
                correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
            label = f"{names[left]}::{names[right]}"
            if correlation < threshold:
                low_pairs.append(label)
            pairwise.append({"pair": label, "spearman_rank_correlation": correlation})
    consensus = np.median(outputs, axis=0)
    spread = np.ptp(outputs, axis=0)
    result = _base_result("cross_model_agreement")
    result.update({"models": models, "observations": observations, "minimum_rank_correlation": threshold, "pairwise": pairwise, "low_agreement_pairs": low_pairs, "agreement_status": "REVIEW_REQUIRED" if low_pairs else "PASS", "consensus_median": consensus.tolist(), "mean_cross_model_spread": float(spread.mean()), "maximum_cross_model_spread": float(spread.max())})
    return result


def vva_acceptance_gate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    checks_raw = _mapping(inputs.get("checks"), "inputs.checks")
    required = ["code_verification", "numerical_benchmark", "input_data_quality", "assumption_register", "sensitivity_analysis", "uncertainty_analysis", "external_validation", "independent_review", "realized_outcome_feedback"]
    high_risk = bool(inputs.get("high_risk", False))
    accepted_values = {"PASS", "NOT_APPLICABLE"}
    rows: list[dict[str, Any]] = []
    blocking: list[str] = []
    for name in required:
        status = str(checks_raw.get(name, "MISSING")).strip().upper()
        if status not in {"PASS", "FAIL", "MISSING", "NOT_APPLICABLE"}:
            raise ComputeError(f"invalid VV&A status for {name}")
        if high_risk and status == "NOT_APPLICABLE" and name in {"external_validation", "independent_review", "uncertainty_analysis"}:
            status = "MISSING"
        if status not in accepted_values:
            blocking.append(name)
        rows.append({"check": name, "status": status})
    passed = len(required) - len(blocking)
    ratio = passed / len(required)
    maturity = "PRODUCTION_ELIGIBLE" if not blocking else "CONTROLLED_PREVIEW_ONLY" if ratio >= 0.75 else "BLOCKED"
    result = _base_result("vva_acceptance_gate")
    result.update({"high_risk": high_risk, "checks_total": len(required), "checks_accepted": passed, "blocking_checks": blocking, "acceptance_status": "PASS" if not blocking else "FAIL", "recommended_maturity": maturity, "checks": rows, "human_approval_required": bool(high_risk or blocking)})
    return result


def bounded_linear_kalman_filter(inputs: Mapping[str, Any]) -> dict[str, Any]:
    transition = _matrix(inputs.get("transition_matrix"), "inputs.transition_matrix", maximum_rows=MAX_DIMENSION, maximum_columns=MAX_DIMENSION)
    state_dimension = transition.shape[0]
    if transition.shape != (state_dimension, state_dimension):
        raise ComputeError("transition_matrix must be square")
    observation_matrix = _matrix(inputs.get("observation_matrix"), "inputs.observation_matrix", maximum_rows=MAX_DIMENSION, maximum_columns=MAX_DIMENSION)
    observation_dimension = observation_matrix.shape[0]
    if observation_matrix.shape[1] != state_dimension:
        raise ComputeError("observation_matrix column count must match state dimension")
    process_covariance = _matrix(inputs.get("process_covariance"), "inputs.process_covariance", maximum_rows=MAX_DIMENSION, maximum_columns=MAX_DIMENSION)
    observation_covariance = _matrix(inputs.get("observation_covariance"), "inputs.observation_covariance", maximum_rows=MAX_DIMENSION, maximum_columns=MAX_DIMENSION)
    initial_covariance = _matrix(inputs.get("initial_covariance"), "inputs.initial_covariance", maximum_rows=MAX_DIMENSION, maximum_columns=MAX_DIMENSION)
    if process_covariance.shape != (state_dimension, state_dimension) or initial_covariance.shape != (state_dimension, state_dimension):
        raise ComputeError("state covariance matrix has invalid shape")
    if observation_covariance.shape != (observation_dimension, observation_dimension):
        raise ComputeError("observation_covariance has invalid shape")
    initial_state = _vector(inputs.get("initial_state"), "inputs.initial_state", state_dimension, state_dimension)
    observations = _matrix(inputs.get("observations"), "inputs.observations", maximum_rows=10_000, maximum_columns=MAX_DIMENSION)
    if observations.shape[1] != observation_dimension:
        raise ComputeError("observation rows have invalid dimension")
    for name, matrix in (("process_covariance", process_covariance), ("observation_covariance", observation_covariance), ("initial_covariance", initial_covariance)):
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ComputeError(f"{name} must be symmetric")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-8:
            raise ComputeError(f"{name} must be positive semidefinite")
    state = initial_state.copy()
    covariance = initial_covariance.copy()
    identity = np.eye(state_dimension)
    filtered: list[list[float]] = []
    covariance_diagonal: list[list[float]] = []
    innovations: list[float] = []
    for observation in observations:
        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process_covariance
        innovation = observation - observation_matrix @ predicted_state
        innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + observation_covariance
        gain = predicted_covariance @ observation_matrix.T @ np.linalg.pinv(innovation_covariance)
        state = predicted_state + gain @ innovation
        joseph_left = identity - gain @ observation_matrix
        covariance = joseph_left @ predicted_covariance @ joseph_left.T + gain @ observation_covariance @ gain.T
        covariance = (covariance + covariance.T) / 2.0
        filtered.append(state.tolist())
        covariance_diagonal.append(np.diag(covariance).tolist())
        innovations.append(float(np.linalg.norm(innovation)))
    result = _base_result("bounded_linear_kalman_filter")
    result.update({"state_dimension": state_dimension, "observation_dimension": observation_dimension, "steps": int(observations.shape[0]), "filtered_states": filtered, "covariance_diagonal": covariance_diagonal, "mean_innovation_norm": float(np.mean(innovations)), "final_state": state.tolist(), "fixed_offline_generic_state_estimation": True, "restrictions": ["no live feed", "no person identification", "no target designation", "no weapons or autonomous-control integration"]})
    return result


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "probabilistic_forecast_scoring": probabilistic_forecast_scoring,
    "calibration_diagnostics": calibration_diagnostics,
    "prediction_interval_validation": prediction_interval_validation,
    "realized_outcome_feedback": realized_outcome_feedback,
    "benchmark_comparison": benchmark_comparison,
    "cross_model_agreement": cross_model_agreement,
    "vva_acceptance_gate": vva_acceptance_gate,
    "bounded_linear_kalman_filter": bounded_linear_kalman_filter,
}
