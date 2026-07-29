#!/usr/bin/env python3
"""Bounded policy-causal evaluation using the pinned DoWhy method pack.

The operation requires explicit treatment, outcome and adjustment variables. It separates
identification, estimation and refutation, and downgrades failed diagnostics to association.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
from scipy import optimize, stats

from compute_runner import ComputeError

EXPECTED_DOWHY = "0.14"
MAX_ROWS = 100_000
MAX_CONFOUNDERS = 30
MAX_REFUTATIONS = 2_000
MODES = {
    "backdoor_adjustment",
    "propensity_weighting",
    "difference_in_differences_refuted",
    "instrumental_variable_refuted",
    "placebo_policy_test",
    "unobserved_confounding_sensitivity",
}


def _dependencies():
    try:
        import pandas as pd
        from dowhy import CausalModel
    except ImportError as exc:
        raise ComputeError("causal engine is not installed; install requirements-causal.txt") from exc
    if version("dowhy") != EXPECTED_DOWHY:
        raise ComputeError(f"DoWhy version must be exactly {EXPECTED_DOWHY}")
    return pd, CausalModel


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _vector(value: Any, name: str, *, binary: bool = False) -> np.ndarray:
    raw = _sequence(value, name)
    if not 2 <= len(raw) <= MAX_ROWS:
        raise ComputeError(f"{name} must contain 2 to {MAX_ROWS} values")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ComputeError(f"{name} must be a finite one-dimensional array")
    if binary and np.any((array != 0) & (array != 1)):
        raise ComputeError(f"{name} must contain only 0 and 1")
    return array


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _matrix(confounders: Mapping[str, Any] | None, size: int) -> tuple[list[str], np.ndarray]:
    if not confounders:
        return [], np.empty((size, 0), dtype=float)
    if len(confounders) > MAX_CONFOUNDERS:
        raise ComputeError(f"confounders cannot exceed {MAX_CONFOUNDERS}")
    names = sorted(str(name) for name in confounders)
    columns = []
    for name in names:
        values = _vector(confounders[name], f"inputs.confounders.{name}")
        if values.size != size:
            raise ComputeError("all confounders must match treatment length")
        columns.append(values)
    return names, np.column_stack(columns)


def _ols(y: np.ndarray, columns: list[np.ndarray]) -> dict[str, Any]:
    design = np.column_stack([np.ones(y.size), *columns])
    if design.shape[0] <= design.shape[1]:
        raise ComputeError("insufficient observations for regression design")
    coefficient, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficient
    residual = y - fitted
    degrees = max(1, y.size - design.shape[1])
    variance = float(np.sum(residual**2) / degrees)
    covariance = np.linalg.pinv(design.T @ design) * variance
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    t_values = np.divide(coefficient, standard_errors, out=np.zeros_like(coefficient), where=standard_errors > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=degrees)
    return {"coefficient": coefficient, "standard_error": standard_errors, "p_value": p_values, "fitted": fitted, "residual": residual, "rank": int(rank), "r_squared": float(1 - np.sum(residual**2) / max(np.sum((y - np.mean(y)) ** 2), 1e-15))}


def _causal_frame(inputs: Mapping[str, Any], *, binary_treatment: bool = True):
    pd, CausalModel = _dependencies()
    treatment = _vector(inputs.get("treatment"), "inputs.treatment", binary=binary_treatment)
    outcome = _vector(inputs.get("outcome"), "inputs.outcome")
    if treatment.shape != outcome.shape:
        raise ComputeError("treatment and outcome must have equal length")
    names, matrix = _matrix(inputs.get("confounders") if isinstance(inputs.get("confounders"), Mapping) else None, treatment.size)
    data = {"treatment": treatment, "outcome": outcome}
    for index, name in enumerate(names):
        data[name] = matrix[:, index]
    frame = pd.DataFrame(data)
    model = CausalModel(data=frame, treatment="treatment", outcome="outcome", common_causes=names)
    try:
        estimand = model.identify_effect(proceed_when_unidentifiable=False)
    except Exception as exc:
        raise ComputeError(f"causal effect is not identifiable: {type(exc).__name__}: {exc}") from exc
    return frame, model, estimand, treatment, outcome, names, matrix


def _backdoor(inputs: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    frame, model, estimand, treatment, outcome, names, matrix = _causal_frame(inputs)
    try:
        estimate = model.estimate_effect(estimand, method_name=method, control_value=0, treatment_value=1, target_units="ate", test_significance=True)
        effect = float(np.asarray(estimate.value).reshape(-1)[0])
    except Exception as exc:
        raise ComputeError(f"DoWhy estimation failed: {type(exc).__name__}: {exc}") from exc
    treated = int(np.sum(treatment == 1)); control = int(np.sum(treatment == 0))
    overlap = {"treated_count": treated, "control_count": control, "both_groups_present": treated > 0 and control > 0}
    claim_allowed = overlap["both_groups_present"] and bool(names)
    return {"effect": effect, "identified": True, "estimand": str(estimand), "method": method, "confounders": names, "overlap": overlap, "causal_claim_allowed": claim_allowed, "claim_type": "causal_effect" if claim_allowed else "conditional_association", "observation_count": int(frame.shape[0])}


def _propensity_weighting(inputs: Mapping[str, Any]) -> dict[str, Any]:
    frame, model, estimand, treatment, outcome, names, matrix = _causal_frame(inputs)
    if not names:
        raise ComputeError("propensity weighting requires at least one declared confounder")
    design = np.column_stack([np.ones(treatment.size), matrix])

    def objective(coefficient: np.ndarray) -> float:
        linear = np.clip(design @ coefficient, -40, 40)
        probability = 1 / (1 + np.exp(-linear))
        probability = np.clip(probability, 1e-9, 1 - 1e-9)
        return float(-np.sum(treatment * np.log(probability) + (1 - treatment) * np.log(1 - probability)) + 1e-6 * np.sum(coefficient[1:] ** 2))

    fitted = optimize.minimize(objective, np.zeros(design.shape[1]), method="BFGS")
    if not fitted.success and not np.isfinite(fitted.fun):
        raise ComputeError(f"propensity model failed: {fitted.message}")
    propensity = 1 / (1 + np.exp(-np.clip(design @ fitted.x, -40, 40)))
    clip = _finite(inputs.get("propensity_clip", 0.01), "inputs.propensity_clip")
    if not 0 < clip < 0.5:
        raise ComputeError("propensity_clip must be within (0,0.5)")
    propensity = np.clip(propensity, clip, 1 - clip)
    weights_t = treatment / propensity
    weights_c = (1 - treatment) / (1 - propensity)
    effect = float(np.sum(weights_t * outcome) / np.sum(weights_t) - np.sum(weights_c * outcome) / np.sum(weights_c))
    overlap_ok = float(np.min(propensity)) <= 0.1 and float(np.max(propensity)) >= 0.9 or (float(np.min(propensity)) < 0.4 and float(np.max(propensity)) > 0.6)
    effective_sample = float((np.sum(weights_t + weights_c) ** 2) / np.sum((weights_t + weights_c) ** 2))
    try:
        dowhy_estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting", target_units="ate")
        dowhy_effect = float(np.asarray(dowhy_estimate.value).reshape(-1)[0])
    except Exception:
        dowhy_effect = None
    return {"effect": effect, "dowhy_effect": dowhy_effect, "identified": True, "confounders": names, "propensity": {"minimum": float(np.min(propensity)), "maximum": float(np.max(propensity)), "mean": float(np.mean(propensity)), "clip": clip, "effective_sample_size": effective_sample}, "overlap_passed": overlap_ok, "causal_claim_allowed": overlap_ok, "claim_type": "causal_effect" if overlap_ok else "conditional_association", "observation_count": int(frame.shape[0])}


def _did(inputs: Mapping[str, Any]) -> dict[str, Any]:
    treated_pre = _vector(inputs.get("treated_pre"), "inputs.treated_pre")
    treated_post = _vector(inputs.get("treated_post"), "inputs.treated_post")
    control_pre = _vector(inputs.get("control_pre"), "inputs.control_pre")
    control_post = _vector(inputs.get("control_post"), "inputs.control_post")
    if not (treated_pre.size == treated_post.size == control_pre.size == control_post.size):
        raise ComputeError("DID arrays must have equal length")
    effect = float((np.mean(treated_post) - np.mean(treated_pre)) - (np.mean(control_post) - np.mean(control_pre)))
    pre_gap = treated_pre - control_pre
    slope = float(stats.linregress(np.arange(pre_gap.size), pre_gap).slope) if pre_gap.size >= 3 else 0.0
    scale = max(float(np.std(pre_gap, ddof=1)) if pre_gap.size > 1 else 0.0, 1e-12)
    normalized_slope = abs(slope) / scale
    parallel_passed = normalized_slope <= _finite(inputs.get("pretrend_tolerance", 0.25), "inputs.pretrend_tolerance")
    placebo_effect = None
    if pre_gap.size >= 4:
        split = pre_gap.size // 2
        placebo_effect = float(np.mean(pre_gap[split:]) - np.mean(pre_gap[:split]))
    return {"effect": effect, "pretrend_slope": slope, "normalized_pretrend_slope": normalized_slope, "parallel_trends_passed": parallel_passed, "placebo_preperiod_effect": placebo_effect, "causal_claim_allowed": parallel_passed, "claim_type": "causal_effect" if parallel_passed else "association_only"}


def _iv(inputs: Mapping[str, Any]) -> dict[str, Any]:
    instrument = _vector(inputs.get("instrument"), "inputs.instrument")
    treatment = _vector(inputs.get("treatment"), "inputs.treatment")
    outcome = _vector(inputs.get("outcome"), "inputs.outcome")
    if not (instrument.size == treatment.size == outcome.size):
        raise ComputeError("instrument, treatment and outcome must have equal length")
    names, controls = _matrix(inputs.get("controls") if isinstance(inputs.get("controls"), Mapping) else None, instrument.size)
    first = _ols(treatment, [instrument, *[controls[:, index] for index in range(controls.shape[1])]])
    predicted_treatment = first["fitted"]
    second = _ols(outcome, [predicted_treatment, *[controls[:, index] for index in range(controls.shape[1])]])
    instrument_t = float(first["coefficient"][1] / max(first["standard_error"][1], 1e-15))
    first_stage_f = instrument_t**2
    threshold = _finite(inputs.get("minimum_first_stage_f", 10.0), "inputs.minimum_first_stage_f")
    weak = first_stage_f < threshold
    return {"effect": float(second["coefficient"][1]), "standard_error": float(second["standard_error"][1]), "p_value": float(second["p_value"][1]), "first_stage_f": first_stage_f, "weak_instrument": weak, "controls": names, "causal_claim_allowed": not weak, "claim_type": "causal_effect" if not weak else "association_only"}


def _placebo(inputs: Mapping[str, Any]) -> dict[str, Any]:
    treatment = _vector(inputs.get("treatment"), "inputs.treatment", binary=True)
    outcome = _vector(inputs.get("outcome"), "inputs.outcome")
    if treatment.shape != outcome.shape:
        raise ComputeError("treatment and outcome must have equal length")
    names, confounders = _matrix(inputs.get("confounders") if isinstance(inputs.get("confounders"), Mapping) else None, treatment.size)
    actual = _ols(outcome, [treatment, *[confounders[:, index] for index in range(confounders.shape[1])]])
    actual_effect = float(actual["coefficient"][1])
    repetitions = _integer(inputs.get("repetitions", 200), "inputs.repetitions", 20, MAX_REFUTATIONS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    placebo = []
    for _ in range(repetitions):
        shuffled = rng.permutation(treatment)
        estimate = _ols(outcome, [shuffled, *[confounders[:, index] for index in range(confounders.shape[1])]])
        placebo.append(float(estimate["coefficient"][1]))
    values = np.asarray(placebo)
    p_value = float((1 + np.sum(np.abs(values) >= abs(actual_effect))) / (repetitions + 1))
    return {"actual_effect": actual_effect, "placebo_mean": float(np.mean(values)), "placebo_standard_deviation": float(np.std(values, ddof=1)), "empirical_p_value": p_value, "refutation_passed": p_value < _finite(inputs.get("alpha", 0.05), "inputs.alpha"), "repetitions": repetitions, "seed": seed, "confounders": names}


def _sensitivity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    estimate = _finite(inputs.get("effect_estimate"), "inputs.effect_estimate")
    standard_error = _finite(inputs.get("standard_error"), "inputs.standard_error")
    if standard_error <= 0:
        raise ComputeError("standard_error must be positive")
    strengths = [_finite(item, "inputs.bias_strengths[]") for item in _sequence(inputs.get("bias_strengths", [0, 0.25, 0.5, 1.0]), "inputs.bias_strengths")]
    if not 1 <= len(strengths) <= 100 or any(item < 0 for item in strengths):
        raise ComputeError("bias_strengths must contain 1 to 100 non-negative values")
    rows = []
    for strength in strengths:
        lower = estimate - strength * standard_error
        upper = estimate + strength * standard_error
        rows.append({"bias_strength": strength, "adjusted_lower": lower, "adjusted_upper": upper, "sign_robust": lower > 0 or upper < 0})
    first_failure = next((row["bias_strength"] for row in rows if not row["sign_robust"]), None)
    return {"effect_estimate": estimate, "standard_error": standard_error, "scenarios": rows, "first_sign_instability_strength": first_failure, "robust_to_all_tested_strengths": first_failure is None}


def causal_policy_evaluation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode not in MODES:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(MODES))}")
    _dependencies()
    if mode == "backdoor_adjustment":
        result = _backdoor(inputs, method="backdoor.linear_regression")
    elif mode == "propensity_weighting":
        result = _propensity_weighting(inputs)
    elif mode == "difference_in_differences_refuted":
        result = _did(inputs)
    elif mode == "instrumental_variable_refuted":
        result = _iv(inputs)
    elif mode == "placebo_policy_test":
        result = _placebo(inputs)
    else:
        result = _sensitivity(inputs)
    return {"engine": {"name": "dowhy-isolated-fixed-adapter", "version": EXPECTED_DOWHY, "network_used": False}, "mode": mode, **result, "interpretation_boundary": "Causal language is permitted only when the mode-specific identification and refutation gates pass."}


OPERATIONS = {"causal_policy_evaluation": causal_policy_evaluation}
