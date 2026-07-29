#!/usr/bin/env python3
"""Repository-internal parameter calibration backends.

The public ticket may select only an allowlisted backend and fixed parameters. The callable
model itself must come from repository code; this module never evaluates ticket-supplied code.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import optimize, stats


class CalibrationError(ValueError):
    """Raised for invalid calibration requests or failed optimizers."""


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} must be a non-empty finite one-dimensional array")
    return array


def _parameter_spec(profile: Mapping[str, Any]) -> tuple[list[str], np.ndarray, list[tuple[float, float]]]:
    rows = profile.get("parameters")
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("calibration profile parameters must be a non-empty array")
    names: list[str] = []
    initial: list[float] = []
    bounds: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CalibrationError("parameter entry must be an object")
        name = str(row.get("name") or "")
        if not name or name in names:
            raise CalibrationError(f"duplicate or empty parameter name: {name}")
        lower, upper, start = float(row["minimum"]), float(row["maximum"]), float(row["initial"])
        if not all(math.isfinite(item) for item in (lower, upper, start)) or lower > upper:
            raise CalibrationError(f"invalid bounds for {name}")
        if not lower <= start <= upper:
            raise CalibrationError(f"initial value is outside bounds for {name}")
        names.append(name); initial.append(start); bounds.append((lower, upper))
    return names, np.asarray(initial, dtype=float), bounds


def _params(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _predictions(model: Callable[[Mapping[str, float]], Sequence[float]], names: Sequence[str], values: Sequence[float], expected_size: int) -> np.ndarray:
    prediction = np.asarray(model(_params(names, values)), dtype=float)
    if prediction.ndim != 1 or prediction.size != expected_size or not np.all(np.isfinite(prediction)):
        raise CalibrationError("model must return a finite vector matching observations")
    return prediction


def _negative_log_likelihood(observations: np.ndarray, predictions: np.ndarray, likelihood: str, options: Mapping[str, Any]) -> float:
    epsilon = 1e-12
    if likelihood == "bernoulli":
        if np.any((observations != 0) & (observations != 1)):
            raise CalibrationError("bernoulli observations must contain only 0 or 1")
        p = np.clip(predictions, epsilon, 1 - epsilon)
        return float(-np.sum(observations * np.log(p) + (1 - observations) * np.log(1 - p)))
    if likelihood == "binomial":
        trials = float(options.get("trials", 0))
        if trials <= 0 or np.any(observations < 0) or np.any(observations > trials):
            raise CalibrationError("binomial requires positive trials and valid observed counts")
        return float(-np.sum(stats.binom.logpmf(observations, trials, np.clip(predictions, epsilon, 1 - epsilon))))
    if likelihood == "poisson":
        if np.any(observations < 0):
            raise CalibrationError("poisson observations must be nonnegative")
        return float(-np.sum(stats.poisson.logpmf(observations, np.clip(predictions, epsilon, None))))
    if likelihood == "negative_binomial":
        dispersion = float(options.get("dispersion", 0))
        if dispersion <= 0 or np.any(observations < 0):
            raise CalibrationError("negative_binomial requires positive dispersion")
        mean = np.clip(predictions, epsilon, None)
        probability = dispersion / (dispersion + mean)
        return float(-np.sum(stats.nbinom.logpmf(observations, dispersion, probability)))
    if likelihood == "normal":
        scale = float(options.get("scale", 0))
        if scale <= 0:
            raise CalibrationError("normal likelihood requires positive scale")
        return float(-np.sum(stats.norm.logpdf(observations, loc=predictions, scale=scale)))
    if likelihood == "lognormal":
        scale = float(options.get("scale", 0))
        if scale <= 0 or np.any(observations <= 0):
            raise CalibrationError("lognormal requires positive scale and observations")
        return float(-np.sum(stats.lognorm.logpdf(observations, s=scale, scale=np.exp(predictions))))
    if likelihood == "gamma":
        shape = float(options.get("shape", 0))
        if shape <= 0 or np.any(observations <= 0):
            raise CalibrationError("gamma requires positive shape and observations")
        mean = np.clip(predictions, epsilon, None)
        return float(-np.sum(stats.gamma.logpdf(observations, a=shape, scale=mean / shape)))
    raise CalibrationError(f"unsupported likelihood: {likelihood}")


def _covariance_from_jacobian(jacobian: np.ndarray, residuals: np.ndarray) -> np.ndarray | None:
    if jacobian.ndim != 2 or jacobian.shape[0] <= jacobian.shape[1]:
        return None
    try:
        inverse = np.linalg.pinv(jacobian.T @ jacobian)
        variance = float(np.sum(residuals ** 2) / max(1, jacobian.shape[0] - jacobian.shape[1]))
        covariance = inverse * variance
        return covariance if np.all(np.isfinite(covariance)) else None
    except np.linalg.LinAlgError:
        return None


def calibrate(model: Callable[[Mapping[str, float]], Sequence[float]], observations: Sequence[float], profile: Mapping[str, Any], *, weights: Sequence[float] | None = None, scipy_constraints: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    observed = _array(observations, "observations")
    names, initial, bounds = _parameter_spec(profile)
    weight_array = np.ones_like(observed) if weights is None else _array(weights, "weights")
    if weight_array.shape != observed.shape or np.any(weight_array <= 0):
        raise CalibrationError("weights must match observations and be positive")
    backend = str(profile.get("backend") or "")
    max_iterations = int(profile.get("max_iterations", 2000))
    loss = str(profile.get("loss") or "linear")
    scipy_constraints = list(scipy_constraints or [])
    evaluation_count = 0

    def residual(values: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        evaluation_count += 1
        return (_predictions(model, names, values, observed.size) - observed) * np.sqrt(weight_array)

    def objective(values: np.ndarray) -> float:
        values_residual = residual(values)
        return float(np.mean(np.abs(values_residual))) if profile.get("objective") == "mae" else float(np.mean(values_residual ** 2))

    jacobian = None
    if backend == "least_squares":
        result = optimize.least_squares(residual, initial, bounds=(np.asarray([item[0] for item in bounds]), np.asarray([item[1] for item in bounds])), loss=loss, max_nfev=max_iterations)
        values, success, message = result.x, bool(result.success), str(result.message)
        objective_value = float(np.mean(residual(values) ** 2)); jacobian = np.asarray(result.jac, dtype=float)
    elif backend == "slsqp":
        result = optimize.minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=scipy_constraints, options={"maxiter": max_iterations, "ftol": 1e-12})
        values, success, message, objective_value = result.x, bool(result.success), str(result.message), float(result.fun)
    elif backend == "differential_evolution":
        result = optimize.differential_evolution(objective, bounds, constraints=tuple(scipy_constraints), seed=int(profile.get("seed", 0)), maxiter=max_iterations, polish=True, updating="immediate")
        values, success, message, objective_value = result.x, bool(result.success), str(result.message), float(result.fun)
    elif backend == "likelihood":
        likelihood = str(profile.get("likelihood") or "")
        options = profile.get("likelihood_options") if isinstance(profile.get("likelihood_options"), Mapping) else {}
        def likelihood_objective(values: np.ndarray) -> float:
            nonlocal evaluation_count
            evaluation_count += 1
            return _negative_log_likelihood(observed, _predictions(model, names, values, observed.size), likelihood, options)
        result = optimize.minimize(likelihood_objective, initial, method="SLSQP", bounds=bounds, constraints=scipy_constraints, options={"maxiter": max_iterations, "ftol": 1e-12})
        values, success, message, objective_value = result.x, bool(result.success), str(result.message), float(result.fun)
    else:
        raise CalibrationError(f"unsupported calibration backend: {backend}")
    if not success or not np.all(np.isfinite(values)):
        raise CalibrationError(f"calibration failed: {message}")
    predictions = _predictions(model, names, values, observed.size)
    raw_residuals = predictions - observed
    covariance = _covariance_from_jacobian(jacobian, raw_residuals) if jacobian is not None else None
    intervals: dict[str, list[float]] = {}
    identifiable = covariance is not None
    if covariance is not None:
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        intervals = {name: [float(value - 1.96 * error), float(value + 1.96 * error)] for name, value, error in zip(names, values, standard_errors, strict=True)}
        identifiable = bool(np.all(np.isfinite(standard_errors)) and np.linalg.matrix_rank(covariance) == len(names))
    return {"schema_version": "compute-calibration-result-v1", "backend": backend, "success": True, "message": message, "objective_value": objective_value, "parameters": _params(names, values), "parameter_intervals_95": intervals, "parameter_identifiable": identifiable, "parameter_non_identifiable": not identifiable, "metrics": {"rmse": float(np.sqrt(np.mean(raw_residuals ** 2))), "mae": float(np.mean(np.abs(raw_residuals))), "bias": float(np.mean(raw_residuals))}, "evaluation_count": evaluation_count, "observation_count": int(observed.size)}
