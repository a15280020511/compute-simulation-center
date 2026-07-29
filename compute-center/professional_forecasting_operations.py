#!/usr/bin/env python3
"""Professional bounded forecasting and global-sensitivity modes."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError

MAX_OBSERVATIONS = 20_000
MAX_SERIES = 20
MAX_HORIZON = 3650
MAX_PARAMETERS = 30


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
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _series(value: Any, name: str, minimum: int = 12) -> np.ndarray:
    result = np.asarray([_finite(item, f"{name}[{i}]") for i, item in enumerate(_sequence(value, name))], dtype=float)
    if not minimum <= result.size <= MAX_OBSERVATIONS:
        raise ComputeError(f"{name} must contain {minimum} to {MAX_OBSERVATIONS} values")
    return result


def _forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residual = actual - predicted
    nonzero = np.abs(actual) > 1e-12
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mape": None if not np.any(nonzero) else float(np.mean(np.abs(residual[nonzero] / actual[nonzero]))),
        "bias": float(np.mean(predicted - actual)),
    }


def sarimax_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ComputeError("forecasting optional dependency statsmodels is not installed") from exc
    data = _series(inputs.get("data"), "inputs.data", 24)
    horizon = _integer(inputs.get("horizon", 6), "inputs.horizon", 1, min(MAX_HORIZON, data.size))
    holdout = _integer(inputs.get("holdout", min(horizon, max(3, data.size // 5))), "inputs.holdout", 1, data.size // 2)
    order_raw = _sequence(inputs.get("order", [1, 1, 1]), "inputs.order")
    seasonal_raw = _sequence(inputs.get("seasonal_order", [0, 0, 0, 0]), "inputs.seasonal_order")
    if len(order_raw) != 3 or len(seasonal_raw) != 4:
        raise ComputeError("order must have 3 and seasonal_order 4 integers")
    order = tuple(_integer(value, f"inputs.order[{i}]", 0, 10) for i, value in enumerate(order_raw))
    seasonal = tuple(_integer(value, f"inputs.seasonal_order[{i}]", 0, 365) for i, value in enumerate(seasonal_raw))
    if seasonal[3] not in {0} and seasonal[3] < 2:
        raise ComputeError("seasonal period must be 0 or at least 2")
    trend = str(inputs.get("trend") or "c")
    if trend not in {"n", "c", "t", "ct"}:
        raise ComputeError("inputs.trend must be n, c, t, or ct")
    train = data[:-holdout]
    validation = data[-holdout:]
    try:
        validation_fit = SARIMAX(train, order=order, seasonal_order=seasonal, trend=trend, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
        validation_forecast = np.asarray(validation_fit.forecast(steps=holdout), dtype=float)
        full_fit = SARIMAX(data, order=order, seasonal_order=seasonal, trend=trend, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
        prediction = full_fit.get_forecast(steps=horizon)
    except Exception as exc:
        raise ComputeError(f"SARIMAX fitting failed: {type(exc).__name__}: {exc}") from exc
    forecast = np.asarray(prediction.predicted_mean, dtype=float)
    interval = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
    if forecast.size != horizon or interval.shape != (horizon, 2) or not np.isfinite(forecast).all():
        raise ComputeError("SARIMAX produced invalid forecast output")
    return {
        "mode": "sarimax_forecast",
        "order": list(order),
        "seasonal_order": list(seasonal),
        "trend": trend,
        "holdout_metrics": _forecast_metrics(validation, validation_forecast),
        "forecast": [float(item) for item in forecast],
        "prediction_interval_95": [[float(row[0]), float(row[1])] for row in interval],
        "aic": float(full_fit.aic),
        "bic": float(full_fit.bic),
        "converged": bool(getattr(full_fit, "mle_retvals", {}).get("converged", True)),
        "decision_support_only": True,
    }


def exponential_smoothing_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as exc:
        raise ComputeError("forecasting optional dependency statsmodels is not installed") from exc
    data = _series(inputs.get("data"), "inputs.data", 12)
    horizon = _integer(inputs.get("horizon", 6), "inputs.horizon", 1, MAX_HORIZON)
    holdout = _integer(inputs.get("holdout", min(horizon, max(3, data.size // 5))), "inputs.holdout", 1, data.size // 2)
    trend = inputs.get("trend", "add")
    seasonal = inputs.get("seasonal")
    trend = None if trend in {None, "none"} else str(trend)
    seasonal = None if seasonal in {None, "none"} else str(seasonal)
    if trend not in {None, "add", "mul"} or seasonal not in {None, "add", "mul"}:
        raise ComputeError("trend and seasonal must be none, add, or mul")
    seasonal_periods = inputs.get("seasonal_periods")
    if seasonal is not None:
        seasonal_periods = _integer(seasonal_periods, "inputs.seasonal_periods", 2, min(365, data.size // 2))
    damped = bool(inputs.get("damped_trend", False))
    train = data[:-holdout]
    validation = data[-holdout:]
    try:
        validation_fit = ExponentialSmoothing(train, trend=trend, damped_trend=damped if trend is not None else False, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method="estimated").fit(optimized=True)
        validation_forecast = np.asarray(validation_fit.forecast(holdout), dtype=float)
        full_fit = ExponentialSmoothing(data, trend=trend, damped_trend=damped if trend is not None else False, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method="estimated").fit(optimized=True)
        forecast = np.asarray(full_fit.forecast(horizon), dtype=float)
    except Exception as exc:
        raise ComputeError(f"exponential smoothing failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "exponential_smoothing_forecast",
        "trend": trend or "none",
        "seasonal": seasonal or "none",
        "seasonal_periods": seasonal_periods,
        "damped_trend": damped,
        "holdout_metrics": _forecast_metrics(validation, validation_forecast),
        "forecast": [float(item) for item in forecast],
        "sse": float(full_fit.sse),
        "decision_support_only": True,
    }


def vector_autoregression_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from statsmodels.tsa.api import VAR
    except ImportError as exc:
        raise ComputeError("forecasting optional dependency statsmodels is not installed") from exc
    raw = _mapping(inputs.get("series"), "inputs.series")
    names = [str(name) for name in raw]
    if not 2 <= len(names) <= MAX_SERIES or any(not name for name in names) or len(set(names)) != len(names):
        raise ComputeError(f"inputs.series must contain 2 to {MAX_SERIES} unique series")
    arrays = [_series(raw[name], f"inputs.series[{name}]", 30) for name in names]
    if len({array.size for array in arrays}) != 1:
        raise ComputeError("all series must have equal length")
    data = np.column_stack(arrays)
    horizon = _integer(inputs.get("horizon", 5), "inputs.horizon", 1, min(MAX_HORIZON, data.shape[0] // 2))
    holdout = _integer(inputs.get("holdout", horizon), "inputs.holdout", 1, data.shape[0] // 3)
    maxlags = _integer(inputs.get("max_lags", min(5, data.shape[0] // 10)), "inputs.max_lags", 1, min(20, data.shape[0] // 4))
    criterion = str(inputs.get("information_criterion") or "aic")
    if criterion not in {"aic", "bic", "hqic", "fpe"}:
        raise ComputeError("information_criterion must be aic, bic, hqic, or fpe")
    train = data[:-holdout]
    validation = data[-holdout:]
    try:
        selected = VAR(train).select_order(maxlags=maxlags)
        lag = int(getattr(selected, criterion) or 1)
        lag = max(1, min(lag, maxlags))
        validation_fit = VAR(train).fit(lag)
        validation_forecast = np.asarray(validation_fit.forecast(train[-lag:], steps=holdout), dtype=float)
        full_fit = VAR(data).fit(lag)
        forecast = np.asarray(full_fit.forecast(data[-lag:], steps=horizon), dtype=float)
    except Exception as exc:
        raise ComputeError(f"VAR fitting failed: {type(exc).__name__}: {exc}") from exc
    metrics = {names[i]: _forecast_metrics(validation[:, i], validation_forecast[:, i]) for i in range(len(names))}
    return {
        "mode": "vector_autoregression_forecast",
        "selected_lags": lag,
        "information_criterion": criterion,
        "holdout_metrics": metrics,
        "forecast": [{names[column]: float(forecast[row, column]) for column in range(len(names))} for row in range(horizon)],
        "stable": bool(full_fit.is_stable(verbose=False)),
        "decision_support_only": True,
    }


def _fixed_model(samples: np.ndarray, names: list[str], spec: Mapping[str, Any]) -> np.ndarray:
    intercept = _finite(spec.get("intercept", 0.0), "inputs.model.intercept")
    linear_raw = _mapping(spec.get("linear", {}), "inputs.model.linear")
    quadratic_raw = _mapping(spec.get("quadratic", {}), "inputs.model.quadratic")
    interactions_raw = _sequence(spec.get("interactions", []), "inputs.model.interactions")
    unknown = (set(linear_raw) | set(quadratic_raw)) - set(names)
    if unknown:
        raise ComputeError(f"model references unknown parameters: {sorted(unknown)}")
    outcome = np.full(samples.shape[0], intercept, dtype=float)
    index = {name: i for i, name in enumerate(names)}
    for name, coefficient in linear_raw.items():
        outcome += _finite(coefficient, f"inputs.model.linear[{name}]") * samples[:, index[str(name)]]
    for name, coefficient in quadratic_raw.items():
        outcome += _finite(coefficient, f"inputs.model.quadratic[{name}]") * samples[:, index[str(name)]] ** 2
    if len(interactions_raw) > 200:
        raise ComputeError("inputs.model.interactions must contain at most 200 entries")
    for row_index, raw in enumerate(interactions_raw):
        row = _mapping(raw, f"inputs.model.interactions[{row_index}]")
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        if left not in index or right not in index:
            raise ComputeError("interaction references unknown parameter")
        outcome += _finite(row.get("coefficient"), f"interaction[{row_index}].coefficient") * samples[:, index[left]] * samples[:, index[right]]
    if not np.isfinite(outcome).all():
        raise ComputeError("fixed sensitivity model produced non-finite output")
    return outcome


def sobol_sensitivity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from SALib.analyze import sobol
        from SALib.sample import sobol as sobol_sample
    except ImportError as exc:
        raise ComputeError("sensitivity optional dependency SALib is not installed") from exc
    raw_parameters = _sequence(inputs.get("parameters"), "inputs.parameters")
    if not 1 <= len(raw_parameters) <= MAX_PARAMETERS:
        raise ComputeError(f"inputs.parameters must contain 1 to {MAX_PARAMETERS} entries")
    names: list[str] = []
    bounds: list[list[float]] = []
    for index, raw in enumerate(raw_parameters):
        row = _mapping(raw, f"inputs.parameters[{index}]")
        name = str(row.get("name") or "")
        low = _finite(row.get("minimum"), f"parameter[{name}].minimum")
        high = _finite(row.get("maximum"), f"parameter[{name}].maximum")
        if not name or name in names or not low < high:
            raise ComputeError("parameter names must be unique and minimum lower than maximum")
        names.append(name)
        bounds.append([low, high])
    base_samples = _integer(inputs.get("base_samples", 256), "inputs.base_samples", 64, 8192)
    if base_samples & (base_samples - 1):
        raise ComputeError("inputs.base_samples must be a power of two")
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    try:
        samples = np.asarray(sobol_sample.sample(problem, base_samples, calc_second_order=False, scramble=True, seed=seed), dtype=float)
        outcomes = _fixed_model(samples, names, _mapping(inputs.get("model"), "inputs.model"))
        analysis = sobol.analyze(problem, outcomes, calc_second_order=False, print_to_console=False, seed=seed)
    except Exception as exc:
        raise ComputeError(f"Sobol sensitivity failed: {type(exc).__name__}: {exc}") from exc
    rows = [{
        "parameter": names[index],
        "first_order": float(analysis["S1"][index]),
        "first_order_confidence": float(analysis["S1_conf"][index]),
        "total_order": float(analysis["ST"][index]),
        "total_order_confidence": float(analysis["ST_conf"][index]),
    } for index in range(len(names))]
    rows.sort(key=lambda row: (-row["total_order"], row["parameter"]))
    quantiles = np.quantile(outcomes, [0.05, 0.5, 0.95])
    return {
        "mode": "sobol_sensitivity",
        "base_samples": base_samples,
        "evaluations": int(samples.shape[0]),
        "seed": seed,
        "ranking": rows,
        "output_distribution": {
            "mean": float(np.mean(outcomes)),
            "standard_deviation": float(np.std(outcomes)),
            "p05": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p95": float(quantiles[2]),
        },
        "fixed_model_only": True,
        "decision_support_only": True,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "sarimax_forecast": sarimax_forecast,
    "exponential_smoothing_forecast": exponential_smoothing_forecast,
    "vector_autoregression_forecast": vector_autoregression_forecast,
    "sobol_sensitivity": sobol_sensitivity,
}
