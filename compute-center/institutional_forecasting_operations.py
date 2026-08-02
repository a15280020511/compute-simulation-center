#!/usr/bin/env python3
"""Allowlisted institutional forecasting, anomaly and extreme-value operations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from institutional_common import engine, integer, jsonable, matrix, vector
from think_tank_common import finite


def scalable_statistical_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("statsforecast")
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, Naive, Theta

    series = matrix(inputs.get("series"), "inputs.series", min_rows=1, max_rows=100, min_columns=12, max_columns=5_000)
    horizon = integer(inputs.get("horizon", 6), "inputs.horizon", 1, 365)
    season_length = integer(inputs.get("season_length", 1), "inputs.season_length", 1, min(365, series.shape[1]))
    model_name = str(inputs.get("model") or "autoets").lower()
    models = {
        "naive": Naive(),
        "autoarima": AutoARIMA(season_length=season_length),
        "autoets": AutoETS(season_length=season_length),
        "theta": Theta(season_length=season_length),
    }
    if model_name not in models:
        raise ComputeError("model must be one of naive, autoarima, autoets, theta")
    rows = []
    dates = pd.date_range("2000-01-01", periods=series.shape[1], freq="D")
    for index, values in enumerate(series):
        rows.extend(
            {"unique_id": f"s{index}", "ds": date, "y": float(value)}
            for date, value in zip(dates, values, strict=True)
        )
    frame = pd.DataFrame(rows)
    sf = StatsForecast(models=[models[model_name]], freq="D", n_jobs=1)
    result = sf.forecast(h=horizon, df=frame)
    value_columns = [column for column in result.columns if column not in {"unique_id", "ds"}]
    if len(value_columns) != 1:
        raise ComputeError("forecast engine returned an unexpected schema")
    forecast_column = value_columns[0]
    output = {}
    for unique_id, group in result.groupby("unique_id", sort=True):
        output[str(unique_id)] = [float(value) for value in group[forecast_column].to_numpy()]
    return {
        "mode": "scalable_statistical_forecast",
        "model": model_name,
        "series_count": int(series.shape[0]),
        "observations_per_series": int(series.shape[1]),
        "horizon": horizon,
        "forecasts": output,
        "engine": engine("statsforecast"),
    }


def hierarchical_forecast_reconciliation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("hierarchicalforecast")
    from hierarchicalforecast.methods import BottomUp

    summing = matrix(inputs.get("summing_matrix"), "inputs.summing_matrix", min_rows=2, max_rows=1_000, min_columns=1, max_columns=500)
    base_forecasts = matrix(
        inputs.get("base_forecasts"),
        "inputs.base_forecasts",
        min_rows=summing.shape[0],
        max_rows=summing.shape[0],
        min_columns=1,
        max_columns=365,
    )
    if base_forecasts.shape[0] != summing.shape[0]:
        raise ComputeError("base_forecasts rows must equal summing_matrix rows")
    bottom_count = summing.shape[1]
    if bottom_count > summing.shape[0]:
        raise ComputeError("summing_matrix has more bottom series than total series")
    idx_bottom = np.arange(summing.shape[0] - bottom_count, summing.shape[0], dtype=int)
    reconciled = BottomUp().fit_predict(S=summing, y_hat=base_forecasts, idx_bottom=idx_bottom)
    if not isinstance(reconciled, Mapping) or "mean" not in reconciled:
        raise ComputeError("hierarchical reconciliation returned an unexpected schema")
    mean = np.asarray(reconciled["mean"], dtype=float)
    coherence_error = float(np.max(np.abs(mean - summing @ mean[idx_bottom, :])))
    return {
        "mode": "hierarchical_forecast_reconciliation",
        "reconciled_forecasts": jsonable(mean),
        "series_count": int(mean.shape[0]),
        "horizon": int(mean.shape[1]),
        "maximum_coherence_error": coherence_error,
        "engine": engine("hierarchicalforecast"),
    }


def garch_volatility(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("arch")
    from arch import arch_model

    returns = vector(inputs.get("returns"), "inputs.returns", minimum=100, maximum=50_000)
    horizon = integer(inputs.get("horizon", 5), "inputs.horizon", 1, 100)
    p = integer(inputs.get("p", 1), "inputs.p", 1, 5)
    q = integer(inputs.get("q", 1), "inputs.q", 1, 5)
    scaled = returns * 100.0
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=p, q=q, dist="normal", rescale=False)
    fit = model.fit(disp="off", show_warning=False)
    forecast = fit.forecast(horizon=horizon, reindex=False)
    variance = np.asarray(forecast.variance.iloc[-1], dtype=float) / 10_000.0
    return {
        "mode": "garch_volatility",
        "p": p,
        "q": q,
        "parameters": {str(key): float(value) for key, value in fit.params.items()},
        "conditional_volatility_last": float(fit.conditional_volatility[-1] / 100.0),
        "forecast_variance": jsonable(variance),
        "forecast_volatility": jsonable(np.sqrt(variance)),
        "engine": engine("arch"),
    }


def anomaly_detection(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pyod")
    from pyod.models.ecod import ECOD

    features = matrix(inputs.get("features"), "inputs.features", min_rows=20, max_rows=50_000, max_columns=50)
    contamination = finite(inputs.get("contamination", 0.05), "inputs.contamination")
    if not 0.001 <= contamination <= 0.49:
        raise ComputeError("contamination must be between 0.001 and 0.49")
    detector = ECOD(contamination=contamination)
    detector.fit(features)
    scores = np.asarray(detector.decision_scores_, dtype=float)
    labels = np.asarray(detector.labels_, dtype=int)
    ranked = np.argsort(scores)[::-1]
    top_n = min(100, features.shape[0])
    return {
        "mode": "anomaly_detection",
        "observations": int(features.shape[0]),
        "features": int(features.shape[1]),
        "anomaly_count": int(np.sum(labels)),
        "threshold": float(detector.threshold_),
        "top_anomalies": [
            {"index": int(index), "score": float(scores[index]), "label": int(labels[index])}
            for index in ranked[:top_n]
        ],
        "engine": engine("pyod"),
    }


def extreme_value_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pyextremes")
    import pandas as pd
    from pyextremes import EVA

    values = vector(inputs.get("values"), "inputs.values", minimum=120, maximum=50_000)
    block_days = integer(inputs.get("block_days", 30), "inputs.block_days", 2, 3650)
    return_period = finite(inputs.get("return_period", 10.0), "inputs.return_period")
    if not 1.1 <= return_period <= 10_000:
        raise ComputeError("return_period must be between 1.1 and 10000")
    dates = pd.date_range("2000-01-01", periods=values.size, freq="D")
    series = pd.Series(values, index=dates, name="value")
    model = EVA(series)
    model.get_extremes(method="BM", extremes_type="high", block_size=f"{block_days}D", errors="ignore")
    model.fit_model(model="MLE", distribution="genextreme")
    result = model.get_return_value(return_period=return_period, return_period_size=f"{block_days}D", alpha=0.95)
    if hasattr(result, "return_value"):
        estimate = float(result.return_value)
        lower = float(result.lower_ci) if getattr(result, "lower_ci", None) is not None else None
        upper = float(result.upper_ci) if getattr(result, "upper_ci", None) is not None else None
    else:
        values_result = tuple(result) if isinstance(result, tuple) else (result,)
        estimate = float(values_result[0])
        lower = float(values_result[1]) if len(values_result) > 1 and values_result[1] is not None else None
        upper = float(values_result[2]) if len(values_result) > 2 and values_result[2] is not None else None
    return {
        "mode": "extreme_value_analysis",
        "block_days": block_days,
        "extreme_count": int(len(model.extremes)),
        "return_period": return_period,
        "return_value": estimate,
        "confidence_interval": [lower, upper],
        "engine": engine("pyextremes"),
    }


def probabilistic_forecast_verification(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("xskillscore", "xarray")
    import xarray as xr
    import xskillscore as xs

    observations = matrix(inputs.get("observations"), "inputs.observations", min_rows=2, max_rows=1_000, min_columns=2, max_columns=365)
    forecasts = matrix(
        inputs.get("forecasts"),
        "inputs.forecasts",
        min_rows=observations.shape[0],
        max_rows=observations.shape[0],
        min_columns=observations.shape[1],
        max_columns=observations.shape[1],
    )
    if forecasts.shape != observations.shape:
        raise ComputeError("forecasts and observations must have identical shapes")
    obs = xr.DataArray(observations, dims=("series", "time"))
    fcst = xr.DataArray(forecasts, dims=("series", "time"))
    rmse = xs.rmse(fcst, obs, dim="time", skipna=False)
    mae = xs.mae(fcst, obs, dim="time", skipna=False)
    correlation = xs.pearson_r(fcst, obs, dim="time", skipna=False)
    bias = (fcst - obs).mean(dim="time")
    return {
        "mode": "probabilistic_forecast_verification",
        "series_count": int(observations.shape[0]),
        "time_points": int(observations.shape[1]),
        "rmse_by_series": jsonable(rmse.values),
        "mae_by_series": jsonable(mae.values),
        "correlation_by_series": jsonable(correlation.values),
        "bias_by_series": jsonable(bias.values),
        "engine": engine("xskillscore", "xarray"),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "scalable_statistical_forecast": scalable_statistical_forecast,
    "hierarchical_forecast_reconciliation": hierarchical_forecast_reconciliation,
    "garch_volatility": garch_volatility,
    "anomaly_detection": anomaly_detection,
    "extreme_value_analysis": extreme_value_analysis,
    "probabilistic_forecast_verification": probabilistic_forecast_verification,
}
