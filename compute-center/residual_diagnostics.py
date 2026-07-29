#!/usr/bin/env python3
"""Residual diagnostics for calibration and model validation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats


class ResidualDiagnosticError(ValueError):
    pass


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        raise ResidualDiagnosticError(f"{name} must be a finite vector with at least 3 values")
    return array


def diagnose_residuals(actual: Sequence[float], predicted: Sequence[float], *, time_index: Sequence[float] | None = None, groups: Sequence[str] | None = None) -> dict[str, Any]:
    observed = _vector(actual, "actual")
    fitted = _vector(predicted, "predicted")
    if observed.shape != fitted.shape:
        raise ResidualDiagnosticError("actual and predicted must have equal length")
    residual = fitted - observed
    centered = residual - np.mean(residual)
    lag1 = float(np.corrcoef(centered[:-1], centered[1:])[0, 1]) if residual.size > 3 else 0.0
    if not np.isfinite(lag1):
        lag1 = 0.0
    fitted_corr = float(stats.spearmanr(np.abs(residual), fitted).statistic)
    if not np.isfinite(fitted_corr):
        fitted_corr = 0.0
    durbin_watson = float(np.sum(np.diff(residual) ** 2) / max(np.sum(residual ** 2), 1e-15))
    diagnostics: dict[str, Any] = {"count": int(residual.size), "rmse": float(np.sqrt(np.mean(residual ** 2))), "mae": float(np.mean(np.abs(residual))), "mean_bias": float(np.mean(residual)), "median_bias": float(np.median(residual)), "residual_standard_deviation": float(np.std(residual, ddof=1)), "skewness": float(stats.skew(residual, bias=False)), "excess_kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False)), "lag1_autocorrelation": lag1, "durbin_watson": durbin_watson, "absolute_residual_fitted_spearman": fitted_corr, "extreme_error_p95": float(np.quantile(np.abs(residual), 0.95))}
    if time_index is not None:
        time_values = _vector(time_index, "time_index")
        if time_values.shape != observed.shape:
            raise ResidualDiagnosticError("time_index must match actual")
        diagnostics["residual_time_trend"] = float(stats.linregress(time_values, residual).slope)
    group_rows: dict[str, Any] = {}
    if groups is not None:
        if len(groups) != observed.size:
            raise ResidualDiagnosticError("groups must match actual")
        group_array = np.asarray(groups, dtype=object)
        for group in sorted({str(item) for item in group_array}):
            mask = np.asarray([str(item) == group for item in group_array], dtype=bool)
            values = residual[mask]
            group_rows[group] = {"count": int(values.size), "mean_bias": float(np.mean(values)), "mae": float(np.mean(np.abs(values)))}
    warnings: list[str] = []
    if abs(diagnostics["mean_bias"]) > 0.25 * max(float(np.std(observed)), 1e-12):
        warnings.append("SYSTEMATIC_BIAS")
    if abs(lag1) > 0.3 or durbin_watson < 1.2 or durbin_watson > 2.8:
        warnings.append("RESIDUAL_AUTOCORRELATION")
    if abs(fitted_corr) > 0.3:
        warnings.append("HETEROSKEDASTICITY_RISK")
    if abs(diagnostics["skewness"]) > 1:
        warnings.append("RESIDUAL_SKEW")
    if diagnostics["excess_kurtosis"] > 3:
        warnings.append("HEAVY_TAIL_RISK")
    return {"schema_version": "compute-residual-diagnostics-v1", "status": "PASS" if not warnings else "WARN", "diagnostics": diagnostics, "group_diagnostics": group_rows, "warnings": warnings}
