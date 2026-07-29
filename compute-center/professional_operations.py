#!/usr/bin/env python3
"""Allowlisted professional GIS, Bayesian, and econometric operations.

All inputs are explicit JSON-compatible values. The module performs no network
access, dynamic imports requested by a ticket, arbitrary code execution, file
reads, or runtime package installation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from scipy import stats

from compute_runner import ComputeError

MAX_GIS_POINTS = 1_000
MAX_GIS_FEATURES = 500
MAX_REGRESSION_ROWS = 100_000
MAX_REGRESSION_COLUMNS = 100
MAX_BAYES_OBSERVATIONS = 100_000


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
    number = float(value)
    if not math.isfinite(number):
        raise ComputeError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ComputeError(f"{name} must be positive")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComputeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _credibility(inputs: Mapping[str, Any]) -> float:
    value = _finite(inputs.get("credibility", 0.95), "inputs.credibility")
    if not 0.5 < value < 1.0:
        raise ComputeError("inputs.credibility must be between 0.5 and 1")
    return value


def _numeric_vector(value: Any, name: str, maximum: int) -> np.ndarray:
    sequence = _sequence(value, name)
    if not sequence or len(sequence) > maximum:
        raise ComputeError(f"{name} must contain 1 to {maximum} values")
    result = np.asarray(
        [_finite(item, f"{name}[{index}]") for index, item in enumerate(sequence)],
        dtype=float,
    )
    if result.ndim != 1:
        raise ComputeError(f"{name} must be one-dimensional")
    return result


def _numeric_matrix(
    value: Any,
    name: str,
    max_rows: int,
    max_columns: int,
) -> np.ndarray:
    rows = _sequence(value, name)
    if not rows or len(rows) > max_rows:
        raise ComputeError(f"{name} must contain 1 to {max_rows} rows")
    parsed: list[list[float]] = []
    width: int | None = None
    for row_index, raw in enumerate(rows):
        row = [
            _finite(item, f"{name}[{row_index}][{column_index}]")
            for column_index, item in enumerate(
                _sequence(raw, f"{name}[{row_index}]")
            )
        ]
        if not row or len(row) > max_columns:
            raise ComputeError(
                f"{name}[{row_index}] must contain 1 to {max_columns} values"
            )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ComputeError(f"{name} rows must have equal length")
        parsed.append(row)
    return np.asarray(parsed, dtype=float)


def _interval(distribution: Any, credibility: float) -> tuple[float, float]:
    alpha = (1.0 - credibility) / 2.0
    low, high = distribution.ppf([alpha, 1.0 - alpha])
    return float(low), float(high)


# ----------------------------- GIS -----------------------------------------


def _point_rows(value: Any, name: str) -> list[dict[str, Any]]:
    rows = _sequence(value, name)
    if not rows or len(rows) > MAX_GIS_POINTS:
        raise ComputeError(f"{name} must contain 1 to {MAX_GIS_POINTS} points")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(rows):
        point = _mapping(raw, f"{name}[{index}]")
        allowed = {"id", "x", "y", "longitude", "latitude"}
        unexpected = sorted(set(point) - allowed)
        if unexpected:
            raise ComputeError(
                f"{name}[{index}] contains unsupported fields: {unexpected}"
            )
        point_id = str(point.get("id") or index)
        if point_id in ids:
            raise ComputeError(f"duplicate point id: {point_id}")
        ids.add(point_id)
        if "longitude" in point or "latitude" in point:
            x = _finite(point.get("longitude"), f"{name}[{index}].longitude")
            y = _finite(point.get("latitude"), f"{name}[{index}].latitude")
        else:
            x = _finite(point.get("x"), f"{name}[{index}].x")
            y = _finite(point.get("y"), f"{name}[{index}].y")
        result.append({"id": point_id, "x": x, "y": y})
    return result


def _geojson_geometry(value: Any, name: str):
    from shapely.geometry import shape
    from shapely.validation import make_valid

    raw = _mapping(value, name)
    try:
        geometry = shape(raw)
    except Exception as exc:  # noqa: BLE001
        raise ComputeError(f"{name} is not valid GeoJSON geometry: {exc}") from exc
    if geometry.is_empty:
        raise ComputeError(f"{name} cannot be empty")
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.is_empty or not geometry.is_valid:
        raise ComputeError(f"{name} cannot be repaired into a valid geometry")
    return geometry


def _gis_geodesic_distance_matrix(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'geodesic_distance_matrix'
    from pyproj import Geod

    points = _point_rows(inputs.get("points"), "inputs.points")
    ellipsoid = str(inputs.get("ellipsoid") or "WGS84")
    try:
        geod = Geod(ellps=ellipsoid)
    except Exception as exc:  # noqa: BLE001
        raise ComputeError(f"invalid ellipsoid {ellipsoid}: {exc}") from exc
    matrix: list[list[float]] = []
    for left in points:
        row: list[float] = []
        for right in points:
            _, _, distance = geod.inv(
                left["x"], left["y"], right["x"], right["y"]
            )
            row.append(float(distance))
        matrix.append(row)
    return {
        "mode": mode,
        "ellipsoid": ellipsoid,
        "unit": "metre",
        "point_ids": [row["id"] for row in points],
        "distance_matrix": matrix,
    }

def _gis_transform_coordinates(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'transform_coordinates'
    from pyproj import CRS, Transformer

    points = _point_rows(inputs.get("points"), "inputs.points")
    source_crs = str(inputs.get("source_crs") or "")
    target_crs = str(inputs.get("target_crs") or "")
    if not source_crs or not target_crs:
        raise ComputeError("source_crs and target_crs are required")
    try:
        source = CRS.from_user_input(source_crs)
        target = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(source, target, always_xy=True)
        transformed = []
        for point in points:
            x, y = transformer.transform(
                point["x"], point["y"], errcheck=True
            )
            transformed.append(
                {"id": point["id"], "x": float(x), "y": float(y)}
            )
    except Exception as exc:  # noqa: BLE001
        raise ComputeError(f"coordinate transformation failed: {exc}") from exc
    return {
        "mode": mode,
        "source_crs": source.to_string(),
        "target_crs": target.to_string(),
        "axis_order": "always_xy",
        "points": transformed,
    }

def _gis_geometry_overlay(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'geometry_overlay'
    from pyproj import CRS
    from shapely.geometry import mapping

    action = str(inputs.get("action") or "intersection")
    if action not in {
        "intersection",
        "union",
        "difference",
        "symmetric_difference",
    }:
        raise ComputeError(
            "action must be intersection, union, difference, or symmetric_difference"
        )
    crs_text = str(inputs.get("crs") or "")
    if not crs_text:
        raise ComputeError("inputs.crs is required for geometry_overlay")
    try:
        crs = CRS.from_user_input(crs_text)
    except Exception as exc:  # noqa: BLE001
        raise ComputeError(f"invalid CRS: {exc}") from exc
    left = _geojson_geometry(inputs.get("left"), "inputs.left")
    right = _geojson_geometry(inputs.get("right"), "inputs.right")
    result = getattr(left, action)(right)
    output_geometry: dict[str, Any] | None
    if result.is_empty:
        output_geometry = None
    else:
        output_geometry = mapping(result)
    return {
        "mode": mode,
        "action": action,
        "crs": crs.to_string(),
        "left_area": float(left.area),
        "right_area": float(right.area),
        "result_area": float(result.area),
        "result_length": float(result.length),
        "result_geometry": output_geometry,
    }

def _gis_spatial_predicate_matrix(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'spatial_predicate_matrix'
    predicate = str(inputs.get("predicate") or "intersects")
    if predicate not in {
        "intersects",
        "contains",
        "within",
        "touches",
        "overlaps",
        "crosses",
    }:
        raise ComputeError("unsupported spatial predicate")
    left_raw = _sequence(
        inputs.get("left_geometries"), "inputs.left_geometries"
    )
    right_raw = _sequence(
        inputs.get("right_geometries"), "inputs.right_geometries"
    )
    if (
        not left_raw
        or not right_raw
        or len(left_raw) > MAX_GIS_FEATURES
        or len(right_raw) > MAX_GIS_FEATURES
    ):
        raise ComputeError(
            f"geometry arrays must contain 1 to {MAX_GIS_FEATURES} geometries"
        )
    if len(left_raw) * len(right_raw) > 100_000:
        raise ComputeError(
            "spatial predicate matrix cannot exceed 100000 comparisons"
        )
    left = [
        _geojson_geometry(value, f"inputs.left_geometries[{index}]")
        for index, value in enumerate(left_raw)
    ]
    right = [
        _geojson_geometry(value, f"inputs.right_geometries[{index}]")
        for index, value in enumerate(right_raw)
    ]
    matrix = [
        [bool(getattr(a, predicate)(b)) for b in right]
        for a in left
    ]
    return {
        "mode": mode,
        "predicate": predicate,
        "matrix": matrix,
        "matches": int(sum(sum(row) for row in matrix)),
    }

def _gis_nearest_features(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'nearest_features'
    from pyproj import CRS

    source_raw = _sequence(
        inputs.get("source_geometries"), "inputs.source_geometries"
    )
    target_raw = _sequence(
        inputs.get("target_geometries"), "inputs.target_geometries"
    )
    if (
        not source_raw
        or not target_raw
        or len(source_raw) > MAX_GIS_FEATURES
        or len(target_raw) > MAX_GIS_FEATURES
    ):
        raise ComputeError(
            f"geometry arrays must contain 1 to {MAX_GIS_FEATURES} geometries"
        )
    if len(source_raw) * len(target_raw) > 100_000:
        raise ComputeError("nearest_features cannot exceed 100000 comparisons")
    crs_text = str(inputs.get("crs") or "")
    if not crs_text:
        raise ComputeError("inputs.crs is required for nearest_features")
    try:
        crs = CRS.from_user_input(crs_text)
    except Exception as exc:  # noqa: BLE001
        raise ComputeError(f"invalid CRS: {exc}") from exc
    if crs.is_geographic:
        raise ComputeError(
            "nearest_features requires a projected CRS; transform coordinates first"
        )
    sources = [
        _geojson_geometry(value, f"inputs.source_geometries[{index}]")
        for index, value in enumerate(source_raw)
    ]
    targets = [
        _geojson_geometry(value, f"inputs.target_geometries[{index}]")
        for index, value in enumerate(target_raw)
    ]
    rows = []
    for index, source in enumerate(sources):
        distances = [float(source.distance(target)) for target in targets]
        target_index = int(np.argmin(distances))
        rows.append(
            {
                "source_index": index,
                "target_index": target_index,
                "distance": distances[target_index],
            }
        )
    return {
        "mode": mode,
        "crs": crs.to_string(),
        "distance_unit": "CRS unit",
        "nearest": rows,
    }


def gis_spatial_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {
    'geodesic_distance_matrix': _gis_geodesic_distance_matrix,
    'transform_coordinates': _gis_transform_coordinates,
    'geometry_overlay': _gis_geometry_overlay,
    'spatial_predicate_matrix': _gis_spatial_predicate_matrix,
    'nearest_features': _gis_nearest_features,
    }
    handler = handlers.get(mode)
    if handler is None:
        raise ComputeError('inputs.mode must be geodesic_distance_matrix, transform_coordinates, geometry_overlay, spatial_predicate_matrix, or nearest_features')
    return handler(inputs)


# ----------------------------- Bayesian ------------------------------------


def _bayesian_beta_binomial(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'beta_binomial'
    credibility = _credibility(inputs)
    prior_alpha = _positive(inputs.get("prior_alpha"), "inputs.prior_alpha")
    prior_beta = _positive(inputs.get("prior_beta"), "inputs.prior_beta")
    successes = _integer(
        inputs.get("successes"),
        "inputs.successes",
        0,
        MAX_BAYES_OBSERVATIONS,
    )
    trials = _integer(
        inputs.get("trials"), "inputs.trials", 0, MAX_BAYES_OBSERVATIONS
    )
    if successes > trials:
        raise ComputeError("successes cannot exceed trials")
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + trials - successes
    distribution = stats.beta(posterior_alpha, posterior_beta)
    low, high = _interval(distribution, credibility)
    return {
        "mode": mode,
        "posterior": {
            "alpha": posterior_alpha,
            "beta": posterior_beta,
            "mean": float(distribution.mean()),
            "variance": float(distribution.var()),
            "credible_interval": {
                "level": credibility,
                "lower": low,
                "upper": high,
            },
        },
        "posterior_predictive_success_probability": float(
            distribution.mean()
        ),
    }

def _bayesian_gamma_poisson(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'gamma_poisson'
    credibility = _credibility(inputs)
    prior_shape = _positive(inputs.get("prior_shape"), "inputs.prior_shape")
    prior_rate = _positive(inputs.get("prior_rate"), "inputs.prior_rate")
    events = _integer(
        inputs.get("events"), "inputs.events", 0, MAX_BAYES_OBSERVATIONS
    )
    exposure = _positive(inputs.get("exposure"), "inputs.exposure")
    posterior_shape = prior_shape + events
    posterior_rate = prior_rate + exposure
    distribution = stats.gamma(
        a=posterior_shape, scale=1.0 / posterior_rate
    )
    low, high = _interval(distribution, credibility)
    return {
        "mode": mode,
        "posterior": {
            "shape": posterior_shape,
            "rate": posterior_rate,
            "mean_rate": float(distribution.mean()),
            "variance": float(distribution.var()),
            "credible_interval": {
                "level": credibility,
                "lower": low,
                "upper": high,
            },
        },
    }

def _bayesian_normal_mean_known_variance(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'normal_mean_known_variance'
    credibility = _credibility(inputs)
    prior_mean = _finite(inputs.get("prior_mean"), "inputs.prior_mean")
    prior_sd = _positive(inputs.get("prior_sd"), "inputs.prior_sd")
    known_sd = _positive(inputs.get("known_sd"), "inputs.known_sd")
    observations = _numeric_vector(
        inputs.get("observations"),
        "inputs.observations",
        MAX_BAYES_OBSERVATIONS,
    )
    prior_precision = 1.0 / (prior_sd**2)
    data_precision = observations.size / (known_sd**2)
    posterior_variance = 1.0 / (prior_precision + data_precision)
    posterior_mean = posterior_variance * (
        prior_precision * prior_mean
        + observations.sum() / (known_sd**2)
    )
    posterior_sd = math.sqrt(posterior_variance)
    distribution = stats.norm(loc=posterior_mean, scale=posterior_sd)
    low, high = _interval(distribution, credibility)
    return {
        "mode": mode,
        "observations": int(observations.size),
        "sample_mean": float(np.mean(observations)),
        "posterior": {
            "mean": float(posterior_mean),
            "standard_deviation": posterior_sd,
            "credible_interval": {
                "level": credibility,
                "lower": low,
                "upper": high,
            },
        },
    }

def _bayesian_regression_data(inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, bool, int, int]:
    x = _numeric_matrix(inputs.get("x"), "inputs.x", MAX_REGRESSION_ROWS, MAX_REGRESSION_COLUMNS)
    y = _numeric_vector(inputs.get("y"), "inputs.y", MAX_REGRESSION_ROWS)
    if x.shape[0] != y.size:
        raise ComputeError("x row count must match y length")
    add_intercept = bool(inputs.get("add_intercept", True))
    if add_intercept:
        x = np.column_stack([np.ones(y.size), x])
    n, k = x.shape
    if n <= k:
        raise ComputeError("bayesian_linear_regression requires more rows than coefficients")
    return x, y, add_intercept, n, k


def _bayesian_prior_mean(inputs: Mapping[str, Any], k: int) -> np.ndarray:
    raw = inputs.get("prior_mean")
    prior_mean = np.zeros(k, dtype=float) if raw is None else _numeric_vector(raw, "inputs.prior_mean", k)
    if prior_mean.size != k:
        raise ComputeError("prior_mean length must match coefficient count")
    return prior_mean


def _bayesian_prior_precision(inputs: Mapping[str, Any], k: int) -> np.ndarray:
    value = inputs.get("prior_precision", 1e-6)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.eye(k) * _positive(value, "inputs.prior_precision")
    matrix = _numeric_matrix(value, "inputs.prior_precision", k, k)
    if matrix.shape != (k, k):
        raise ComputeError("prior_precision matrix must be square and match coefficient count")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ComputeError("prior_precision must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ComputeError("prior_precision must be positive definite")
    return matrix


def _bayesian_coefficient_names(inputs: Mapping[str, Any], k: int, add_intercept: bool) -> list[str]:
    raw = inputs.get("coefficient_names")
    if raw is None:
        return (["intercept"] if add_intercept else []) + [
            f"x{index + 1}" for index in range(k - (1 if add_intercept else 0))
        ]
    names = [str(value) for value in _sequence(raw, "inputs.coefficient_names")]
    if len(names) != k or len(set(names)) != k:
        raise ComputeError("coefficient_names must be unique and match coefficient count")
    return names


def _bayesian_coefficient_rows(
    names: list[str], posterior_mean: np.ndarray, coefficient_scale: np.ndarray, credibility: float, quantile: float
) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(names):
        mean, scale = float(posterior_mean[index]), float(coefficient_scale[index])
        rows.append({
            "name": name,
            "posterior_mean": mean,
            "posterior_scale": scale,
            "credible_interval": {
                "level": credibility,
                "lower": mean - quantile * scale,
                "upper": mean + quantile * scale,
            },
        })
    return rows


def _bayesian_predictive_rows(
    inputs: Mapping[str, Any], *, add_intercept: bool, k: int, posterior_mean: np.ndarray,
    posterior_scale: float, posterior_shape: float, posterior_covariance_factor: np.ndarray,
    credibility: float, quantile: float,
) -> list[dict[str, Any]]:
    if "x_new" not in inputs:
        return []
    x_new = _numeric_matrix(inputs.get("x_new"), "inputs.x_new", 10_000, MAX_REGRESSION_COLUMNS)
    if add_intercept:
        x_new = np.column_stack([np.ones(x_new.shape[0]), x_new])
    if x_new.shape[1] != k:
        raise ComputeError("x_new column count must match fitted design")
    rows = []
    variance_scale = posterior_scale / posterior_shape
    for index, row in enumerate(x_new):
        mean = float(row @ posterior_mean)
        scale = math.sqrt(float(variance_scale * (1.0 + row @ posterior_covariance_factor @ row)))
        rows.append({
            "row": index,
            "mean": mean,
            "credible_interval": {
                "level": credibility,
                "lower": mean - quantile * scale,
                "upper": mean + quantile * scale,
            },
        })
    return rows


def _bayesian_linear_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    credibility = _credibility(inputs)
    x, y, add_intercept, n, k = _bayesian_regression_data(inputs)
    prior_mean = _bayesian_prior_mean(inputs, k)
    prior_precision = _bayesian_prior_precision(inputs, k)
    prior_shape = _positive(inputs.get("prior_shape", 1e-3), "inputs.prior_shape")
    prior_scale = _positive(inputs.get("prior_scale", 1e-3), "inputs.prior_scale")
    posterior_precision = prior_precision + x.T @ x
    covariance_factor = np.linalg.inv(posterior_precision)
    posterior_mean = covariance_factor @ (prior_precision @ prior_mean + x.T @ y)
    posterior_shape = prior_shape + n / 2.0
    quadratic = float(y @ y + prior_mean @ prior_precision @ prior_mean - posterior_mean @ posterior_precision @ posterior_mean)
    posterior_scale = prior_scale + 0.5 * max(0.0, quadratic)
    degrees_freedom = 2.0 * posterior_shape
    coefficient_scale = np.sqrt(np.diag((posterior_scale / posterior_shape) * covariance_factor))
    quantile = float(stats.t.ppf(1.0 - (1.0 - credibility) / 2.0, degrees_freedom))
    names = _bayesian_coefficient_names(inputs, k, add_intercept)
    coefficients = _bayesian_coefficient_rows(names, posterior_mean, coefficient_scale, credibility, quantile)
    predictive = _bayesian_predictive_rows(
        inputs,
        add_intercept=add_intercept,
        k=k,
        posterior_mean=posterior_mean,
        posterior_scale=posterior_scale,
        posterior_shape=posterior_shape,
        posterior_covariance_factor=covariance_factor,
        credibility=credibility,
        quantile=quantile,
    )
    sigma2_mean = None if posterior_shape <= 1 else posterior_scale / (posterior_shape - 1.0)
    return {
        "mode": "bayesian_linear_regression",
        "observations": n,
        "coefficients": coefficients,
        "posterior_noise_variance_mean": None if sigma2_mean is None else float(sigma2_mean),
        "posterior_shape": posterior_shape,
        "posterior_scale": posterior_scale,
        "degrees_freedom": degrees_freedom,
        "predictive": predictive,
    }


def bayesian_inference(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {
    'beta_binomial': _bayesian_beta_binomial,
    'gamma_poisson': _bayesian_gamma_poisson,
    'normal_mean_known_variance': _bayesian_normal_mean_known_variance,
    'bayesian_linear_regression': _bayesian_linear_regression,
    }
    handler = handlers.get(mode)
    if handler is None:
        raise ComputeError('inputs.mode must be beta_binomial, gamma_poisson, normal_mean_known_variance, or bayesian_linear_regression')
    return handler(inputs)


# ----------------------------- Econometrics --------------------------------


def _design_matrix(
    inputs: Mapping[str, Any], key: str = "x"
) -> tuple[np.ndarray, bool]:
    x = _numeric_matrix(
        inputs.get(key),
        f"inputs.{key}",
        MAX_REGRESSION_ROWS,
        MAX_REGRESSION_COLUMNS,
    )
    add_intercept = bool(inputs.get("add_intercept", True))
    if add_intercept:
        x = np.column_stack([np.ones(x.shape[0]), x])
    return x, add_intercept


def _coefficient_names(
    inputs: Mapping[str, Any], columns: int, add_intercept: bool
) -> list[str]:
    raw = inputs.get("coefficient_names")
    if raw is None:
        return (["intercept"] if add_intercept else []) + [
            f"x{index + 1}"
            for index in range(columns - (1 if add_intercept else 0))
        ]
    names = [
        str(value)
        for value in _sequence(raw, "inputs.coefficient_names")
    ]
    if len(names) != columns or len(set(names)) != columns:
        raise ComputeError(
            "coefficient_names must be unique and match coefficient count"
        )
    return names


def _regression_result(
    y: np.ndarray,
    x: np.ndarray,
    names: list[str],
    *,
    weights: np.ndarray | None,
    covariance_type: str,
    confidence: float,
) -> dict[str, Any]:
    n, k = x.shape
    if n <= k:
        raise ComputeError("regression requires more rows than coefficients")
    if np.linalg.matrix_rank(x) < k:
        raise ComputeError("design matrix is rank deficient")
    if weights is None:
        root_weights = np.ones(n)
    else:
        if weights.shape != (n,) or np.any(weights <= 0):
            raise ComputeError(
                "weights must be positive and match row count"
            )
        root_weights = np.sqrt(weights)
    xw = x * root_weights[:, None]
    yw = y * root_weights
    bread = np.linalg.inv(xw.T @ xw)
    beta = bread @ xw.T @ yw
    residuals = y - x @ beta
    weighted_residuals = residuals * root_weights
    rank = int(np.linalg.matrix_rank(xw))
    df_resid = n - rank
    ssr = float(weighted_residuals @ weighted_residuals)
    sigma2 = ssr / df_resid
    leverage = np.sum((xw @ bread) * xw, axis=1)
    covariance_type = covariance_type.upper()
    if covariance_type == "NONROBUST":
        covariance = sigma2 * bread
    elif covariance_type in {"HC0", "HC1", "HC2", "HC3"}:
        squared = weighted_residuals**2
        if covariance_type == "HC1":
            squared = squared * n / df_resid
        elif covariance_type == "HC2":
            squared = squared / np.maximum(1.0 - leverage, 1e-12)
        elif covariance_type == "HC3":
            squared = squared / np.maximum(
                (1.0 - leverage) ** 2, 1e-12
            )
        meat = xw.T @ (xw * squared[:, None])
        covariance = bread @ meat @ bread
    else:
        raise ComputeError(
            "covariance_type must be nonrobust, HC0, HC1, HC2, or HC3"
        )
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0.0)
    )
    t_statistics = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df_resid)
    alpha = (1.0 - confidence) / 2.0
    critical = float(stats.t.ppf(1.0 - alpha, df_resid))
    weighted_mean = float(
        np.average(y, weights=np.ones(n) if weights is None else weights)
    )
    total_sum = float(
        np.sum((root_weights * (y - weighted_mean)) ** 2)
    )
    r_squared = 1.0 - ssr / total_sum if total_sum > 0 else 1.0
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / df_resid
    residual_sum = float(np.sum(residuals**2))
    durbin_watson = (
        float(np.sum(np.diff(residuals) ** 2) / residual_sum)
        if residual_sum > 0
        else 0.0
    )
    jb = stats.jarque_bera(residuals)
    condition_number = float(np.linalg.cond(xw))
    coefficients = []
    for index, name in enumerate(names):
        coefficients.append(
            {
                "name": name,
                "estimate": float(beta[index]),
                "standard_error": float(standard_errors[index]),
                "t_statistic": float(t_statistics[index]),
                "p_value": float(p_values[index]),
                "confidence_interval": {
                    "level": confidence,
                    "lower": float(
                        beta[index] - critical * standard_errors[index]
                    ),
                    "upper": float(
                        beta[index] + critical * standard_errors[index]
                    ),
                },
            }
        )
    return {
        "observations": n,
        "rank": rank,
        "degrees_freedom_residual": df_resid,
        "covariance_type": covariance_type,
        "coefficients": coefficients,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "residual_standard_error": math.sqrt(sigma2),
        "durbin_watson": durbin_watson,
        "jarque_bera": {
            "statistic": float(jb.statistic),
            "p_value": float(jb.pvalue),
        },
        "condition_number": condition_number,
        "warnings": (
            [
                "High condition number may indicate multicollinearity or scaling problems."
            ]
            if condition_number > 1e8
            else []
        ),
    }


def _econometric_linear_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    confidence = _credibility(inputs)
    covariance_type = str(inputs.get("covariance_type") or "HC1")
    x, add_intercept = _design_matrix(inputs)
    y = _numeric_vector(
        inputs.get("y"), "inputs.y", MAX_REGRESSION_ROWS
    )
    if x.shape[0] != y.size:
        raise ComputeError("x row count must match y length")
    weights = None
    if mode == "wls":
        weights = _numeric_vector(
            inputs.get("weights"),
            "inputs.weights",
            MAX_REGRESSION_ROWS,
        )
        if weights.size != y.size:
            raise ComputeError("weights length must match y")
    names = _coefficient_names(inputs, x.shape[1], add_intercept)
    result = _regression_result(
        y,
        x,
        names,
        weights=weights,
        covariance_type=covariance_type,
        confidence=confidence,
    )
    result["mode"] = mode
    return result

def _econometric_difference_in_differences(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = 'difference_in_differences'
    confidence = _credibility(inputs)
    covariance_type = str(inputs.get("covariance_type") or "HC1")
    outcome = _numeric_vector(
        inputs.get("outcome"),
        "inputs.outcome",
        MAX_REGRESSION_ROWS,
    )
    treatment = _numeric_vector(
        inputs.get("treatment"),
        "inputs.treatment",
        MAX_REGRESSION_ROWS,
    )
    post = _numeric_vector(
        inputs.get("post"), "inputs.post", MAX_REGRESSION_ROWS
    )
    if not (outcome.size == treatment.size == post.size):
        raise ComputeError(
            "outcome, treatment, and post must have equal lengths"
        )
    if not set(np.unique(treatment)).issubset(
        {0.0, 1.0}
    ) or not set(np.unique(post)).issubset({0.0, 1.0}):
        raise ComputeError("treatment and post must be binary 0/1")
    columns = [treatment, post, treatment * post]
    names = [
        "intercept",
        "treatment",
        "post",
        "treatment_x_post",
    ]
    covariates_raw = inputs.get("covariates")
    if covariates_raw is not None:
        covariates = _numeric_matrix(
            covariates_raw,
            "inputs.covariates",
            MAX_REGRESSION_ROWS,
            MAX_REGRESSION_COLUMNS,
        )
        if covariates.shape[0] != outcome.size:
            raise ComputeError(
                "covariates row count must match outcome"
            )
        columns.extend(
            [covariates[:, index] for index in range(covariates.shape[1])]
        )
        names.extend(
            [
                f"covariate_{index + 1}"
                for index in range(covariates.shape[1])
            ]
        )
    x = np.column_stack([np.ones(outcome.size), *columns])
    result = _regression_result(
        outcome,
        x,
        names,
        weights=None,
        covariance_type=covariance_type,
        confidence=confidence,
    )
    result["mode"] = mode
    result["difference_in_differences_estimate"] = next(
        row
        for row in result["coefficients"]
        if row["name"] == "treatment_x_post"
    )
    result["identification_warning"] = (
        "Causal interpretation requires parallel trends and no differential "
        "concurrent shocks."
    )
    return result

def _iv_matrices(inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, np.ndarray]:
    y = _numeric_vector(inputs.get("y"), "inputs.y", MAX_REGRESSION_ROWS)
    endogenous = _numeric_matrix(inputs.get("endogenous"), "inputs.endogenous", MAX_REGRESSION_ROWS, MAX_REGRESSION_COLUMNS)
    instruments = _numeric_matrix(inputs.get("instruments"), "inputs.instruments", MAX_REGRESSION_ROWS, MAX_REGRESSION_COLUMNS)
    if endogenous.shape[0] != y.size or instruments.shape[0] != y.size:
        raise ComputeError("endogenous and instruments row counts must match y")
    raw = inputs.get("exogenous")
    exogenous = np.empty((y.size, 0)) if raw is None else _numeric_matrix(raw, "inputs.exogenous", MAX_REGRESSION_ROWS, MAX_REGRESSION_COLUMNS)
    if exogenous.shape[0] != y.size:
        raise ComputeError("exogenous row count must match y")
    add_intercept = bool(inputs.get("add_intercept", True))
    base = np.column_stack([np.ones(y.size), exogenous]) if add_intercept else exogenous
    x, z = np.column_stack([base, endogenous]), np.column_stack([base, instruments])
    if x.shape[1] > z.shape[1]:
        raise ComputeError("model is underidentified: instruments must cover endogenous regressors")
    if np.linalg.matrix_rank(z) < z.shape[1]:
        raise ComputeError("instrument matrix is rank deficient")
    return y, endogenous, instruments, exogenous, add_intercept, base, x, z


def _iv_estimate(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, np.ndarray]:
    ztz_inv = np.linalg.inv(z.T @ z)
    normal = x.T @ z @ ztz_inv @ z.T @ x
    if np.linalg.matrix_rank(normal) < x.shape[1]:
        raise ComputeError("2SLS normal matrix is rank deficient")
    bread = np.linalg.inv(normal)
    beta = bread @ x.T @ z @ ztz_inv @ z.T @ y
    residuals = y - x @ beta
    n, k = x.shape
    df = n - k
    if df <= 0:
        raise ComputeError("iv_2sls requires more rows than coefficients")
    return beta, residuals, bread, n, k, df, ztz_inv


def _iv_covariance(
    covariance_type: str, residuals: np.ndarray, bread: np.ndarray, x: np.ndarray,
    z: np.ndarray, ztz_inv: np.ndarray, n: int, df: int,
) -> tuple[str, np.ndarray]:
    kind = covariance_type.upper()
    if kind == "NONROBUST":
        return kind, float(residuals @ residuals / df) * bread
    if kind not in {"HC0", "HC1"}:
        raise ComputeError("iv_2sls covariance_type must be nonrobust, HC0, or HC1")
    projected_x = z @ ztz_inv @ z.T @ x
    meat = projected_x.T @ (projected_x * (residuals**2)[:, None])
    covariance = bread @ meat @ bread
    return kind, covariance * (n / df) if kind == "HC1" else covariance


def _iv_coefficient_names(
    inputs: Mapping[str, Any], *, add_intercept: bool, exogenous_columns: int,
    endogenous_columns: int, coefficient_count: int,
) -> list[str]:
    raw = inputs.get("coefficient_names")
    if raw is None:
        return (["intercept"] if add_intercept else []) + [
            f"exog_{index + 1}" for index in range(exogenous_columns)
        ] + [f"endogenous_{index + 1}" for index in range(endogenous_columns)]
    names = [str(value) for value in _sequence(raw, "inputs.coefficient_names")]
    if len(names) != coefficient_count or len(set(names)) != coefficient_count:
        raise ComputeError("coefficient_names must be unique and match coefficient count")
    return names


def _iv_coefficient_rows(
    names: list[str], beta: np.ndarray, covariance: np.ndarray, df: int, confidence: float
) -> list[dict[str, Any]]:
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistics = np.divide(beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0)
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df)
    critical = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df))
    rows = []
    for index, name in enumerate(names):
        estimate, error = float(beta[index]), float(standard_errors[index])
        rows.append({
            "name": name,
            "estimate": estimate,
            "standard_error": error,
            "t_statistic": float(t_statistics[index]),
            "p_value": float(p_values[index]),
            "confidence_interval": {
                "level": confidence,
                "lower": estimate - critical * error,
                "upper": estimate + critical * error,
            },
        })
    return rows


def _iv_first_stage(
    endogenous: np.ndarray, instruments: np.ndarray, base: np.ndarray, z: np.ndarray, confidence: float
) -> list[dict[str, Any]]:
    rows = []
    for column in range(endogenous.shape[1]):
        target = endogenous[:, column]
        unrestricted = _regression_result(target, z, [f"z{index}" for index in range(z.shape[1])], weights=None, covariance_type="NONROBUST", confidence=confidence)
        restricted = _regression_result(target, base, [f"base{index}" for index in range(base.shape[1])], weights=None, covariance_type="NONROBUST", confidence=confidence)
        ssr_u = unrestricted["residual_standard_error"] ** 2 * unrestricted["degrees_freedom_residual"]
        ssr_r = restricted["residual_standard_error"] ** 2 * restricted["degrees_freedom_residual"]
        q = instruments.shape[1]
        f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / unrestricted["degrees_freedom_residual"]) if ssr_u > 0 else float("inf")
        rows.append({
            "endogenous_index": column,
            "excluded_instruments_f_statistic": float(f_stat),
            "rule_of_thumb_weak": bool(f_stat < 10.0),
        })
    return rows


def _econometric_iv_2sls(inputs: Mapping[str, Any]) -> dict[str, Any]:
    confidence = _credibility(inputs)
    y, endogenous, instruments, exogenous, add_intercept, base, x, z = _iv_matrices(inputs)
    beta, residuals, bread, n, k, df, ztz_inv = _iv_estimate(y, x, z)
    covariance_type, covariance = _iv_covariance(
        str(inputs.get("covariance_type") or "HC1"), residuals, bread, x, z, ztz_inv, n, df
    )
    names = _iv_coefficient_names(
        inputs,
        add_intercept=add_intercept,
        exogenous_columns=exogenous.shape[1],
        endogenous_columns=endogenous.shape[1],
        coefficient_count=k,
    )
    return {
        "mode": "iv_2sls",
        "observations": n,
        "covariance_type": covariance_type,
        "coefficients": _iv_coefficient_rows(names, beta, covariance, df, confidence),
        "first_stage": _iv_first_stage(endogenous, instruments, base, z, confidence),
        "identification_warning": "Instrument validity requires relevance and exclusion restrictions; the first-stage statistic does not prove exogeneity.",
        "exogenous_columns": base.shape[1],
        "endogenous_columns": endogenous.shape[1],
        "instrument_columns": instruments.shape[1],
    }


def econometric_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {
    'ols': _econometric_linear_regression,
    'wls': _econometric_linear_regression,
    'difference_in_differences': _econometric_difference_in_differences,
    'iv_2sls': _econometric_iv_2sls,
    }
    handler = handlers.get(mode)
    if handler is None:
        raise ComputeError('inputs.mode must be ols, wls, difference_in_differences, or iv_2sls')
    return handler(inputs)


OPERATIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "gis_spatial_analysis": gis_spatial_analysis,
    "bayesian_inference": bayesian_inference,
    "econometric_analysis": econometric_analysis,
}
