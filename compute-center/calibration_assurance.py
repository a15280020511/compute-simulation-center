#!/usr/bin/env python3
"""Fixed calibration, conformal and predictive-check helpers.

No function evaluates ticket-supplied code. OpenTURNS and MAPIE are imported lazily and
all public helpers accept only bounded numeric arrays or fixed sklearn estimators.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
import math
import numpy as np

class AssuranceError(ValueError):
    pass


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or a.size == 0 or a.size > 100_000 or not np.isfinite(a).all():
        raise AssuranceError(f"{name} must be a non-empty finite vector")
    return a


def coverage_evaluation(observed: Sequence[float], lower: Sequence[float], upper: Sequence[float], target: float) -> dict[str, Any]:
    y, lo, hi = (_vector(observed, 'observed'), _vector(lower, 'lower'), _vector(upper, 'upper'))
    if not (y.shape == lo.shape == hi.shape) or np.any(lo > hi):
        raise AssuranceError('observed/lower/upper must have equal shapes and lower <= upper')
    if not 0 < target < 1:
        raise AssuranceError('target must be between 0 and 1')
    covered = (y >= lo) & (y <= hi)
    empirical = float(np.mean(covered))
    return {
        'schema_version': 'coverage-evaluation-v1', 'target_coverage': float(target),
        'empirical_coverage': empirical, 'coverage_gap': empirical - float(target),
        'average_interval_width': float(np.mean(hi - lo)),
        'undercoverage': empirical + 1e-12 < target,
        'overly_conservative': empirical > min(0.999, target + 0.08),
        'observations': int(y.size),
    }


def conditional_coverage_check(observed: Sequence[float], lower: Sequence[float], upper: Sequence[float], groups: Sequence[str], target: float, minimum_group_size: int = 20) -> dict[str, Any]:
    y, lo, hi = (_vector(observed, 'observed'), _vector(lower, 'lower'), _vector(upper, 'upper'))
    if len(groups) != y.size:
        raise AssuranceError('groups must match observed length')
    rows = []
    for group in sorted(set(map(str, groups))):
        idx = np.asarray([str(v) == group for v in groups], dtype=bool)
        if int(idx.sum()) < minimum_group_size:
            rows.append({'group': group, 'status': 'insufficient-sample', 'count': int(idx.sum())})
        else:
            row = coverage_evaluation(y[idx], lo[idx], hi[idx], target)
            row.update({'group': group, 'count': int(idx.sum()), 'status': 'evaluated'})
            rows.append(row)
    evaluated = [r for r in rows if r['status'] == 'evaluated']
    return {'schema_version': 'conditional-coverage-v1', 'groups': rows, 'minimum_empirical_coverage': min((r['empirical_coverage'] for r in evaluated), default=None)}


def parameter_identifiability_check(jacobian: Sequence[Sequence[float]], condition_limit: float = 1e8) -> dict[str, Any]:
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or j.size == 0 or max(j.shape) > 10_000 or not np.isfinite(j).all():
        raise AssuranceError('jacobian must be a finite two-dimensional matrix')
    singular = np.linalg.svd(j, compute_uv=False)
    rank = int(np.linalg.matrix_rank(j))
    smallest = float(singular[-1]) if singular.size else 0.0
    condition = math.inf if smallest <= 0 else float(singular[0] / smallest)
    return {'schema_version': 'parameter-identifiability-v1', 'rows': j.shape[0], 'parameters': j.shape[1], 'rank': rank, 'full_column_rank': rank == j.shape[1], 'condition_number': condition, 'identifiable': rank == j.shape[1] and condition <= condition_limit}


def predictive_check(samples: Sequence[Sequence[float]], observed: Sequence[float] | None = None, plausible_minimum: float | None = None, plausible_maximum: float | None = None, label: str = 'prior') -> dict[str, Any]:
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or a.size == 0 or a.shape[0] > 100_000 or not np.isfinite(a).all():
        raise AssuranceError('samples must be a finite draw-by-observation matrix')
    result = {'schema_version': 'predictive-check-v1', 'label': label, 'draws': int(a.shape[0]), 'observations_per_draw': int(a.shape[1]), 'mean': float(np.mean(a)), 'p05': float(np.quantile(a, .05)), 'p50': float(np.quantile(a, .5)), 'p95': float(np.quantile(a, .95))}
    if plausible_minimum is not None or plausible_maximum is not None:
        lower = -math.inf if plausible_minimum is None else float(plausible_minimum)
        upper = math.inf if plausible_maximum is None else float(plausible_maximum)
        result['plausible_fraction'] = float(np.mean((a >= lower) & (a <= upper)))
        result['plausibility_pass'] = result['plausible_fraction'] >= 0.95
    if observed is not None:
        y = _vector(observed, 'observed')
        if y.size != a.shape[1]:
            raise AssuranceError('observed length must match predictive columns')
        lo, hi = np.quantile(a, [.05, .95], axis=0)
        result['observed_90_coverage'] = float(np.mean((y >= lo) & (y <= hi)))
        result['mean_absolute_error_of_predictive_mean'] = float(np.mean(np.abs(y - np.mean(a, axis=0))))
    return result


def openturns_linear_least_squares_calibration(features: Sequence[Sequence[float]], observations: Sequence[float], starting_point: Sequence[float], nonlinear: bool = False) -> dict[str, Any]:
    try:
        import openturns as ot
    except ImportError as exc:
        raise AssuranceError('OpenTURNS optional dependency is not installed') from exc
    x = np.asarray(features, dtype=float); y = _vector(observations, 'observations'); start = _vector(starting_point, 'starting_point')
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != start.size or not np.isfinite(x).all():
        raise AssuranceError('features must be finite rows matching observations and parameter count')
    p = x.shape[1]
    def model_value(row):
        values = np.asarray(row, dtype=float)
        return [float(values[:p] @ values[p:])]
    full = ot.PythonFunction(2 * p, 1, model_value)
    model = ot.ParametricFunction(full, list(range(p, 2 * p)), list(map(float, start)))
    input_obs = ot.Sample(x.tolist())
    output_obs = ot.Sample([[float(v)] for v in y])
    cls = ot.NonLinearLeastSquaresCalibration if nonlinear else ot.LinearLeastSquaresCalibration
    algo = cls(model, input_obs, output_obs, list(map(float, start)), 'SVD') if not nonlinear else cls(model, input_obs, output_obs, list(map(float, start)))
    algo.run(); result = algo.getResult(); posterior = result.getParameterPosterior()
    mean = list(map(float, posterior.getMean()))
    covariance = np.asarray(posterior.getCovariance(), dtype=float).tolist()
    return {'schema_version': 'openturns-calibration-result-v1', 'backend': 'NonLinearLeastSquaresCalibration' if nonlinear else 'LinearLeastSquaresCalibration', 'posterior_mean': mean, 'posterior_covariance': covariance, 'residual_function_available': True}


def split_conformal_regression(x_train, y_train, x_calibration, y_calibration, x_test, confidence_levels: Iterable[float] = (.8, .9)) -> dict[str, Any]:
    try:
        from mapie.regression import SplitConformalRegressor
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise AssuranceError('MAPIE/scikit-learn optional dependencies are not installed') from exc
    levels = tuple(float(v) for v in confidence_levels)
    if not levels or any(not 0 < v < 1 for v in levels):
        raise AssuranceError('confidence levels must be between zero and one')
    model = SplitConformalRegressor(estimator=Ridge(alpha=1.0), confidence_level=levels, prefit=False)
    model.fit(np.asarray(x_train, dtype=float), _vector(y_train, 'y_train'))
    model.conformalize(np.asarray(x_calibration, dtype=float), _vector(y_calibration, 'y_calibration'))
    point, intervals = model.predict_interval(np.asarray(x_test, dtype=float))
    return {'schema_version': 'mapie-split-conformal-v1', 'confidence_levels': list(levels), 'point_prediction': np.asarray(point).tolist(), 'intervals': np.asarray(intervals).tolist()}


def cross_conformal_regression(x, y, x_test, confidence_levels: Iterable[float] = (.8, .9), folds: int = 5) -> dict[str, Any]:
    try:
        from mapie.regression import CrossConformalRegressor
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise AssuranceError('MAPIE/scikit-learn optional dependencies are not installed') from exc
    levels = tuple(float(v) for v in confidence_levels)
    model = CrossConformalRegressor(estimator=Ridge(alpha=1.0), confidence_level=levels, cv=int(folds), random_state=0)
    model.fit_conformalize(np.asarray(x, dtype=float), _vector(y, 'y'))
    point, intervals = model.predict_interval(np.asarray(x_test, dtype=float))
    return {'schema_version': 'mapie-cross-conformal-v1', 'confidence_levels': list(levels), 'point_prediction': np.asarray(point).tolist(), 'intervals': np.asarray(intervals).tolist()}
