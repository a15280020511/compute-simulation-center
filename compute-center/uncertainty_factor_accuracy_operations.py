#!/usr/bin/env python3
"""Bounded uncertainty, variable, factor and accuracy completion modes.

All handlers operate on fixed numeric inputs. They never evaluate ticket-supplied
code, fetch external data, connect to accounts, or execute trades.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Callable

import numpy as np
from scipy import optimize, stats

from compute_runner import ComputeError

MAX_ROWS = 100_000
MAX_COLUMNS = 200
MAX_SAMPLES = 20_000
MAX_BOOTSTRAPS = 2_000
MAX_FACTORS = 100
MAX_STRATEGIES = 200
MAX_BLOCKS = 12

VARIABLE_ROLES = {
    "target", "outcome", "exogenous", "endogenous", "control", "decision",
    "state", "latent", "nuisance", "constraint", "mediator", "moderator",
    "confounder", "instrument", "treatment", "exposure",
}

DISTRIBUTIONS = {
    "constant", "uniform", "triangular", "normal", "lognormal", "beta",
    "bernoulli", "binomial", "poisson", "negative_binomial", "gamma",
    "exponential", "weibull", "student_t", "pareto", "gev", "gpd",
    "truncated_normal", "zero_inflated_poisson", "hurdle_poisson",
    "gaussian_mixture", "empirical",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ComputeError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ComputeError(f"{name} must be <= {maximum}")
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
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ComputeError(f"{name} must contain numeric values") from exc
    if array.ndim != 1 or not minimum <= array.size <= maximum or not np.isfinite(array).all():
        raise ComputeError(f"{name} must be a finite vector with {minimum} to {maximum} values")
    return array


def _matrix(value: Any, name: str, minimum_rows: int = 1, maximum_rows: int = MAX_ROWS, maximum_columns: int = MAX_COLUMNS) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ComputeError(f"{name} must contain numeric values") from exc
    if (
        array.ndim != 2
        or not minimum_rows <= array.shape[0] <= maximum_rows
        or not 1 <= array.shape[1] <= maximum_columns
        or not np.isfinite(array).all()
    ):
        raise ComputeError(
            f"{name} must be a finite matrix with {minimum_rows} to {maximum_rows} rows "
            f"and 1 to {maximum_columns} columns"
        )
    return array


def _base(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_allowed": False,
        "brokerage_execution": False,
        "decision_support_only": True,
    }


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _correlation_matrix(value: Any, dimension: int, name: str) -> np.ndarray:
    matrix = _matrix(value, name, dimension, dimension, dimension)
    if matrix.shape != (dimension, dimension):
        raise ComputeError(f"{name} must be {dimension}x{dimension}")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ComputeError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise ComputeError(f"{name} diagonal must equal 1")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ComputeError(f"{name} must be positive semidefinite")
    values, vectors = np.linalg.eigh(matrix)
    stabilized = vectors @ np.diag(np.clip(values, 1e-12, None)) @ vectors.T
    scale = np.sqrt(np.diag(stabilized))
    return stabilized / np.outer(scale, scale)


def _distribution_parameters(spec: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    distribution = str(spec.get("distribution") or "")
    if distribution not in DISTRIBUTIONS:
        raise ComputeError(f"unsupported distribution: {distribution}")
    parameters = spec.get("parameters")
    return distribution, _mapping(parameters or {}, "variable.parameters")


def _ppf(distribution: str, parameters: Mapping[str, Any], uniforms: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1 - 1e-12)
    if distribution == "constant":
        return np.full(u.shape, _finite(parameters.get("value"), "value"))
    if distribution == "uniform":
        low = _finite(parameters.get("minimum"), "minimum")
        high = _finite(parameters.get("maximum"), "maximum")
        if high <= low:
            raise ComputeError("uniform maximum must exceed minimum")
        return stats.uniform.ppf(u, loc=low, scale=high - low)
    if distribution == "triangular":
        low = _finite(parameters.get("minimum"), "minimum")
        mode = _finite(parameters.get("mode"), "mode")
        high = _finite(parameters.get("maximum"), "maximum")
        if not low <= mode <= high or high <= low:
            raise ComputeError("triangular requires minimum <= mode <= maximum and nonzero range")
        return stats.triang.ppf(u, c=(mode - low) / (high - low), loc=low, scale=high - low)
    if distribution == "normal":
        return stats.norm.ppf(u, loc=_finite(parameters.get("mean"), "mean"), scale=_finite(parameters.get("standard_deviation"), "standard_deviation", 1e-12))
    if distribution == "lognormal":
        return stats.lognorm.ppf(u, s=_finite(parameters.get("sigma"), "sigma", 1e-12), scale=math.exp(_finite(parameters.get("mu"), "mu")))
    if distribution == "beta":
        return stats.beta.ppf(u, _finite(parameters.get("alpha"), "alpha", 1e-12), _finite(parameters.get("beta"), "beta", 1e-12))
    if distribution == "bernoulli":
        return stats.bernoulli.ppf(u, _finite(parameters.get("probability"), "probability", 0.0, 1.0))
    if distribution == "binomial":
        return stats.binom.ppf(u, _integer(parameters.get("trials"), "trials", 1, 1_000_000), _finite(parameters.get("probability"), "probability", 0.0, 1.0))
    if distribution == "poisson":
        return stats.poisson.ppf(u, _finite(parameters.get("rate"), "rate", 0.0))
    if distribution == "negative_binomial":
        mean = _finite(parameters.get("mean"), "mean", 1e-12)
        dispersion = _finite(parameters.get("dispersion"), "dispersion", 1e-12)
        probability = dispersion / (dispersion + mean)
        return stats.nbinom.ppf(u, dispersion, probability)
    if distribution == "gamma":
        return stats.gamma.ppf(u, a=_finite(parameters.get("shape"), "shape", 1e-12), scale=_finite(parameters.get("scale"), "scale", 1e-12))
    if distribution == "exponential":
        return stats.expon.ppf(u, scale=_finite(parameters.get("scale"), "scale", 1e-12))
    if distribution == "weibull":
        return stats.weibull_min.ppf(u, c=_finite(parameters.get("shape"), "shape", 1e-12), scale=_finite(parameters.get("scale"), "scale", 1e-12))
    if distribution == "student_t":
        return stats.t.ppf(u, df=_finite(parameters.get("degrees_of_freedom"), "degrees_of_freedom", 1.01), loc=_finite(parameters.get("location", 0.0), "location"), scale=_finite(parameters.get("scale", 1.0), "scale", 1e-12))
    if distribution == "pareto":
        return stats.pareto.ppf(u, b=_finite(parameters.get("shape"), "shape", 1e-12), scale=_finite(parameters.get("scale"), "scale", 1e-12))
    if distribution == "gev":
        return stats.genextreme.ppf(u, c=_finite(parameters.get("shape"), "shape"), loc=_finite(parameters.get("location", 0.0), "location"), scale=_finite(parameters.get("scale", 1.0), "scale", 1e-12))
    if distribution == "gpd":
        return stats.genpareto.ppf(u, c=_finite(parameters.get("shape"), "shape"), loc=_finite(parameters.get("location", 0.0), "location"), scale=_finite(parameters.get("scale", 1.0), "scale", 1e-12))
    if distribution == "truncated_normal":
        mean = _finite(parameters.get("mean"), "mean")
        sd = _finite(parameters.get("standard_deviation"), "standard_deviation", 1e-12)
        low = _finite(parameters.get("minimum"), "minimum")
        high = _finite(parameters.get("maximum"), "maximum")
        if high <= low:
            raise ComputeError("truncated_normal maximum must exceed minimum")
        return stats.truncnorm.ppf(u, (low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
    if distribution == "zero_inflated_poisson":
        rate = _finite(parameters.get("rate"), "rate", 1e-12)
        zero_probability = _finite(parameters.get("zero_probability"), "zero_probability", 0.0, 1.0)
        values = np.zeros(u.shape, dtype=float)
        mask = u > zero_probability
        adjusted = np.clip((u[mask] - zero_probability) / max(1 - zero_probability, 1e-12), 1e-12, 1 - 1e-12)
        values[mask] = stats.poisson.ppf(adjusted, rate)
        return values
    if distribution == "hurdle_poisson":
        rate = _finite(parameters.get("rate"), "rate", 1e-12)
        zero_probability = _finite(parameters.get("zero_probability"), "zero_probability", 0.0, 1.0)
        values = np.zeros(u.shape, dtype=float)
        mask = u > zero_probability
        adjusted = np.clip((u[mask] - zero_probability) / max(1 - zero_probability, 1e-12), 1e-12, 1 - 1e-12)
        p0 = math.exp(-rate)
        values[mask] = stats.poisson.ppf(p0 + adjusted * (1 - p0), rate)
        values[mask] = np.maximum(values[mask], 1.0)
        return values
    if distribution == "gaussian_mixture":
        weights = _vector(parameters.get("weights"), "weights", 1, 20)
        means = _vector(parameters.get("means"), "means", weights.size, weights.size)
        sds = _vector(parameters.get("standard_deviations"), "standard_deviations", weights.size, weights.size)
        if np.any(weights < 0) or not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8) or np.any(sds <= 0):
            raise ComputeError("gaussian_mixture requires nonnegative weights summing to 1 and positive standard deviations")
        low = float(np.min(means - 10 * sds))
        high = float(np.max(means + 10 * sds))
        left = np.full(u.shape, low)
        right = np.full(u.shape, high)
        for _ in range(80):
            middle = (left + right) / 2
            cdf = np.zeros(u.shape, dtype=float)
            for weight, mean, sd in zip(weights, means, sds, strict=True):
                cdf += weight * stats.norm.cdf(middle, loc=mean, scale=sd)
            move_right = cdf < u
            left[move_right] = middle[move_right]
            right[~move_right] = middle[~move_right]
        return (left + right) / 2
    if distribution == "empirical":
        values = np.sort(_vector(parameters.get("values"), "values", 2, MAX_ROWS))
        positions = np.minimum((u * values.size).astype(int), values.size - 1)
        return values[positions]
    raise ComputeError(f"unsupported distribution: {distribution}")


def joint_random_sample(inputs: Mapping[str, Any]) -> dict[str, Any]:
    variables = _sequence(inputs.get("variables"), "inputs.variables")
    if not 1 <= len(variables) <= 50:
        raise ComputeError("variables must contain 1 to 50 entries")
    sample_count = _integer(inputs.get("sample_count", 1000), "inputs.sample_count", 10, MAX_SAMPLES)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    names: list[str] = []
    specs: list[tuple[str, Mapping[str, Any]]] = []
    for index, raw in enumerate(variables):
        row = _mapping(raw, f"inputs.variables[{index}]")
        name = str(row.get("name") or "")
        if not name or name in names or len(name) > 100:
            raise ComputeError("variable names must be unique and 1 to 100 characters")
        names.append(name)
        specs.append(_distribution_parameters(row))
    dependence = _mapping(inputs.get("dependence") or {"method": "independent"}, "inputs.dependence")
    method = str(dependence.get("method") or "independent")
    rng = np.random.default_rng(seed)
    if method == "independent":
        uniforms = rng.random((sample_count, len(names)))
    else:
        correlation = _correlation_matrix(dependence.get("correlation_matrix"), len(names), "inputs.dependence.correlation_matrix")
        normals = rng.multivariate_normal(np.zeros(len(names)), correlation, size=sample_count)
        if method == "gaussian_copula":
            uniforms = stats.norm.cdf(normals)
        elif method == "t_copula":
            df = _finite(dependence.get("degrees_of_freedom", 5.0), "degrees_of_freedom", 2.01, 500.0)
            chi = rng.chisquare(df, size=sample_count)
            t_values = normals / np.sqrt(chi[:, None] / df)
            uniforms = stats.t.cdf(t_values, df=df)
        else:
            raise ComputeError("dependence.method must be independent, gaussian_copula, or t_copula")
    matrix = np.column_stack([
        _ppf(distribution, parameters, uniforms[:, index])
        for index, (distribution, parameters) in enumerate(specs)
    ])
    summaries = {}
    for index, name in enumerate(names):
        values = matrix[:, index]
        summaries[name] = {
            "distribution": specs[index][0],
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
    if len(names) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            empirical_correlation = np.corrcoef(matrix, rowvar=False)
        empirical_correlation = np.where(np.isfinite(empirical_correlation), empirical_correlation, 0.0)
        np.fill_diagonal(empirical_correlation, 1.0)
    else:
        empirical_correlation = np.asarray([[1.0]])
    result = _base("joint_random_sample")
    result.update({
        "sample_count": sample_count,
        "seed": seed,
        "dependence_method": method,
        "variables": summaries,
        "empirical_correlation": empirical_correlation.tolist(),
        "sample_sha256": _sha(np.round(matrix, 12).tolist()),
    })
    if bool(inputs.get("include_samples", False)):
        if sample_count > 1000:
            raise ComputeError("include_samples is limited to 1000 samples")
        result["samples"] = {name: matrix[:, index].tolist() for index, name in enumerate(names)}
    return result


_CONTINUOUS_FITS: dict[str, Any] = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "exponential": stats.expon,
    "weibull": stats.weibull_min,
    "student_t": stats.t,
    "pareto": stats.pareto,
    "gev": stats.genextreme,
    "gpd": stats.genpareto,
}


def _continuous_fit(name: str, values: np.ndarray) -> dict[str, Any]:
    distribution = _CONTINUOUS_FITS[name]
    fit_kwargs = {"floc": 0.0} if name in {"lognormal", "gamma", "exponential", "weibull", "pareto"} and np.all(values > 0) else {}
    parameters = distribution.fit(values, **fit_kwargs)
    log_likelihood = float(np.sum(distribution.logpdf(values, *parameters)))
    if not math.isfinite(log_likelihood):
        raise ComputeError(f"{name} fit produced non-finite likelihood")
    ks = stats.kstest(values, distribution.cdf, args=parameters)
    return {
        "distribution": name,
        "parameters": [float(value) for value in parameters],
        "log_likelihood": log_likelihood,
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "parameter_count": len(parameters),
    }


def _discrete_loglikelihood(name: str, values: np.ndarray) -> tuple[dict[str, float], float]:
    if np.any(values < 0) or not np.allclose(values, np.round(values)):
        raise ComputeError(f"{name} requires nonnegative integer observations")
    counts = np.round(values).astype(int)
    mean = float(np.mean(counts))
    if name == "poisson":
        parameters = {"rate": max(mean, 1e-12)}
        return parameters, float(np.sum(stats.poisson.logpmf(counts, parameters["rate"])))
    if name == "negative_binomial":
        variance = float(np.var(counts, ddof=1)) if counts.size > 1 else mean
        dispersion = max(mean**2 / max(variance - mean, 1e-9), 1e-6)
        probability = dispersion / (dispersion + max(mean, 1e-12))
        parameters = {"mean": mean, "dispersion": dispersion}
        return parameters, float(np.sum(stats.nbinom.logpmf(counts, dispersion, probability)))
    if name == "zero_inflated_poisson":
        def objective(raw: np.ndarray) -> float:
            rate = math.exp(float(raw[0]))
            zero_probability = 1 / (1 + math.exp(-float(raw[1])))
            pmf = (1 - zero_probability) * stats.poisson.pmf(counts, rate)
            pmf[counts == 0] += zero_probability
            return float(-np.sum(np.log(np.clip(pmf, 1e-300, None))))
        zero_fraction = float(np.mean(counts == 0))
        initial = np.asarray([math.log(max(mean, 1e-6)), math.log(max(zero_fraction, 1e-6) / max(1 - zero_fraction, 1e-6))])
        fitted = optimize.minimize(objective, initial, method="BFGS")
        rate = math.exp(float(fitted.x[0]))
        zero_probability = 1 / (1 + math.exp(-float(fitted.x[1])))
        return {"rate": rate, "zero_probability": zero_probability}, -float(fitted.fun)
    raise ComputeError(f"unsupported discrete distribution fit: {name}")


def distribution_fit_select(inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = _vector(inputs.get("observations"), "inputs.observations", 10, MAX_ROWS)
    raw_candidates = inputs.get("candidates") or [
        "normal", "lognormal", "gamma", "exponential", "weibull",
        "student_t", "pareto", "gev", "gpd",
    ]
    candidates = [str(item) for item in _sequence(raw_candidates, "inputs.candidates")]
    allowed = set(_CONTINUOUS_FITS) | {"poisson", "negative_binomial", "zero_inflated_poisson"}
    if not candidates or len(candidates) > 20 or any(item not in allowed for item in candidates):
        raise ComputeError("candidates contains an unsupported distribution")
    rows = []
    failures = []
    for name in candidates:
        try:
            if name in _CONTINUOUS_FITS:
                row = _continuous_fit(name, values)
            else:
                parameters, log_likelihood = _discrete_loglikelihood(name, values)
                row = {
                    "distribution": name,
                    "parameters": parameters,
                    "log_likelihood": log_likelihood,
                    "ks_statistic": None,
                    "ks_p_value": None,
                    "parameter_count": len(parameters),
                }
            count = int(values.size)
            parameters_count = int(row["parameter_count"])
            row["aic"] = float(2 * parameters_count - 2 * row["log_likelihood"])
            row["bic"] = float(math.log(count) * parameters_count - 2 * row["log_likelihood"])
            rows.append(row)
        except Exception as exc:
            failures.append({"distribution": name, "reason": str(exc)})
    if not rows:
        raise ComputeError("all distribution fits failed")
    rows.sort(key=lambda row: (row["bic"], row["aic"], row["distribution"]))
    result = _base("distribution_fit_select")
    result.update({
        "observation_count": int(values.size),
        "selected_distribution": rows[0]["distribution"],
        "selection_rule": "minimum BIC, then AIC",
        "fits": rows,
        "failed_candidates": failures,
    })
    return result


def variable_role_validate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = _sequence(inputs.get("variables"), "inputs.variables")
    if not 1 <= len(rows) <= 200:
        raise ComputeError("variables must contain 1 to 200 entries")
    variables: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = dict(_mapping(raw, f"inputs.variables[{index}]"))
        name = str(row.get("name") or "")
        role = str(row.get("role") or "")
        if not name or name in variables or len(name) > 100:
            raise ComputeError("variable names must be unique and 1 to 100 characters")
        if role not in VARIABLE_ROLES:
            raise ComputeError(f"unsupported variable role: {role}")
        dependencies = [str(item) for item in _sequence(row.get("dependencies") or [], f"{name}.dependencies")]
        variables[name] = {
            "role": role,
            "dependencies": dependencies,
            "observed": bool(row.get("observed", role != "latent")),
            "manipulable": bool(row.get("manipulable", role in {"decision", "treatment", "exposure"})),
        }
    for name, row in variables.items():
        missing = sorted(set(row["dependencies"]) - set(variables))
        if missing:
            raise ComputeError(f"{name} dependencies reference unknown variables: {missing}")
        if name in row["dependencies"]:
            raise ComputeError(f"{name} cannot depend on itself")
    state: dict[str, int] = {}
    order: list[str] = []
    def visit(name: str) -> None:
        marker = state.get(name, 0)
        if marker == 1:
            raise ComputeError("variable dependency graph contains a cycle")
        if marker == 2:
            return
        state[name] = 1
        for dependency in variables[name]["dependencies"]:
            visit(dependency)
        state[name] = 2
        order.append(name)
    for name in variables:
        visit(name)
    warnings = []
    for name, row in variables.items():
        role = row["role"]
        if role == "latent" and row["observed"]:
            warnings.append({"variable": name, "code": "LATENT_MARKED_OBSERVED"})
        if role in {"instrument", "confounder", "mediator", "moderator"} and not row["dependencies"]:
            warnings.append({"variable": name, "code": "CAUSAL_ROLE_WITHOUT_DECLARED_DEPENDENCIES"})
        if role in {"outcome", "target"} and row["manipulable"]:
            warnings.append({"variable": name, "code": "OUTCOME_MARKED_MANIPULABLE"})
        if role == "decision" and not row["manipulable"]:
            warnings.append({"variable": name, "code": "DECISION_NOT_MANIPULABLE"})
    counts: dict[str, int] = {}
    for row in variables.values():
        counts[row["role"]] = counts.get(row["role"], 0) + 1
    result = _base("variable_role_validate")
    result.update({
        "variable_count": len(variables),
        "role_counts": dict(sorted(counts.items())),
        "topological_order": order,
        "warnings": warnings,
        "status": "PASS" if not warnings else "WARN",
        "roles": sorted(VARIABLE_ROLES),
    })
    return result


def _roc_auc(actual: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(np.sum(actual == 1))
    negatives = int(np.sum(actual == 0))
    if positives == 0 or negatives == 0:
        return None
    ranks = stats.rankdata(probabilities, method="average")
    return float((np.sum(ranks[actual == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def _pr_auc(actual: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(np.sum(actual == 1))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_actual = actual[order]
    tp = np.cumsum(sorted_actual)
    fp = np.cumsum(1 - sorted_actual)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _calibration_line(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped))
    def objective(parameters: np.ndarray) -> float:
        z = parameters[0] + parameters[1] * logits
        predicted = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        return float(-np.sum(actual * np.log(np.clip(predicted, 1e-15, 1)) + (1 - actual) * np.log(np.clip(1 - predicted, 1e-15, 1))))
    fitted = optimize.minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not fitted.success:
        return {"intercept": None, "slope": None}
    return {"intercept": float(fitted.x[0]), "slope": float(fitted.x[1])}


def probabilistic_accuracy(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actual = _vector(inputs.get("actual"), "inputs.actual", 3, MAX_ROWS)
    probabilities = _vector(inputs.get("probabilities"), "inputs.probabilities", actual.size, actual.size)
    if not np.all((actual == 0) | (actual == 1)):
        raise ComputeError("actual must contain only 0 and 1")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ComputeError("probabilities must be between 0 and 1")
    bins = _integer(inputs.get("bins", 10), "inputs.bins", 2, 100)
    threshold = _finite(inputs.get("threshold", 0.5), "inputs.threshold", 0.0, 1.0)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, boundaries[1:-1], right=False), bins - 1)
    reliability = []
    ece = 0.0
    maximum_gap = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(np.sum(mask))
        if count:
            mean_probability = float(np.mean(probabilities[mask]))
            observed_rate = float(np.mean(actual[mask]))
            gap = abs(mean_probability - observed_rate)
            ece += count / actual.size * gap
            maximum_gap = max(maximum_gap, gap)
            reliability.append({
                "bin": index,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": gap,
            })
    predicted = (probabilities >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (actual == 1)))
    tn = int(np.sum((predicted == 0) & (actual == 0)))
    fp = int(np.sum((predicted == 1) & (actual == 0)))
    fn = int(np.sum((predicted == 0) & (actual == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    result = _base("probabilistic_accuracy")
    result.update({
        "observation_count": int(actual.size),
        "prevalence": float(np.mean(actual)),
        "brier_score": float(np.mean((probabilities - actual) ** 2)),
        "log_loss": float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))),
        "roc_auc": _roc_auc(actual, probabilities),
        "pr_auc": _pr_auc(actual, probabilities),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(maximum_gap),
        "calibration_line": _calibration_line(actual, probabilities),
        "reliability_bins": reliability,
        "threshold": threshold,
        "threshold_metrics": {
            "accuracy": float((tp + tn) / actual.size),
            "balanced_accuracy": float((recall + specificity) / 2),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "matthews_correlation_coefficient": float(mcc),
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        },
    })
    return result


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def forecast_accuracy(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actual = _vector(inputs.get("actual"), "inputs.actual", 3, MAX_ROWS)
    predicted = _vector(inputs.get("predicted"), "inputs.predicted", actual.size, actual.size)
    error = predicted - actual
    nonzero = np.abs(actual) > 1e-12
    denominator = np.abs(actual) + np.abs(predicted)
    seasonality = _integer(inputs.get("seasonality", 1), "inputs.seasonality", 1, max(1, actual.size - 1))
    naive_scale = float(np.mean(np.abs(actual[seasonality:] - actual[:-seasonality]))) if actual.size > seasonality else math.nan
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "mape": None if not np.any(nonzero) else float(np.mean(np.abs(error[nonzero] / actual[nonzero]))),
        "smape": float(np.mean(np.divide(2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 1e-12))),
        "mase": None if not math.isfinite(naive_scale) or naive_scale <= 1e-15 else float(np.mean(np.abs(error)) / naive_scale),
    }
    quantile_predictions_raw = inputs.get("quantile_predictions") or {}
    quantile_predictions = _mapping(quantile_predictions_raw, "inputs.quantile_predictions")
    quantile_losses = {}
    quantile_pairs = []
    for raw_quantile, raw_values in quantile_predictions.items():
        quantile = _finite(float(raw_quantile), f"quantile {raw_quantile}", 1e-6, 1 - 1e-6)
        values = _vector(raw_values, f"quantile_predictions[{raw_quantile}]", actual.size, actual.size)
        loss = _pinball(actual, values, quantile)
        quantile_losses[str(raw_quantile)] = loss
        quantile_pairs.append((quantile, loss))
    interval = None
    if inputs.get("lower") is not None or inputs.get("upper") is not None:
        lower = _vector(inputs.get("lower"), "inputs.lower", actual.size, actual.size)
        upper = _vector(inputs.get("upper"), "inputs.upper", actual.size, actual.size)
        if np.any(lower > upper):
            raise ComputeError("lower must not exceed upper")
        alpha = _finite(inputs.get("interval_alpha", 0.1), "inputs.interval_alpha", 1e-6, 0.999999)
        below = np.maximum(lower - actual, 0.0)
        above = np.maximum(actual - upper, 0.0)
        interval_score = upper - lower + 2 / alpha * (below + above)
        interval = {
            "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "target_coverage": float(1 - alpha),
            "average_width": float(np.mean(upper - lower)),
            "weighted_interval_score": float(np.mean(interval_score)),
        }
    baseline_comparison = None
    if inputs.get("baseline_predicted") is not None:
        baseline = _vector(inputs.get("baseline_predicted"), "inputs.baseline_predicted", actual.size, actual.size)
        differential = (predicted - actual) ** 2 - (baseline - actual) ** 2
        lag = _integer(inputs.get("dm_lag", 0), "inputs.dm_lag", 0, min(100, actual.size - 2))
        centered = differential - np.mean(differential)
        variance = float(np.mean(centered**2))
        for k in range(1, lag + 1):
            covariance = float(np.mean(centered[k:] * centered[:-k]))
            variance += 2 * (1 - k / (lag + 1)) * covariance
        standard_error = math.sqrt(max(variance / actual.size, 1e-15))
        statistic = float(np.mean(differential) / standard_error)
        baseline_comparison = {
            "diebold_mariano_statistic": statistic,
            "two_sided_p_value": float(2 * stats.norm.sf(abs(statistic))),
            "candidate_mean_squared_error": float(np.mean((predicted - actual) ** 2)),
            "baseline_mean_squared_error": float(np.mean((baseline - actual) ** 2)),
        }
    groups = inputs.get("fold_ids")
    fold_metrics = []
    if groups is not None:
        group_values = [str(item) for item in _sequence(groups, "inputs.fold_ids")]
        if len(group_values) != actual.size:
            raise ComputeError("fold_ids must match actual length")
        for group in sorted(set(group_values)):
            mask = np.asarray([item == group for item in group_values], dtype=bool)
            fold_error = error[mask]
            fold_metrics.append({
                "fold": group,
                "count": int(np.sum(mask)),
                "rmse": float(np.sqrt(np.mean(fold_error**2))),
                "mae": float(np.mean(np.abs(fold_error))),
                "bias": float(np.mean(fold_error)),
            })
    result = _base("forecast_accuracy")
    result.update({
        "observation_count": int(actual.size),
        "metrics": metrics,
        "quantile_pinball_loss": quantile_losses,
        "quantile_crps_approximation": None if not quantile_pairs else float(2 * np.mean([loss for _, loss in quantile_pairs])),
        "interval": interval,
        "baseline_comparison": baseline_comparison,
        "fold_metrics": fold_metrics,
    })
    return result


def bayesian_linear_calibration(inputs: Mapping[str, Any]) -> dict[str, Any]:
    features = _matrix(inputs.get("features"), "inputs.features", 3, 50_000, 100)
    observations = _vector(inputs.get("observations"), "inputs.observations", features.shape[0], features.shape[0])
    if features.shape[0] != observations.size:
        raise ComputeError("features and observations row counts must match")
    include_intercept = bool(inputs.get("include_intercept", True))
    design = np.column_stack([np.ones(features.shape[0]), features]) if include_intercept else features
    dimension = design.shape[1]
    prior_mean = np.zeros(dimension) if inputs.get("prior_mean") is None else _vector(inputs.get("prior_mean"), "inputs.prior_mean", dimension, dimension)
    prior_precision = _finite(inputs.get("prior_precision", 1e-6), "inputs.prior_precision", 1e-12)
    alpha0 = _finite(inputs.get("alpha0", 1e-3), "inputs.alpha0", 1e-12)
    beta0 = _finite(inputs.get("beta0", 1e-3), "inputs.beta0", 1e-12)
    precision0 = np.eye(dimension) * prior_precision
    precision_n = precision0 + design.T @ design
    covariance_n = np.linalg.pinv(precision_n)
    mean_n = covariance_n @ (precision0 @ prior_mean + design.T @ observations)
    alpha_n = alpha0 + observations.size / 2
    beta_n = beta0 + 0.5 * (
        observations @ observations
        + prior_mean @ precision0 @ prior_mean
        - mean_n @ precision_n @ mean_n
    )
    if beta_n <= 0:
        raise ComputeError("posterior scale is non-positive")
    fitted = design @ mean_n
    residual = fitted - observations
    posterior_sigma2_mean = float(beta_n / max(alpha_n - 1, 1e-12))
    coefficient_covariance = covariance_n * posterior_sigma2_mean
    prediction = None
    if inputs.get("prediction_features") is not None:
        x_new = _matrix(inputs.get("prediction_features"), "inputs.prediction_features", 1, 50_000, features.shape[1])
        if x_new.shape[1] != features.shape[1]:
            raise ComputeError("prediction_features column count must match features")
        new_design = np.column_stack([np.ones(x_new.shape[0]), x_new]) if include_intercept else x_new
        predictive_mean = new_design @ mean_n
        predictive_variance = (beta_n / alpha_n) * (1 + np.sum((new_design @ covariance_n) * new_design, axis=1))
        quantile = float(stats.t.ppf(0.975, df=2 * alpha_n))
        scale = np.sqrt(np.maximum(predictive_variance, 0))
        prediction = {
            "mean": predictive_mean.tolist(),
            "lower_95": (predictive_mean - quantile * scale).tolist(),
            "upper_95": (predictive_mean + quantile * scale).tolist(),
        }
    result = _base("bayesian_linear_calibration")
    result.update({
        "observations": int(observations.size),
        "parameters": int(dimension),
        "posterior_mean": mean_n.tolist(),
        "posterior_covariance": coefficient_covariance.tolist(),
        "posterior_alpha": float(alpha_n),
        "posterior_beta": float(beta_n),
        "posterior_noise_variance_mean": posterior_sigma2_mean,
        "calibration_metrics": {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
        },
        "prediction": prediction,
        "fixed_conjugate_model": True,
    })
    return result


def reliability_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    method = str(inputs.get("method") or "monte_carlo")
    result = _base("reliability_analysis")
    if method == "form_linear":
        means = _vector(inputs.get("means"), "inputs.means", 1, 100)
        covariance = _matrix(inputs.get("covariance"), "inputs.covariance", means.size, means.size, means.size)
        coefficients = _vector(inputs.get("coefficients"), "inputs.coefficients", means.size, means.size)
        if covariance.shape != (means.size, means.size):
            raise ComputeError("covariance shape must match means")
        threshold = _finite(inputs.get("threshold"), "inputs.threshold")
        direction = str(inputs.get("failure_when") or "below")
        mean_response = float(coefficients @ means)
        standard_deviation = math.sqrt(max(float(coefficients @ covariance @ coefficients), 0.0))
        if standard_deviation <= 0:
            raise ComputeError("linear limit-state standard deviation must be positive")
        if direction == "below":
            beta = (mean_response - threshold) / standard_deviation
        elif direction == "above":
            beta = (threshold - mean_response) / standard_deviation
        else:
            raise ComputeError("failure_when must be below or above")
        result.update({
            "method": method,
            "reliability_index_beta": float(beta),
            "failure_probability": float(stats.norm.cdf(-beta)),
            "mean_response": mean_response,
            "response_standard_deviation": standard_deviation,
        })
        return result
    if method != "monte_carlo":
        raise ComputeError("method must be monte_carlo or form_linear")
    values = _vector(inputs.get("limit_state_values"), "inputs.limit_state_values", 10, MAX_ROWS)
    failures = values <= 0
    count = int(values.size)
    failure_count = int(np.sum(failures))
    probability = failure_count / count
    z = 1.96
    denominator = 1 + z**2 / count
    center = (probability + z**2 / (2 * count)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / count + z**2 / (4 * count**2)) / denominator
    clipped_probability = min(max(probability, 1e-12), 1 - 1e-12)
    importance = None
    if inputs.get("factors") is not None:
        factors = _matrix(inputs.get("factors"), "inputs.factors", count, count, MAX_FACTORS)
        if factors.shape[0] != count:
            raise ComputeError("factors row count must match limit_state_values")
        names_raw = inputs.get("factor_names") or [f"factor_{index + 1}" for index in range(factors.shape[1])]
        names = [str(item) for item in _sequence(names_raw, "inputs.factor_names")]
        if len(names) != factors.shape[1] or len(set(names)) != len(names):
            raise ComputeError("factor_names must uniquely match factors columns")
        indicator = failures.astype(float)
        importance = {}
        for index, name in enumerate(names):
            correlation = float(np.corrcoef(factors[:, index], indicator)[0, 1]) if np.std(factors[:, index]) > 0 and np.std(indicator) > 0 else 0.0
            importance[name] = correlation if math.isfinite(correlation) else 0.0
    result.update({
        "method": method,
        "sample_count": count,
        "failure_count": failure_count,
        "failure_probability": float(probability),
        "failure_probability_wilson_95": [float(max(0, center - radius)), float(min(1, center + radius))],
        "reliability_index_beta": float(-stats.norm.ppf(clipped_probability)),
        "mean_limit_state": float(np.mean(values)),
        "minimum_limit_state": float(np.min(values)),
        "failure_tail_mean": None if not failure_count else float(np.mean(values[failures])),
        "factor_failure_correlations": importance,
    })
    return result


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    value = float(stats.spearmanr(x, y).statistic)
    return value if math.isfinite(value) else 0.0


def _cross_sectional_residualize(factor: np.ndarray, controls: np.ndarray | None) -> np.ndarray:
    if controls is None:
        return factor
    design = np.column_stack([np.ones(factor.size), controls])
    coefficients = np.linalg.pinv(design) @ factor
    return factor - design @ coefficients


def factor_information_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    returns = _matrix(inputs.get("forward_returns"), "inputs.forward_returns", 2, 5000, 1000)
    factors_raw = _mapping(inputs.get("factors"), "inputs.factors")
    if not 1 <= len(factors_raw) <= MAX_FACTORS:
        raise ComputeError(f"factors must contain 1 to {MAX_FACTORS} entries")
    factor_names = [str(name) for name in factors_raw]
    if any(not name for name in factor_names) or len(set(factor_names)) != len(factor_names):
        raise ComputeError("factor names must be unique and non-empty")
    factors = {name: _matrix(factors_raw[name], f"inputs.factors[{name}]", returns.shape[0], returns.shape[0], returns.shape[1]) for name in factor_names}
    if any(matrix.shape != returns.shape for matrix in factors.values()):
        raise ComputeError("every factor matrix must match forward_returns")
    quantiles = _integer(inputs.get("quantiles", 5), "inputs.quantiles", 2, 20)
    controls = None
    if inputs.get("controls") is not None:
        raw_controls = _mapping(inputs.get("controls"), "inputs.controls")
        control_matrices = [_matrix(value, f"inputs.controls[{name}]", returns.shape[0], returns.shape[0], returns.shape[1]) for name, value in raw_controls.items()]
        if any(matrix.shape != returns.shape for matrix in control_matrices):
            raise ComputeError("all controls must match forward_returns")
        controls = np.stack(control_matrices, axis=2) if control_matrices else None
    regimes = None
    if inputs.get("regimes") is not None:
        regimes = [str(item) for item in _sequence(inputs.get("regimes"), "inputs.regimes")]
        if len(regimes) != returns.shape[0]:
            raise ComputeError("regimes length must match time rows")
    rows = []
    for name in factor_names:
        matrix = factors[name]
        ic_values = []
        rank_ic_values = []
        bucket_returns = [[] for _ in range(quantiles)]
        top_memberships = []
        regime_values: dict[str, list[float]] = {}
        for time_index in range(returns.shape[0]):
            factor = matrix[time_index]
            outcome = returns[time_index]
            control = controls[time_index] if controls is not None else None
            factor = _cross_sectional_residualize(factor, control)
            if np.std(factor) <= 1e-15 or np.std(outcome) <= 1e-15:
                ic = 0.0
                rank_ic = 0.0
            else:
                ic = float(np.corrcoef(factor, outcome)[0, 1])
                rank_ic = _rank_correlation(factor, outcome)
            ic_values.append(ic if math.isfinite(ic) else 0.0)
            rank_ic_values.append(rank_ic)
            if regimes is not None:
                regime_values.setdefault(regimes[time_index], []).append(rank_ic)
            order = np.argsort(factor, kind="mergesort")
            buckets = np.array_split(order, quantiles)
            for bucket_index, bucket in enumerate(buckets):
                bucket_returns[bucket_index].append(float(np.mean(outcome[bucket])))
            top_memberships.append(set(map(int, buckets[-1])))
        average_ic = float(np.mean(ic_values))
        average_rank_ic = float(np.mean(rank_ic_values))
        rank_ic_sd = float(np.std(rank_ic_values, ddof=1)) if len(rank_ic_values) > 1 else 0.0
        average_bucket_returns = [float(np.mean(values)) for values in bucket_returns]
        turnover = []
        for previous, current in zip(top_memberships[:-1], top_memberships[1:], strict=True):
            union = len(previous | current)
            turnover.append(1 - len(previous & current) / union if union else 0.0)
        rows.append({
            "factor": name,
            "mean_information_coefficient": average_ic,
            "mean_rank_information_coefficient": average_rank_ic,
            "rank_ic_standard_deviation": rank_ic_sd,
            "rank_ic_information_ratio": None if rank_ic_sd <= 1e-15 else average_rank_ic / rank_ic_sd,
            "positive_rank_ic_rate": float(np.mean(np.asarray(rank_ic_values) > 0)),
            "quantile_mean_returns": average_bucket_returns,
            "quantile_monotonicity": _rank_correlation(np.arange(quantiles, dtype=float), np.asarray(average_bucket_returns)),
            "top_quantile_turnover": float(np.mean(turnover)) if turnover else 0.0,
            "regime_rank_ic": {key: float(np.mean(values)) for key, values in sorted(regime_values.items())},
            "neutralized": controls is not None,
        })
    rows.sort(key=lambda row: (-abs(row["mean_rank_information_coefficient"]), row["factor"]))
    result = _base("factor_information_analysis")
    result.update({
        "time_periods": int(returns.shape[0]),
        "assets_per_period": int(returns.shape[1]),
        "factors": rows,
        "selection_warning": "Information coefficients are research evidence, not guaranteed returns.",
    })
    return result


def _lasso_coordinate_descent(x: np.ndarray, y: np.ndarray, penalty: float, iterations: int = 5000) -> np.ndarray:
    coefficients = np.zeros(x.shape[1])
    column_norms = np.sum(x**2, axis=0)
    for _ in range(iterations):
        previous = coefficients.copy()
        for index in range(x.shape[1]):
            residual = y - x @ coefficients + x[:, index] * coefficients[index]
            rho = float(x[:, index] @ residual)
            if rho < -penalty:
                coefficients[index] = (rho + penalty) / max(column_norms[index], 1e-15)
            elif rho > penalty:
                coefficients[index] = (rho - penalty) / max(column_norms[index], 1e-15)
            else:
                coefficients[index] = 0.0
        if np.max(np.abs(coefficients - previous)) < 1e-10:
            break
    return coefficients


def factor_selection_diagnostics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _matrix(inputs.get("factor_matrix"), "inputs.factor_matrix", 20, 50_000, MAX_FACTORS)
    target = _vector(inputs.get("target"), "inputs.target", matrix.shape[0], matrix.shape[0])
    if target.size != matrix.shape[0]:
        raise ComputeError("target length must match factor_matrix rows")
    raw_names = inputs.get("factor_names") or [f"factor_{index + 1}" for index in range(matrix.shape[1])]
    names = [str(item) for item in _sequence(raw_names, "inputs.factor_names")]
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ComputeError("factor_names must uniquely match factor_matrix columns")
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0, ddof=1)
    if np.any(scales <= 1e-15):
        raise ComputeError("factor_matrix contains a constant factor")
    x = (matrix - means) / scales
    y = target - np.mean(target)
    correlation = np.corrcoef(x, rowvar=False)
    inverse_correlation = np.linalg.pinv(correlation)
    vif = np.diag(inverse_correlation)
    covariance = np.cov(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / max(float(np.sum(eigenvalues)), 1e-15)
    cumulative = np.cumsum(explained)
    threshold = _finite(inputs.get("pca_variance_threshold", 0.9), "inputs.pca_variance_threshold", 0.5, 0.999999)
    components_needed = int(np.searchsorted(cumulative, threshold) + 1)
    ridge_penalty = _finite(inputs.get("ridge_penalty", 1.0), "inputs.ridge_penalty", 0.0)
    ridge = np.linalg.pinv(x.T @ x + ridge_penalty * np.eye(x.shape[1])) @ x.T @ y
    lasso_penalty = _finite(inputs.get("lasso_penalty", 0.1), "inputs.lasso_penalty", 0.0)
    lasso = _lasso_coordinate_descent(x, y, lasso_penalty)
    univariate = []
    p_values = []
    for index, name in enumerate(names):
        slope, intercept, r_value, p_value, standard_error = stats.linregress(x[:, index], target)
        p_values.append(float(p_value))
        univariate.append({
            "factor": name,
            "coefficient": float(slope),
            "p_value": float(p_value),
            "correlation": float(r_value),
            "standard_error": float(standard_error),
        })
    order_p = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 1.0
    for reverse_rank, index in enumerate(order_p[::-1], start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    fdr = _finite(inputs.get("fdr_level", 0.05), "inputs.fdr_level", 1e-6, 0.5)
    for index, row in enumerate(univariate):
        row["fdr_adjusted_p_value"] = float(min(adjusted[index], 1.0))
        row["passes_fdr"] = bool(adjusted[index] <= fdr)
        row["vif"] = float(vif[index])
        row["ridge_coefficient"] = float(ridge[index])
        row["lasso_coefficient"] = float(lasso[index])
    result = _base("factor_selection_diagnostics")
    result.update({
        "observations": int(matrix.shape[0]),
        "factor_count": int(matrix.shape[1]),
        "factor_diagnostics": univariate,
        "selected_by_fdr": [row["factor"] for row in univariate if row["passes_fdr"]],
        "selected_by_lasso": [names[index] for index, value in enumerate(lasso) if abs(value) > 1e-12],
        "high_vif_factors": [names[index] for index, value in enumerate(vif) if value > 10],
        "pca": {
            "explained_variance_ratio": explained.tolist(),
            "components_for_threshold": components_needed,
            "variance_threshold": threshold,
            "loadings": eigenvectors[:, :components_needed].tolist(),
        },
        "factor_correlation_matrix": correlation.tolist(),
        "crowding_proxy_mean_absolute_correlation": float(np.mean(np.abs(correlation[np.triu_indices_from(correlation, k=1)]))) if matrix.shape[1] > 1 else 0.0,
    })
    return result


def _block_bootstrap_indices(rng: np.random.Generator, rows: int, block_length: int) -> np.ndarray:
    starts = rng.integers(0, rows, size=math.ceil(rows / block_length))
    indices = np.concatenate([np.arange(start, start + block_length) % rows for start in starts])
    return indices[:rows]


def factor_overfit_diagnostics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    returns = _matrix(inputs.get("strategy_returns"), "inputs.strategy_returns", 40, 20_000, MAX_STRATEGIES)
    if returns.shape[1] < 2:
        raise ComputeError("strategy_returns must contain at least two strategies")
    blocks = _integer(inputs.get("blocks", 8), "inputs.blocks", 4, min(MAX_BLOCKS, returns.shape[0]))
    if blocks % 2:
        raise ComputeError("blocks must be even")
    split_rows = np.array_split(np.arange(returns.shape[0]), blocks)
    combinations_list = list(combinations(range(blocks), blocks // 2))
    max_combinations = _integer(inputs.get("max_combinations", 500), "inputs.max_combinations", 1, 2000)
    if len(combinations_list) > max_combinations:
        step = len(combinations_list) / max_combinations
        combinations_list = [combinations_list[min(int(index * step), len(combinations_list) - 1)] for index in range(max_combinations)]
    logits = []
    below_median = 0
    for selected_blocks in combinations_list:
        selected = set(selected_blocks)
        train_indices = np.concatenate([split_rows[index] for index in selected])
        test_indices = np.concatenate([split_rows[index] for index in range(blocks) if index not in selected])
        train_score = np.mean(returns[train_indices], axis=0) / np.maximum(np.std(returns[train_indices], axis=0, ddof=1), 1e-12)
        best = int(np.argmax(train_score))
        test_score = np.mean(returns[test_indices], axis=0) / np.maximum(np.std(returns[test_indices], axis=0, ddof=1), 1e-12)
        rank = int(np.sum(test_score <= test_score[best]))
        relative_rank = (rank - 0.5) / returns.shape[1]
        relative_rank = min(max(relative_rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(relative_rank / (1 - relative_rank)))
        below_median += int(relative_rank <= 0.5)
    pbo = below_median / len(combinations_list)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    bootstraps = _integer(inputs.get("bootstraps", 500), "inputs.bootstraps", 100, MAX_BOOTSTRAPS)
    block_length = _integer(inputs.get("bootstrap_block_length", max(1, int(round(returns.shape[0] ** (1 / 3))))), "inputs.bootstrap_block_length", 1, returns.shape[0])
    observed_means = np.mean(returns, axis=0)
    observed_max = float(np.max(observed_means))
    centered = returns - observed_means
    rng = np.random.default_rng(seed)
    bootstrap_max = np.empty(bootstraps)
    for index in range(bootstraps):
        indices = _block_bootstrap_indices(rng, returns.shape[0], block_length)
        bootstrap_max[index] = float(np.max(np.mean(centered[indices], axis=0)))
    reality_p = float((1 + np.sum(bootstrap_max >= observed_max)) / (bootstraps + 1))
    result = _base("factor_overfit_diagnostics")
    result.update({
        "observations": int(returns.shape[0]),
        "strategy_count": int(returns.shape[1]),
        "cscv_combinations": len(combinations_list),
        "probability_of_backtest_overfitting": float(pbo),
        "median_logit_oos_rank": float(np.median(logits)),
        "white_reality_check": {
            "observed_best_mean_return": observed_max,
            "bootstrap_p_value": reality_p,
            "bootstraps": bootstraps,
            "block_length": block_length,
            "seed": seed,
        },
        "overfit_gate_passed": bool(pbo < _finite(inputs.get("maximum_pbo", 0.5), "inputs.maximum_pbo", 0.0, 1.0) and reality_p <= _finite(inputs.get("maximum_reality_check_p", 0.1), "inputs.maximum_reality_check_p", 0.0, 1.0)),
    })
    return result


def cross_validation_plan(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = _integer(inputs.get("rows"), "inputs.rows", 10, MAX_ROWS)
    strategy = str(inputs.get("strategy") or "rolling")
    splits = _integer(inputs.get("splits", 5), "inputs.splits", 2, 20)
    folds: list[dict[str, Any]] = []
    indices = np.arange(rows)
    if strategy in {"rolling", "expanding"}:
        test_size = _integer(inputs.get("test_size", max(1, rows // (splits + 1))), "inputs.test_size", 1, rows // 2)
        minimum_train = _integer(inputs.get("minimum_train_size", max(test_size, rows - splits * test_size)), "inputs.minimum_train_size", 2, rows - test_size)
        for index in range(splits):
            test_start = minimum_train + index * test_size
            test_end = min(rows, test_start + test_size)
            if test_start >= rows:
                break
            train_start = max(0, test_start - minimum_train) if strategy == "rolling" else 0
            folds.append({"fold": len(folds) + 1, "train_indices": indices[train_start:test_start].tolist(), "test_indices": indices[test_start:test_end].tolist()})
    elif strategy in {"grouped", "spatial_block"}:
        labels_raw = inputs.get("groups") if strategy == "grouped" else inputs.get("spatial_blocks")
        labels = [str(item) for item in _sequence(labels_raw, f"inputs.{ 'groups' if strategy == 'grouped' else 'spatial_blocks'}")]
        if len(labels) != rows:
            raise ComputeError("group labels must match rows")
        unique = sorted(set(labels))
        if len(unique) < splits:
            raise ComputeError("not enough unique groups for requested splits")
        fold_groups = np.array_split(np.asarray(unique, dtype=object), splits)
        for index, selected in enumerate(fold_groups, start=1):
            selected_set = set(map(str, selected))
            test_mask = np.asarray([label in selected_set for label in labels], dtype=bool)
            folds.append({"fold": index, "train_indices": indices[~test_mask].tolist(), "test_indices": indices[test_mask].tolist(), "held_out_groups": sorted(selected_set)})
    elif strategy == "nested":
        outer = _integer(inputs.get("outer_splits", splits), "inputs.outer_splits", 2, 10)
        inner = _integer(inputs.get("inner_splits", 3), "inputs.inner_splits", 2, 10)
        outer_parts = np.array_split(indices, outer)
        for outer_index, test in enumerate(outer_parts):
            train = np.setdiff1d(indices, test, assume_unique=True)
            inner_parts = np.array_split(train, inner)
            inner_folds = []
            for inner_index, validation in enumerate(inner_parts):
                inner_train = np.setdiff1d(train, validation, assume_unique=True)
                inner_folds.append({"fold": inner_index + 1, "train_indices": inner_train.tolist(), "validation_indices": validation.tolist()})
            folds.append({"fold": outer_index + 1, "train_indices": train.tolist(), "test_indices": test.tolist(), "inner_folds": inner_folds})
    else:
        raise ComputeError("strategy must be rolling, expanding, grouped, spatial_block, or nested")
    if not folds:
        raise ComputeError("cross-validation plan produced no folds")
    for fold in folds:
        train = set(fold["train_indices"])
        test = set(fold["test_indices"])
        if train & test:
            raise ComputeError("cross-validation plan contains train/test leakage")
    result = _base("cross_validation_plan")
    result.update({
        "strategy": strategy,
        "rows": rows,
        "fold_count": len(folds),
        "folds": folds,
        "leakage_check_passed": True,
    })
    return result


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "joint_random_sample": joint_random_sample,
    "distribution_fit_select": distribution_fit_select,
    "variable_role_validate": variable_role_validate,
    "probabilistic_accuracy": probabilistic_accuracy,
    "forecast_accuracy": forecast_accuracy,
    "bayesian_linear_calibration": bayesian_linear_calibration,
    "reliability_analysis": reliability_analysis,
    "factor_information_analysis": factor_information_analysis,
    "factor_selection_diagnostics": factor_selection_diagnostics,
    "factor_overfit_diagnostics": factor_overfit_diagnostics,
    "cross_validation_plan": cross_validation_plan,
}
