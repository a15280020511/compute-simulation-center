#!/usr/bin/env python3
"""Institutional econometric, survey, evidence-synthesis and policy modes."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, integer, matrix, package, probability, sequence, vector


def robust_glm(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    import statsmodels.api as sm

    x = matrix(inputs.get("x"), "inputs.x", min_rows=10)
    y = vector(inputs.get("y"), "inputs.y", minimum=10)
    if x.shape[0] != y.size:
        raise ComputeError("inputs.x and inputs.y row counts must match")
    family_name = str(inputs.get("family") or "gaussian")
    family_factories = {
        "gaussian": sm.families.Gaussian,
        "binomial": sm.families.Binomial,
        "poisson": sm.families.Poisson,
        "gamma": sm.families.Gamma,
    }
    if family_name == "negative_binomial":
        alpha = finite(inputs.get("dispersion", 1.0), "inputs.dispersion")
        if alpha <= 0:
            raise ComputeError("inputs.dispersion must be positive")
        family = sm.families.NegativeBinomial(alpha=alpha)
    else:
        factory = family_factories.get(family_name)
        if factory is None:
            raise ComputeError("unsupported GLM family")
        family = factory()
    cov_type = str(inputs.get("cov_type") or "HC3")
    if cov_type not in {"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"}:
        raise ComputeError("unsupported covariance type")
    fit_kwargs: dict[str, Any] = {"cov_type": cov_type}
    if cov_type == "HAC":
        fit_kwargs["cov_kwds"] = {"maxlags": integer(inputs.get("max_lags", 1), "inputs.max_lags", 1, 20)}
    try:
        fitted = sm.GLM(y, sm.add_constant(x, has_constant="add"), family=family).fit(**fit_kwargs)
    except Exception as exc:
        raise ComputeError(f"GLM fitting failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "robust_glm",
        "family": family_name,
        "covariance_type": cov_type,
        "coefficients": [float(v) for v in fitted.params],
        "standard_errors": [float(v) for v in fitted.bse],
        "p_values": [float(v) for v in fitted.pvalues],
        "deviance": float(fitted.deviance),
        "aic": float(fitted.aic),
        "engine": {"statsmodels": package("statsmodels")},
    }


def panel_fixed_effects(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("linearmodels")
    import pandas as pd
    from linearmodels.panel import PanelOLS

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20)
    y = vector(inputs.get("y"), "inputs.y", minimum=20)
    entity = [str(v) for v in sequence(inputs.get("entity"), "inputs.entity")]
    raw_time = sequence(inputs.get("time"), "inputs.time")
    time = [finite(value, f"inputs.time[{index}]") for index, value in enumerate(raw_time)]
    if not (x.shape[0] == y.size == len(entity) == len(time)):
        raise ComputeError("panel arrays must have equal row counts")
    if len(set(zip(entity, time, strict=True))) != len(entity):
        raise ComputeError("panel entity-time observations must be unique")
    columns = [f"x{i + 1}" for i in range(x.shape[1])]
    index = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])
    frame = pd.DataFrame(x, index=index, columns=columns)
    dependent = pd.Series(y, index=index, name="y")
    try:
        result = PanelOLS(
            dependent,
            frame,
            entity_effects=bool(inputs.get("entity_effects", True)),
            time_effects=bool(inputs.get("time_effects", False)),
            drop_absorbed=True,
        ).fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        raise ComputeError(f"panel estimation failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "panel_fixed_effects",
        "coefficients": {str(k): float(v) for k, v in result.params.items()},
        "standard_errors": {str(k): float(v) for k, v in result.std_errors.items()},
        "p_values": {str(k): float(v) for k, v in result.pvalues.items()},
        "r_squared": float(result.rsquared),
        "entity_count": len(set(entity)),
        "time_count": len(set(time)),
        "engine": {"linearmodels": package("linearmodels")},
    }


def survey_weighted_estimation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    samplics_version = package("samplics")
    values = vector(inputs.get("values"), "inputs.values", minimum=5)
    weights = vector(inputs.get("weights"), "inputs.weights", minimum=5)
    if values.size != weights.size or np.any(weights <= 0):
        raise ComputeError("values and positive weights must have equal length")
    total_weight = float(np.sum(weights))
    mean = float(np.sum(weights * values) / total_weight)
    centered = values - mean
    effective_n = float(total_weight**2 / np.sum(weights**2))
    variance = float(np.sum(weights * centered**2) / total_weight)
    standard_error = math.sqrt(variance / max(effective_n, 1.0))
    return {
        "mode": "survey_weighted_estimation",
        "weighted_mean": mean,
        "standard_error": standard_error,
        "effective_sample_size": effective_n,
        "design_effect_weighting": float(values.size / max(effective_n, 1.0)),
        "engine": {"samplics": samplics_version, "estimator": "audited weighted Taylor approximation"},
    }


def meta_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    from statsmodels.stats.meta_analysis import combine_effects

    effects = vector(inputs.get("effects"), "inputs.effects", minimum=2, maximum=500)
    variances = vector(inputs.get("variances"), "inputs.variances", minimum=2, maximum=500)
    if effects.size != variances.size or np.any(variances <= 0):
        raise ComputeError("effects and positive variances must have equal length")
    method = str(inputs.get("method_re") or "iterated")
    if method not in {"iterated", "chi2"}:
        raise ComputeError("inputs.method_re must be iterated or chi2")
    try:
        result = combine_effects(effects, variances, method_re=method, use_t=False)
    except Exception as exc:
        raise ComputeError(f"meta-analysis failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "meta_analysis",
        "studies": int(effects.size),
        "method_re": method,
        "fixed_effect": float(result.mean_effect_fe),
        "random_effect": float(result.mean_effect_re),
        "tau_squared": float(result.tau2),
        "q_statistic": float(result.q),
        "i_squared": float(max(0.0, result.i2)),
        "engine": {"statsmodels": package("statsmodels")},
    }


def survival_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("lifelines")
    import pandas as pd
    from lifelines import CoxPHFitter

    durations = vector(inputs.get("durations"), "inputs.durations", minimum=10)
    events = vector(inputs.get("events"), "inputs.events", minimum=10)
    covariates = matrix(inputs.get("covariates"), "inputs.covariates", min_rows=10)
    if durations.size != events.size or covariates.shape[0] != durations.size:
        raise ComputeError("survival arrays must have equal row counts")
    if np.any(durations <= 0) or np.any((events != 0) & (events != 1)):
        raise ComputeError("durations must be positive and events must be binary")
    frame = pd.DataFrame(covariates, columns=[f"x{i + 1}" for i in range(covariates.shape[1])])
    frame["duration"] = durations
    frame["event"] = events.astype(int)
    try:
        model = CoxPHFitter(penalizer=finite(inputs.get("penalizer", 0.01), "inputs.penalizer"))
        model.fit(frame, duration_col="duration", event_col="event", robust=True)
    except Exception as exc:
        raise ComputeError(f"survival fitting failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "survival_analysis",
        "hazard_ratios": {str(k): float(v) for k, v in model.hazard_ratios_.items()},
        "coefficients": {str(k): float(v) for k, v in model.params_.items()},
        "concordance_index": float(model.concordance_index_),
        "engine": {"lifelines": package("lifelines")},
    }


def change_point_detection(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("ruptures")
    import ruptures as rpt

    values = vector(inputs.get("values"), "inputs.values", minimum=20)
    model_name = str(inputs.get("cost_model") or "l2")
    if model_name not in {"l1", "l2", "rbf", "normal"}:
        raise ComputeError("unsupported change-point cost model")
    try:
        points = rpt.Pelt(model=model_name, min_size=3, jump=1).fit(values).predict(
            pen=finite(inputs.get("penalty", 5.0), "inputs.penalty")
        )
    except Exception as exc:
        raise ComputeError(f"change-point detection failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "change_point_detection",
        "change_points": [int(point) for point in points if point < values.size],
        "terminal_index": int(values.size),
        "engine": {"ruptures": package("ruptures")},
    }


def mixed_effects_model(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    import statsmodels.api as sm

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20)
    y = vector(inputs.get("y"), "inputs.y", minimum=20)
    groups = [str(v) for v in sequence(inputs.get("groups"), "inputs.groups")]
    if x.shape[0] != y.size or len(groups) != y.size or len(set(groups)) < 2:
        raise ComputeError("mixed-effects arrays must align and contain at least two groups")
    try:
        fit = sm.MixedLM(y, sm.add_constant(x, has_constant="add"), groups=groups).fit(
            reml=False, method="lbfgs", maxiter=500, disp=False
        )
    except Exception as exc:
        raise ComputeError(f"mixed-effects fitting failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "mixed_effects_model",
        "fixed_effects": [float(v) for v in fit.fe_params],
        "random_effect_variance": float(np.asarray(fit.cov_re)[0, 0]),
        "residual_variance": float(fit.scale),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "converged": bool(fit.converged),
    }


def quantile_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    import statsmodels.api as sm

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20)
    y = vector(inputs.get("y"), "inputs.y", minimum=20)
    quantile = probability(inputs.get("quantile", 0.5), "inputs.quantile")
    if quantile in {0.0, 1.0} or x.shape[0] != y.size:
        raise ComputeError("quantile must be strictly between 0 and 1 and rows must align")
    try:
        fit = sm.QuantReg(y, sm.add_constant(x, has_constant="add")).fit(q=quantile, max_iter=5_000)
    except Exception as exc:
        raise ComputeError(f"quantile regression failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "quantile_regression",
        "quantile": quantile,
        "coefficients": [float(v) for v in fit.params],
        "standard_errors": [float(v) for v in fit.bse],
        "p_values": [float(v) for v in fit.pvalues],
    }


def granger_causality(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    from statsmodels.tsa.stattools import grangercausalitytests

    cause = vector(inputs.get("cause"), "inputs.cause", minimum=30)
    effect = vector(inputs.get("effect"), "inputs.effect", minimum=30)
    max_lag = integer(inputs.get("max_lag", 5), "inputs.max_lag", 1, min(20, cause.size // 5))
    if cause.size != effect.size:
        raise ComputeError("cause and effect must have equal length")
    try:
        tests = grangercausalitytests(np.column_stack([effect, cause]), maxlag=max_lag, verbose=False)
    except Exception as exc:
        raise ComputeError(f"Granger test failed: {type(exc).__name__}: {exc}") from exc
    rows = []
    for lag, result in tests.items():
        statistic, p_value, _, _ = result[0]["ssr_ftest"]
        rows.append({"lag": int(lag), "f_statistic": float(statistic), "p_value": float(p_value)})
    return {
        "mode": "granger_causality",
        "lags": rows,
        "minimum_p_value": min(row["p_value"] for row in rows),
        "causal_claim_allowed": False,
    }


def power_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    from statsmodels.stats.power import TTestIndPower

    effect_size = finite(inputs.get("effect_size"), "inputs.effect_size")
    alpha = probability(inputs.get("alpha", 0.05), "inputs.alpha")
    power = probability(inputs.get("power", 0.8), "inputs.power")
    ratio = finite(inputs.get("group_ratio", 1.0), "inputs.group_ratio")
    if effect_size <= 0 or alpha in {0.0, 1.0} or power in {0.0, 1.0} or ratio <= 0:
        raise ComputeError("effect_size, ratio must be positive and alpha/power strictly between 0 and 1")
    sample = TTestIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio)
    return {
        "mode": "power_analysis",
        "sample_size_group_1": int(math.ceil(sample)),
        "sample_size_group_2": int(math.ceil(sample * ratio)),
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
    }


def unobserved_components_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    values = vector(inputs.get("values"), "inputs.values", minimum=30)
    horizon = integer(inputs.get("horizon", 6), "inputs.horizon", 1, min(365, values.size // 2))
    seasonal = inputs.get("seasonal")
    seasonal_period = None if seasonal in {None, 0} else integer(seasonal, "inputs.seasonal", 2, min(365, values.size // 2))
    try:
        fit = UnobservedComponents(
            values,
            level="local linear trend",
            seasonal=seasonal_period,
            autoregressive=1,
        ).fit(disp=False, maxiter=500)
        forecast = fit.get_forecast(horizon)
    except Exception as exc:
        raise ComputeError(f"unobserved-components model failed: {type(exc).__name__}: {exc}") from exc
    interval = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
    return {
        "mode": "unobserved_components_forecast",
        "forecast": np.asarray(forecast.predicted_mean, dtype=float).tolist(),
        "prediction_interval_95": interval.tolist(),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }


def markov_regime_model(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    values = vector(inputs.get("values"), "inputs.values", minimum=50)
    regimes = integer(inputs.get("regimes", 2), "inputs.regimes", 2, 4)
    try:
        fit = MarkovRegression(values, k_regimes=regimes, trend="c", switching_variance=True).fit(
            disp=False, maxiter=500
        )
    except Exception as exc:
        raise ComputeError(f"Markov regime model failed: {type(exc).__name__}: {exc}") from exc
    probabilities = np.asarray(fit.smoothed_marginal_probabilities, dtype=float)
    return {
        "mode": "markov_regime_model",
        "regimes": regimes,
        "smoothed_probabilities": probabilities.tolist(),
        "most_likely_regime": np.argmax(probabilities, axis=1).astype(int).tolist(),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "robust_glm": robust_glm,
    "panel_fixed_effects": panel_fixed_effects,
    "survey_weighted_estimation": survey_weighted_estimation,
    "meta_analysis": meta_analysis,
    "survival_analysis": survival_analysis,
    "change_point_detection": change_point_detection,
    "mixed_effects_model": mixed_effects_model,
    "quantile_regression": quantile_regression,
    "granger_causality": granger_causality,
    "power_analysis": power_analysis,
    "unobserved_components_forecast": unobserved_components_forecast,
    "markov_regime_model": markov_regime_model,
}
