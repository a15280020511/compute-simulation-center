#!/usr/bin/env python3
"""Commercial, operating, consumer and industrial analysis modes."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, integer, matrix, package, probability, sequence, vector


def price_elasticity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    import statsmodels.api as sm

    price = vector(inputs.get("price"), "inputs.price", minimum=10)
    quantity = vector(inputs.get("quantity"), "inputs.quantity", minimum=10)
    if price.size != quantity.size or np.any(price <= 0) or np.any(quantity <= 0):
        raise ComputeError("price and quantity must be positive and have equal length")
    result = sm.OLS(np.log(quantity), sm.add_constant(np.log(price), has_constant="add")).fit(cov_type="HC3")
    elasticity = float(result.params[1])
    return {
        "mode": "price_elasticity",
        "elasticity": elasticity,
        "standard_error": float(result.bse[1]),
        "p_value": float(result.pvalues[1]),
        "r_squared": float(result.rsquared),
        "interpretation": "elastic" if abs(elasticity) > 1 else "inelastic",
    }


def customer_lifetime_value(inputs: Mapping[str, Any]) -> dict[str, Any]:
    margin = finite(inputs.get("period_margin"), "inputs.period_margin")
    retention = probability(inputs.get("retention_rate"), "inputs.retention_rate")
    discount = finite(inputs.get("discount_rate"), "inputs.discount_rate")
    acquisition = finite(inputs.get("acquisition_cost", 0.0), "inputs.acquisition_cost")
    periods = integer(inputs.get("periods", 36), "inputs.periods", 1, 600)
    if discount <= -1:
        raise ComputeError("discount_rate must be greater than -1")
    cashflows = [margin * retention**period / (1 + discount) ** period for period in range(1, periods + 1)]
    return {
        "mode": "customer_lifetime_value",
        "gross_present_value": float(sum(cashflows)),
        "net_lifetime_value": float(sum(cashflows) - acquisition),
        "periods": periods,
        "cashflows": [float(v) for v in cashflows],
    }


def customer_segmentation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("scikit-learn")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = matrix(inputs.get("features"), "inputs.features", min_rows=10)
    clusters = integer(inputs.get("clusters", 3), "inputs.clusters", 2, min(20, data.shape[0] - 1))
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    scaled = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=20)
    labels = model.fit_predict(scaled)
    return {
        "mode": "customer_segmentation",
        "labels": [int(v) for v in labels],
        "cluster_sizes": {str(i): int(np.sum(labels == i)) for i in range(clusters)},
        "centers_standardized": model.cluster_centers_.tolist(),
        "inertia": float(model.inertia_),
        "engine": {"scikit-learn": package("scikit-learn")},
    }


def churn_probability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("scikit-learn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x = matrix(inputs.get("features"), "inputs.features", min_rows=30)
    y = vector(inputs.get("churned"), "inputs.churned", minimum=30)
    if x.shape[0] != y.size or np.any((y != 0) & (y != 1)):
        raise ComputeError("features and binary churned labels must match")
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y.astype(int), test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=1_000, random_state=seed).fit(scaler.transform(x_train), y_train)
    probability_values = model.predict_proba(scaler.transform(x_test))[:, 1]
    return {
        "mode": "churn_probability",
        "roc_auc_holdout": float(roc_auc_score(y_test, probability_values)),
        "brier_holdout": float(brier_score_loss(y_test, probability_values)),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "holdout_probabilities": probability_values.tolist(),
        "holdout_labels": y_test.tolist(),
        "engine": {"scikit-learn": package("scikit-learn")},
    }


def marketing_mix_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("scikit-learn")
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit

    x = matrix(inputs.get("channels"), "inputs.channels", min_rows=30)
    y = vector(inputs.get("outcome"), "inputs.outcome", minimum=30)
    if x.shape[0] != y.size:
        raise ComputeError("channels and outcome row counts must match")
    alphas = [finite(v, f"inputs.alphas[{i}]") for i, v in enumerate(sequence(inputs.get("alphas", [0.1, 1.0, 10.0, 100.0]), "inputs.alphas"))]
    if any(v <= 0 for v in alphas):
        raise ComputeError("alphas must be positive")
    split = max(10, int(0.8 * y.size))
    model = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5)).fit(x[:split], y[:split])
    predicted = model.predict(x[split:])
    return {
        "mode": "marketing_mix_regression",
        "selected_alpha": float(model.alpha_),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "holdout_rmse": float(math.sqrt(mean_squared_error(y[split:], predicted))),
        "engine": {"scikit-learn": package("scikit-learn")},
        "causal_claim_allowed": False,
    }


def inventory_policy(inputs: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.stats import norm

    demand = finite(inputs.get("annual_demand"), "inputs.annual_demand")
    order_cost = finite(inputs.get("order_cost"), "inputs.order_cost")
    holding_cost = finite(inputs.get("holding_cost_per_unit"), "inputs.holding_cost_per_unit")
    lead_mean = finite(inputs.get("lead_time_demand_mean"), "inputs.lead_time_demand_mean")
    lead_sd = finite(inputs.get("lead_time_demand_sd"), "inputs.lead_time_demand_sd")
    service = probability(inputs.get("service_level", 0.95), "inputs.service_level")
    if min(demand, order_cost, holding_cost) <= 0 or lead_sd < 0:
        raise ComputeError("demand, order cost and holding cost must be positive; lead_time_demand_sd non-negative")
    eoq = math.sqrt(2 * demand * order_cost / holding_cost)
    safety_stock = float(norm.ppf(service) * lead_sd)
    return {
        "mode": "inventory_policy",
        "economic_order_quantity": eoq,
        "safety_stock": safety_stock,
        "reorder_point": lead_mean + safety_stock,
        "service_level": service,
    }


def input_output_shock(inputs: Mapping[str, Any]) -> dict[str, Any]:
    pymrio_version = package("pymrio")
    coefficients = matrix(
        inputs.get("technical_coefficients"), "inputs.technical_coefficients", max_rows=100, max_columns=100
    )
    demand = vector(inputs.get("final_demand"), "inputs.final_demand", maximum=100)
    shock = vector(inputs.get("demand_shock"), "inputs.demand_shock", maximum=100)
    if coefficients.shape[0] != coefficients.shape[1] or coefficients.shape[0] != demand.size or demand.size != shock.size:
        raise ComputeError("input-output dimensions must be square and aligned")
    try:
        inverse = np.linalg.inv(np.eye(coefficients.shape[0]) - coefficients)
    except np.linalg.LinAlgError as exc:
        raise ComputeError("Leontief matrix is singular") from exc
    baseline = inverse @ demand
    shocked = inverse @ (demand + shock)
    return {
        "mode": "input_output_shock",
        "baseline_output": baseline.tolist(),
        "shocked_output": shocked.tolist(),
        "output_change": (shocked - baseline).tolist(),
        "total_multiplier": float(np.sum(shocked - baseline) / max(abs(np.sum(shock)), 1e-12)),
        "engine": {"pymrio": pymrio_version, "solver": "audited Leontief inverse"},
    }


def consumer_choice_logit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    import statsmodels.api as sm

    x = matrix(inputs.get("features"), "inputs.features", min_rows=30)
    choices = vector(inputs.get("choices"), "inputs.choices", minimum=30)
    if x.shape[0] != choices.size or np.any(choices < 0) or np.any(choices != np.floor(choices)):
        raise ComputeError("features and non-negative integer choices must align")
    if len(np.unique(choices)) < 2:
        raise ComputeError("choices must contain at least two alternatives")
    try:
        fit = sm.MNLogit(choices.astype(int), sm.add_constant(x, has_constant="add")).fit(
            method="newton", maxiter=300, disp=False
        )
    except Exception as exc:
        raise ComputeError(f"consumer choice model failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "consumer_choice_logit",
        "coefficients": np.asarray(fit.params, dtype=float).tolist(),
        "standard_errors": np.asarray(fit.bse, dtype=float).tolist(),
        "pseudo_r_squared": float(fit.prsquared),
        "aic": float(fit.aic),
    }


def process_capability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = vector(inputs.get("values"), "inputs.values", minimum=10)
    lower = finite(inputs.get("lower_specification"), "inputs.lower_specification")
    upper = finite(inputs.get("upper_specification"), "inputs.upper_specification")
    if not lower < upper:
        raise ComputeError("lower_specification must be below upper_specification")
    mean = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    if sigma <= 0:
        raise ComputeError("process variation must be positive")
    return {
        "mode": "process_capability",
        "mean": mean,
        "standard_deviation": sigma,
        "cp": (upper - lower) / (6 * sigma),
        "cpk": min((upper - mean) / (3 * sigma), (mean - lower) / (3 * sigma)),
        "outside_specification_rate": float(np.mean((values < lower) | (values > upper))),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "price_elasticity": price_elasticity,
    "customer_lifetime_value": customer_lifetime_value,
    "customer_segmentation": customer_segmentation,
    "churn_probability": churn_probability,
    "marketing_mix_regression": marketing_mix_regression,
    "inventory_policy": inventory_policy,
    "input_output_shock": input_output_shock,
    "consumer_choice_logit": consumer_choice_logit,
    "process_capability": process_capability,
}
