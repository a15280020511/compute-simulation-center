#!/usr/bin/env python3
"""Fairness, explainability, label-quality, reliability and control-system modes."""
from __future__ import annotations

import math

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, integer, matrix, package, sequence, vector


def model_fairness_audit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("fairlearn")
    from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate, selection_rate
    from sklearn.metrics import accuracy_score

    y_true = np.asarray([int(v) for v in sequence(inputs.get("y_true"), "inputs.y_true")], dtype=int)
    y_pred = np.asarray([int(v) for v in sequence(inputs.get("y_pred"), "inputs.y_pred")], dtype=int)
    groups = np.asarray([str(v) for v in sequence(inputs.get("groups"), "inputs.groups")], dtype=object)
    if not 20 <= y_true.size <= 100_000 or y_pred.size != y_true.size or groups.size != y_true.size:
        raise ComputeError("labels and groups must align and contain 20 to 100000 rows")
    if set(np.unique(y_true)) - {0, 1} or set(np.unique(y_pred)) - {0, 1} or len(set(groups)) < 2:
        raise ComputeError("fairness audit requires binary labels and at least two groups")
    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
    overall = frame.overall.to_dict() if hasattr(frame.overall, "to_dict") else dict(frame.overall)
    by_group = frame.by_group
    difference = frame.difference()
    ratio = frame.ratio()
    return {
        "mode": "model_fairness_audit",
        "overall": {str(k): float(v) for k, v in overall.items()},
        "by_group": {
            str(group): {str(k): float(v) for k, v in row.to_dict().items()}
            for group, row in by_group.iterrows()
        },
        "difference": {str(k): float(v) for k, v in difference.to_dict().items()},
        "ratio": {
            str(k): 0.0 if not np.isfinite(float(v)) else float(v)
            for k, v in ratio.to_dict().items()
        },
        "engine": {"fairlearn": package("fairlearn")},
        "automated_individual_decision_allowed": False,
    }


def model_explainability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("shap")
    package("scikit-learn")
    import shap
    from sklearn.ensemble import RandomForestRegressor

    x = matrix(inputs.get("x"), "inputs.x", min_rows=30, max_rows=20_000, min_columns=1, max_columns=30)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    trees = integer(inputs.get("trees", 100), "inputs.trees", 20, 500)
    sample_count = min(integer(inputs.get("sample_count", 100), "inputs.sample_count", 1, 500), x.shape[0])
    model = RandomForestRegressor(n_estimators=trees, max_depth=integer(inputs.get("max_depth", 6), "inputs.max_depth", 1, 30), random_state=seed, n_jobs=1)
    model.fit(x, y)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x[:sample_count]), dtype=float)
    if values.ndim > 2:
        values = values.reshape(sample_count, x.shape[1], -1).mean(axis=2)
    mean_abs = np.mean(np.abs(values), axis=0)
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    return {
        "mode": "model_explainability",
        "feature_importance_mean_absolute_shap": {f"x{i}": float(v) for i, v in enumerate(mean_abs)},
        "base_value": float(expected[0]),
        "sample_explanations": values.tolist(),
        "training_r_squared": float(model.score(x, y)),
        "engine": {"shap": package("shap"), "scikit-learn": package("scikit-learn")},
        "causal_interpretation_allowed": False,
    }


def label_quality_audit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("cleanlab")
    package("scikit-learn")
    from cleanlab.rank import get_label_quality_scores
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = matrix(inputs.get("x"), "inputs.x", min_rows=50, max_rows=50_000, min_columns=1, max_columns=50)
    labels = np.asarray([int(v) for v in sequence(inputs.get("labels"), "inputs.labels")], dtype=int)
    if labels.size != x.shape[0] or len(np.unique(labels)) < 2 or np.min(labels) < 0:
        raise ComputeError("labels must align and contain at least two nonnegative classes")
    folds = integer(inputs.get("folds", 5), "inputs.folds", 2, 10)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    class_counts = np.bincount(labels)
    if np.min(class_counts[class_counts > 0]) < folds:
        raise ComputeError("each class must have at least folds observations")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(LogisticRegression(max_iter=2000, random_state=seed), x, labels, cv=cv, method="predict_proba", n_jobs=1)
    scores = np.asarray(get_label_quality_scores(labels=labels, pred_probs=probabilities), dtype=float)
    ranked = np.argsort(scores)
    top_n = min(integer(inputs.get("top_n", 50), "inputs.top_n", 1, 500), labels.size)
    threshold = finite(inputs.get("quality_threshold", 0.5), "inputs.quality_threshold")
    if not 0 <= threshold <= 1:
        raise ComputeError("quality_threshold must be between zero and one")
    flagged = np.where(scores < threshold)[0]
    return {
        "mode": "label_quality_audit",
        "flagged_count": int(flagged.size),
        "flagged_fraction": float(flagged.size / labels.size),
        "quality_threshold": threshold,
        "lowest_quality_rows": [
            {"row": int(i), "quality_score": float(scores[i]), "flagged": bool(scores[i] < threshold)}
            for i in ranked[:top_n]
        ],
        "engine": {"cleanlab": package("cleanlab"), "scikit-learn": package("scikit-learn")},
    }


def reliability_life_distribution(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("reliability")
    from reliability.Fitters import Fit_Weibull_2P

    failures = vector(inputs.get("failures"), "inputs.failures", minimum=5, maximum=50_000)
    raw_censored = inputs.get("right_censored")
    censored = [] if raw_censored in (None, []) else vector(raw_censored, "inputs.right_censored", minimum=1, maximum=50_000).tolist()
    if np.any(failures <= 0) or any(v <= 0 for v in censored):
        raise ComputeError("failure and censoring times must be positive")
    fit = Fit_Weibull_2P(failures=failures.tolist(), right_censored=censored, show_probability_plot=False, print_results=False)
    alpha = float(getattr(fit, "alpha"))
    beta = float(getattr(fit, "beta"))
    mission_time = finite(inputs.get("mission_time", float(np.median(failures))), "inputs.mission_time")
    if mission_time <= 0:
        raise ComputeError("mission_time must be positive")
    reliability = float(np.exp(-((mission_time / alpha) ** beta)))
    distribution = getattr(fit, "distribution", None)
    mean_life = float(getattr(distribution, "mean", alpha * math.gamma(1 + 1 / beta)))
    median_life = float(getattr(distribution, "median", alpha * np.log(2) ** (1 / beta)))
    return {
        "mode": "reliability_life_distribution",
        "weibull_scale_alpha": alpha,
        "weibull_shape_beta": beta,
        "mean_life": mean_life,
        "median_life": median_life,
        "mission_time": mission_time,
        "mission_reliability": reliability,
        "log_likelihood": float(getattr(fit, "loglik", getattr(fit, "loglik2", 0.0))),
        "aic": float(getattr(fit, "AICc", getattr(fit, "AIC", 0.0))),
        "bic": float(getattr(fit, "BIC", 0.0)),
        "engine": {"reliability": package("reliability")},
    }


def control_system_stability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("control")
    import control as ct

    a = matrix(inputs.get("a"), "inputs.a", min_rows=1, max_rows=50, min_columns=1, max_columns=50)
    if a.shape[0] != a.shape[1]:
        raise ComputeError("state matrix a must be square")
    b = matrix(inputs.get("b"), "inputs.b", min_rows=a.shape[0], max_rows=a.shape[0], min_columns=1, max_columns=20)
    c = matrix(inputs.get("c"), "inputs.c", min_rows=1, max_rows=20, min_columns=a.shape[0], max_columns=a.shape[0])
    d = matrix(inputs.get("d"), "inputs.d", min_rows=c.shape[0], max_rows=c.shape[0], min_columns=b.shape[1], max_columns=b.shape[1])
    system = ct.ss(a, b, c, d)
    poles = np.asarray(ct.poles(system), dtype=complex)
    controllability_rank = int(np.linalg.matrix_rank(ct.ctrb(a, b)))
    observability_rank = int(np.linalg.matrix_rank(ct.obsv(a, c)))
    points = integer(inputs.get("response_points", 100), "inputs.response_points", 10, 5000)
    duration = finite(inputs.get("response_duration", 10.0), "inputs.response_duration")
    if duration <= 0:
        raise ComputeError("response_duration must be positive")
    response = ct.step_response(system, T=np.linspace(0, duration, points))
    return {
        "mode": "control_system_stability",
        "poles": [{"real": float(v.real), "imaginary": float(v.imag)} for v in poles],
        "asymptotically_stable": bool(np.all(np.real(poles) < 0)),
        "controllability_rank": controllability_rank,
        "observability_rank": observability_rank,
        "state_dimension": int(a.shape[0]),
        "step_response_time": np.asarray(response.time, dtype=float).tolist(),
        "step_response_outputs": np.asarray(response.outputs, dtype=float).tolist(),
        "engine": {"control": package("control")},
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "model_fairness_audit": model_fairness_audit,
    "model_explainability": model_explainability,
    "label_quality_audit": label_quality_audit,
    "reliability_life_distribution": reliability_life_distribution,
    "control_system_stability": control_system_stability,
}
