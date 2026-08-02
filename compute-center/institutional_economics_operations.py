#!/usr/bin/env python3
"""Allowlisted institutional economics and causal-inference operations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from institutional_common import engine, equal_length, integer, jsonable, matrix, safe_names, strings, vector


def high_dimensional_fixed_effects(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pyfixest")
    import pandas as pd
    import pyfixest as pf

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20, max_rows=50_000, max_columns=50)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    fixed_effect = strings(inputs.get("fixed_effect"), "inputs.fixed_effect", minimum=x.shape[0], maximum=x.shape[0])
    columns = [f"x{i}" for i in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    frame["y"] = y
    frame["fixed_effect"] = fixed_effect
    formula = "y ~ " + " + ".join(columns) + " | fixed_effect"
    fit = pf.feols(formula, data=frame, vcov="HC1")
    coefficients = fit.coef()
    standard_errors = fit.se()
    return {
        "mode": "high_dimensional_fixed_effects",
        "formula": formula,
        "observations": int(x.shape[0]),
        "fixed_effect_levels": len(set(fixed_effect)),
        "coefficients": jsonable(coefficients),
        "standard_errors": jsonable(standard_errors),
        "engine": engine("pyfixest"),
    }


def double_machine_learning(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("doubleml", "scikit-learn")
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor

    x = matrix(inputs.get("x"), "inputs.x", min_rows=80, max_rows=20_000, max_columns=30)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    treatment = vector(inputs.get("treatment"), "inputs.treatment", minimum=x.shape[0], maximum=x.shape[0])
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    folds = integer(inputs.get("folds", 2), "inputs.folds", 2, 5)
    data = DoubleMLData.from_arrays(x, y, treatment)
    ml_l = RandomForestRegressor(n_estimators=80, max_depth=5, min_samples_leaf=3, random_state=seed, n_jobs=1)
    ml_m = RandomForestRegressor(n_estimators=80, max_depth=5, min_samples_leaf=3, random_state=seed + 1, n_jobs=1)
    fit = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=folds, n_rep=1).fit()
    return {
        "mode": "double_machine_learning",
        "coefficient": float(np.asarray(fit.coef).reshape(-1)[0]),
        "standard_error": float(np.asarray(fit.se).reshape(-1)[0]),
        "p_value": float(np.asarray(fit.pval).reshape(-1)[0]),
        "observations": int(x.shape[0]),
        "folds": folds,
        "engine": engine("doubleml", "scikit-learn"),
    }


def heterogeneous_treatment_effects(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("econml", "scikit-learn")
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    x = matrix(inputs.get("x"), "inputs.x", min_rows=100, max_rows=20_000, max_columns=30)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    treatment = vector(inputs.get("treatment"), "inputs.treatment", minimum=x.shape[0], maximum=x.shape[0])
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    learner_y = RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=4, random_state=seed, n_jobs=1)
    learner_t = RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=4, random_state=seed + 1, n_jobs=1)
    estimator = LinearDML(
        model_y=learner_y,
        model_t=learner_t,
        discrete_treatment=False,
        cv=2,
        random_state=seed,
    )
    estimator.fit(y, treatment, X=x)
    effects = np.asarray(estimator.effect(x), dtype=float).reshape(-1)
    intervals = estimator.effect_interval(x, alpha=0.05)
    lower = np.asarray(intervals[0], dtype=float).reshape(-1)
    upper = np.asarray(intervals[1], dtype=float).reshape(-1)
    return {
        "mode": "heterogeneous_treatment_effects",
        "average_effect": float(np.mean(effects)),
        "effect_standard_deviation": float(np.std(effects, ddof=1)),
        "effect_quantiles": {
            "q05": float(np.quantile(effects, 0.05)),
            "q50": float(np.quantile(effects, 0.50)),
            "q95": float(np.quantile(effects, 0.95)),
        },
        "average_interval": [float(np.mean(lower)), float(np.mean(upper))],
        "observations": int(x.shape[0]),
        "engine": engine("econml", "scikit-learn"),
    }


def structural_equation_model(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("semopy")
    import pandas as pd
    from semopy import Model

    names = safe_names(inputs.get("variable_names"), "inputs.variable_names", maximum=30)
    data = matrix(
        inputs.get("data"),
        "inputs.data",
        min_rows=max(30, len(names) * 5),
        max_rows=20_000,
        min_columns=len(names),
        max_columns=len(names),
    )
    if data.shape[1] != len(names):
        raise ComputeError("data columns must match variable_names")
    raw_regressions = inputs.get("regressions")
    if not isinstance(raw_regressions, list) or not raw_regressions or len(raw_regressions) > 50:
        raise ComputeError("inputs.regressions must contain 1 to 50 structured equations")
    equations: list[str] = []
    for index, raw in enumerate(raw_regressions):
        if not isinstance(raw, Mapping):
            raise ComputeError(f"inputs.regressions[{index}] must be an object")
        dependent = str(raw.get("dependent") or "")
        predictors = [str(item) for item in raw.get("predictors") or []]
        if dependent not in names or not predictors or any(item not in names or item == dependent for item in predictors):
            raise ComputeError("regression variables must reference distinct declared variables")
        equations.append(f"{dependent} ~ {' + '.join(predictors)}")
    description = "\n".join(equations)
    frame = pd.DataFrame(data, columns=names)
    model = Model(description)
    result = model.fit(frame)
    estimates = model.inspect(std_est=True)
    selected = estimates[estimates["op"] == "~"]
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "dependent": str(row["lval"]),
                "predictor": str(row["rval"]),
                "estimate": float(row["Estimate"]),
                "standard_error": float(row["Std. Err"]),
                "p_value": float(row["p-value"]),
                "standardized_estimate": float(row.get("Est. Std", np.nan))
                if np.isfinite(float(row.get("Est. Std", np.nan)))
                else None,
            }
        )
    return {
        "mode": "structural_equation_model",
        "description": description,
        "converged": bool(getattr(result, "success", True)),
        "objective_value": float(getattr(result, "fun", np.nan)) if np.isfinite(float(getattr(result, "fun", np.nan))) else None,
        "paths": rows,
        "engine": engine("semopy"),
    }


def blp_demand_instruments(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pyblp")
    import pandas as pd
    import pyblp

    characteristics = matrix(
        inputs.get("characteristics"),
        "inputs.characteristics",
        min_rows=4,
        max_rows=20_000,
        max_columns=20,
    )
    market_ids = strings(inputs.get("market_ids"), "inputs.market_ids", minimum=characteristics.shape[0], maximum=characteristics.shape[0])
    firm_ids = strings(inputs.get("firm_ids"), "inputs.firm_ids", minimum=characteristics.shape[0], maximum=characteristics.shape[0])
    equal_length("product data", market_ids, firm_ids, characteristics)
    columns = [f"x{i}" for i in range(characteristics.shape[1])]
    frame = pd.DataFrame(characteristics, columns=columns)
    frame["market_ids"] = market_ids
    frame["firm_ids"] = firm_ids
    formulation = pyblp.Formulation("0 + " + " + ".join(columns))
    instruments = np.asarray(pyblp.build_blp_instruments(formulation, frame), dtype=float)
    return {
        "mode": "blp_demand_instruments",
        "products": int(characteristics.shape[0]),
        "markets": len(set(market_ids)),
        "firms": len(set(firm_ids)),
        "instrument_shape": list(instruments.shape),
        "instruments": jsonable(instruments),
        "engine": engine("pyblp"),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "high_dimensional_fixed_effects": high_dimensional_fixed_effects,
    "double_machine_learning": double_machine_learning,
    "heterogeneous_treatment_effects": heterogeneous_treatment_effects,
    "structural_equation_model": structural_equation_model,
    "blp_demand_instruments": blp_demand_instruments,
}
