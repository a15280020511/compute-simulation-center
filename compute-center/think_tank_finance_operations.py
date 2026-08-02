#!/usr/bin/env python3
"""Advanced portfolio risk, financial diagnostics and attribution modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import MAX_ASSETS, MAX_SCENARIOS, finite, matrix, package, probability, vector


def cvar_portfolio(inputs: Mapping[str, Any]) -> dict[str, Any]:
    riskfolio_version = package("riskfolio-lib")
    from scipy.optimize import linprog

    returns = matrix(inputs.get("returns"), "inputs.returns", min_rows=30, max_columns=MAX_ASSETS)
    alpha = probability(inputs.get("alpha", 0.95), "inputs.alpha")
    target = finite(inputs.get("minimum_expected_return", -1e9), "inputs.minimum_expected_return")
    scenarios, assets = returns.shape
    if scenarios > MAX_SCENARIOS or alpha in {0.0, 1.0}:
        raise ComputeError("returns exceed scenario limit or alpha is not strictly between 0 and 1")
    count = assets + 1 + scenarios
    objective = np.zeros(count)
    objective[assets] = 1.0
    objective[assets + 1 :] = 1.0 / ((1 - alpha) * scenarios)
    a_ub = []
    b_ub = []
    for scenario in range(scenarios):
        row = np.zeros(count)
        row[:assets] = -returns[scenario]
        row[assets] = -1.0
        row[assets + 1 + scenario] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    expected = np.mean(returns, axis=0)
    a_ub.append(np.r_[-expected, np.zeros(1 + scenarios)])
    b_ub.append(-target)
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray([np.r_[np.ones(assets), np.zeros(1 + scenarios)]]),
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * assets + [(None, None)] + [(0.0, None)] * scenarios,
        method="highs",
    )
    if not result.success:
        raise ComputeError(f"CVaR optimization failed: {result.message}")
    weights = result.x[:assets]
    portfolio_returns = returns @ weights
    losses = -portfolio_returns
    value_at_risk = float(np.quantile(losses, alpha))
    tail = losses[losses >= value_at_risk]
    return {
        "mode": "cvar_portfolio",
        "weights": weights.tolist(),
        "expected_return": float(expected @ weights),
        "value_at_risk": value_at_risk,
        "conditional_value_at_risk": float(np.mean(tail)),
        "engine": {"riskfolio-lib": riskfolio_version, "solver": "scipy-highs-audited-cvar"},
        "brokerage_execution": False,
    }


def _maximum_drawdown(series: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + series)
    running_max = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / np.maximum(running_max, 1e-12)))


def drawdown_constrained_portfolio(inputs: Mapping[str, Any]) -> dict[str, Any]:
    riskfolio_version = package("riskfolio-lib")
    from scipy.optimize import minimize

    returns = matrix(inputs.get("returns"), "inputs.returns", min_rows=50, max_columns=MAX_ASSETS)
    maximum_drawdown = probability(inputs.get("maximum_drawdown", 0.2), "inputs.maximum_drawdown")
    target = finite(inputs.get("minimum_expected_return", -1e9), "inputs.minimum_expected_return")
    expected = np.mean(returns, axis=0)
    covariance = np.cov(returns, rowvar=False)
    assets = returns.shape[1]

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        {"type": "ineq", "fun": lambda weights: float(expected @ weights - target)},
        {
            "type": "ineq",
            "fun": lambda weights: float(maximum_drawdown - _maximum_drawdown(returns @ weights)),
        },
    ]
    result = minimize(
        objective,
        np.full(assets, 1.0 / assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * assets,
        constraints=constraints,
        options={"maxiter": 1_000, "ftol": 1e-10},
    )
    if not result.success:
        raise ComputeError(f"drawdown-constrained optimization failed: {result.message}")
    portfolio = returns @ result.x
    return {
        "mode": "drawdown_constrained_portfolio",
        "weights": result.x.tolist(),
        "expected_return": float(expected @ result.x),
        "volatility": float(np.std(portfolio, ddof=1)),
        "maximum_drawdown": _maximum_drawdown(portfolio),
        "constraint_limit": maximum_drawdown,
        "engine": {"riskfolio-lib": riskfolio_version, "solver": "scipy-slsqp-fixed-form"},
        "brokerage_execution": False,
    }


def financial_ratio_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    toolkit_version = package("financetoolkit")
    revenue = finite(inputs.get("revenue"), "inputs.revenue")
    cogs = finite(inputs.get("cost_of_goods_sold"), "inputs.cost_of_goods_sold")
    operating_income = finite(inputs.get("operating_income"), "inputs.operating_income")
    net_income = finite(inputs.get("net_income"), "inputs.net_income")
    assets = finite(inputs.get("total_assets"), "inputs.total_assets")
    equity = finite(inputs.get("total_equity"), "inputs.total_equity")
    current_assets = finite(inputs.get("current_assets"), "inputs.current_assets")
    current_liabilities = finite(inputs.get("current_liabilities"), "inputs.current_liabilities")
    debt = finite(inputs.get("total_debt"), "inputs.total_debt")
    if revenue == 0 or assets == 0 or equity == 0 or current_liabilities == 0:
        raise ComputeError("revenue, assets, equity and current liabilities must be non-zero")
    net_margin = net_income / revenue
    asset_turnover = revenue / assets
    equity_multiplier = assets / equity
    return {
        "mode": "financial_ratio_analysis",
        "gross_margin": (revenue - cogs) / revenue,
        "operating_margin": operating_income / revenue,
        "net_margin": net_margin,
        "return_on_assets": net_income / assets,
        "return_on_equity": net_income / equity,
        "current_ratio": current_assets / current_liabilities,
        "debt_to_equity": debt / equity,
        "dupont": {
            "net_margin": net_margin,
            "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier,
            "roe_product": net_margin * asset_turnover * equity_multiplier,
        },
        "engine": {"financetoolkit": toolkit_version, "formula_policy": "audited-fixed-ratios"},
    }


def factor_attribution(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("statsmodels")
    import statsmodels.api as sm

    returns = vector(inputs.get("portfolio_returns"), "inputs.portfolio_returns", minimum=20)
    factors = matrix(inputs.get("factor_returns"), "inputs.factor_returns", min_rows=20)
    if returns.size != factors.shape[0]:
        raise ComputeError("portfolio and factor return rows must match")
    result = sm.OLS(returns, sm.add_constant(factors, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(5, returns.size // 5)}
    )
    fitted = np.asarray(result.fittedvalues)
    residual = returns - fitted
    return {
        "mode": "factor_attribution",
        "alpha": float(result.params[0]),
        "factor_loadings": [float(v) for v in result.params[1:]],
        "p_values": [float(v) for v in result.pvalues],
        "r_squared": float(result.rsquared),
        "systematic_variance_share": float(np.var(fitted) / max(np.var(returns), 1e-12)),
        "residual_volatility": float(np.std(residual, ddof=1)),
        "brokerage_execution": False,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "cvar_portfolio": cvar_portfolio,
    "drawdown_constrained_portfolio": drawdown_constrained_portfolio,
    "financial_ratio_analysis": financial_ratio_analysis,
    "factor_attribution": factor_attribution,
}
