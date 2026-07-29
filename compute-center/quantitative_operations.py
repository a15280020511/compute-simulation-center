#!/usr/bin/env python3
"""Bounded quantitative-investment research modes.

No live prices, brokerage connections, order execution, leverage recommendation,
or guaranteed-profit claims are supported.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from compute_runner import ComputeError

MAX_ASSETS = 50
MAX_OBSERVATIONS = 20_000
MAX_SCENARIOS = 200


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
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _series(value: Any, name: str, minimum: int = 3) -> np.ndarray:
    rows = np.asarray([_finite(item, f"{name}[{i}]") for i, item in enumerate(_sequence(value, name))], dtype=float)
    if not minimum <= rows.size <= MAX_OBSERVATIONS:
        raise ComputeError(f"{name} must contain {minimum} to {MAX_OBSERVATIONS} values")
    return rows


def _performance(returns: np.ndarray, periods: int, risk_free: float = 0.0) -> dict[str, Any]:
    if returns.size == 0 or np.any(returns <= -1):
        raise ComputeError("backtest produced invalid returns")
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(wealth)
    volatility = float(np.std(returns, ddof=1) * math.sqrt(periods)) if returns.size > 1 else 0.0
    periodic_rf = (1 + risk_free) ** (1 / periods) - 1
    excess = returns - periodic_rf
    std = float(np.std(excess, ddof=1)) if returns.size > 1 else 0.0
    sharpe = None if std == 0 else float(np.mean(excess) / std * math.sqrt(periods))
    return {
        "total_return": float(wealth[-1] - 1),
        "annualized_return": float(wealth[-1] ** (periods / returns.size) - 1),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(np.min(wealth / peaks - 1)),
        "positive_period_rate": float(np.mean(returns > 0)),
    }


def factor_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ComputeError("quantitative optional dependency statsmodels is not installed") from exc
    asset = _series(inputs.get("asset_returns"), "inputs.asset_returns", 10)
    raw_factors = _mapping(inputs.get("factors"), "inputs.factors")
    if not 1 <= len(raw_factors) <= 20:
        raise ComputeError("inputs.factors must contain 1 to 20 factors")
    names = [str(name) for name in raw_factors]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ComputeError("factor names must be non-empty and unique")
    columns = []
    for name in names:
        values = _series(raw_factors[name], f"inputs.factors[{name}]", 10)
        if values.size != asset.size:
            raise ComputeError("all factor series must match asset return length")
        columns.append(values)
    x = np.column_stack(columns)
    if bool(inputs.get("include_intercept", True)):
        x = sm.add_constant(x, has_constant="add")
        parameter_names = ["alpha", *names]
    else:
        parameter_names = names
    covariance = str(inputs.get("covariance_type") or "HAC")
    model = sm.OLS(asset, x)
    if covariance == "HAC":
        maxlags = _integer(inputs.get("hac_lags", 5), "inputs.hac_lags", 0, min(250, asset.size - 2))
        fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif covariance in {"HC0", "HC1", "HC2", "HC3"}:
        fitted = model.fit(cov_type=covariance)
    elif covariance == "nonrobust":
        fitted = model.fit()
    else:
        raise ComputeError("inputs.covariance_type must be HAC, HC0, HC1, HC2, HC3, or nonrobust")
    parameters = {
        name: {
            "coefficient": float(fitted.params[index]),
            "standard_error": float(fitted.bse[index]),
            "t_statistic": float(fitted.tvalues[index]),
            "p_value": float(fitted.pvalues[index]),
        }
        for index, name in enumerate(parameter_names)
    }
    return {
        "mode": "factor_regression",
        "observations": int(asset.size),
        "parameters": parameters,
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "covariance_type": covariance,
        "residual_volatility": float(np.std(fitted.resid, ddof=max(1, len(parameter_names)))),
        "decision_support_only": True,
    }


def walk_forward_backtest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    prices = _series(inputs.get("prices"), "inputs.prices", 60)
    if np.any(prices <= 0):
        raise ComputeError("inputs.prices must contain positive values")
    fast = _integer(inputs.get("fast_window", 20), "inputs.fast_window", 2, 500)
    slow = _integer(inputs.get("slow_window", 60), "inputs.slow_window", 3, 1000)
    validation_window = _integer(inputs.get("validation_window", 20), "inputs.validation_window", 5, 1000)
    if fast >= slow or slow + validation_window >= prices.size:
        raise ComputeError("require fast < slow and enough observations for validation")
    fee = _finite(inputs.get("fee_rate", 0.0), "inputs.fee_rate")
    slippage = _finite(inputs.get("slippage_rate", 0.0), "inputs.slippage_rate")
    if not 0 <= fee <= 0.05 or not 0 <= slippage <= 0.05:
        raise ComputeError("fee_rate and slippage_rate must be between 0 and 0.05")
    periods = _integer(inputs.get("periods_per_year", 252), "inputs.periods_per_year", 1, 100000)
    returns = prices[1:] / prices[:-1] - 1.0
    fast_ma = np.full(prices.size, np.nan)
    slow_ma = np.full(prices.size, np.nan)
    for index in range(fast - 1, prices.size):
        fast_ma[index] = float(np.mean(prices[index - fast + 1:index + 1]))
    for index in range(slow - 1, prices.size):
        slow_ma[index] = float(np.mean(prices[index - slow + 1:index + 1]))
    signal = np.zeros(prices.size, dtype=float)
    valid = np.isfinite(fast_ma) & np.isfinite(slow_ma)
    signal[valid] = (fast_ma[valid] > slow_ma[valid]).astype(float)
    positions = signal[:-1]
    turnover = np.abs(np.diff(np.concatenate(([0.0], positions))))
    strategy_returns = positions * returns - turnover * (fee + slippage)
    start = slow - 1
    folds = []
    validation_chunks = []
    fold_index = 0
    while start + validation_window <= strategy_returns.size:
        end = start + validation_window
        chunk = strategy_returns[start:end]
        benchmark = returns[start:end]
        validation_chunks.append(chunk)
        fold_index += 1
        fold_metrics = _performance(chunk, periods)
        fold_metrics.update({
            "fold": fold_index,
            "start_index": int(start + 1),
            "end_index": int(end),
            "benchmark_total_return": float(np.prod(1 + benchmark) - 1),
        })
        folds.append(fold_metrics)
        start = end
    if not validation_chunks:
        raise ComputeError("no complete validation fold was produced")
    out_of_sample = np.concatenate(validation_chunks)
    return {
        "mode": "walk_forward_backtest",
        "strategy": "fixed moving-average crossover",
        "parameters": {"fast_window": fast, "slow_window": slow, "validation_window": validation_window},
        "fold_count": len(folds),
        "folds": folds,
        "out_of_sample_metrics": _performance(out_of_sample, periods, _finite(inputs.get("risk_free_rate", 0.0), "inputs.risk_free_rate")),
        "turnover_events": int(np.sum(turnover > 0)),
        "fee_rate": fee,
        "slippage_rate": slippage,
        "lookahead_bias_control": "signals use information through t and apply to return t->t+1",
        "arbitrary_strategy_code_allowed": False,
        "decision_support_only": True,
    }


def _covariance_from_returns(raw: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    names = [str(name) for name in raw]
    if not 2 <= len(names) <= MAX_ASSETS or any(not name for name in names) or len(set(names)) != len(names):
        raise ComputeError(f"asset returns must contain 2 to {MAX_ASSETS} unique assets")
    arrays = [_series(raw[name], f"inputs.returns_by_asset[{name}]", 10) for name in names]
    if len({array.size for array in arrays}) != 1:
        raise ComputeError("all asset return series must have equal length")
    matrix = np.column_stack(arrays)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    if not np.isfinite(covariance).all():
        raise ComputeError("covariance matrix is non-finite")
    return names, covariance


def risk_parity_allocation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    names, covariance = _covariance_from_returns(_mapping(inputs.get("returns_by_asset"), "inputs.returns_by_asset"))
    count = len(names)
    lower = _finite(inputs.get("minimum_weight", 0.0), "inputs.minimum_weight")
    upper = _finite(inputs.get("maximum_weight", 1.0), "inputs.maximum_weight")
    if lower < 0 or upper > 1 or lower > upper or lower * count > 1 or upper * count < 1:
        raise ComputeError("weight bounds cannot produce a fully invested long-only portfolio")

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ covariance @ weights)
        if variance <= 0:
            return 1e12
        contribution = weights * (covariance @ weights) / variance
        return float(np.sum((contribution - np.full(count, 1.0 / count)) ** 2))

    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(lower, upper)] * count,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise ComputeError(f"risk parity optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    variance = float(weights @ covariance @ weights)
    contributions = weights * (covariance @ weights) / variance
    return {
        "mode": "risk_parity_allocation",
        "weights": {names[i]: float(weights[i]) for i in range(count)},
        "variance_contributions": {names[i]: float(contributions[i]) for i in range(count)},
        "portfolio_volatility_per_period": float(math.sqrt(max(variance, 0.0))),
        "objective_value": float(result.fun),
        "long_only": True,
        "decision_support_only": True,
    }


def portfolio_stress_test(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw_weights = _mapping(inputs.get("weights"), "inputs.weights")
    names = [str(name) for name in raw_weights]
    if not 1 <= len(names) <= MAX_ASSETS or any(not name for name in names):
        raise ComputeError(f"inputs.weights must contain 1 to {MAX_ASSETS} assets")
    weights = np.asarray([_finite(raw_weights[name], f"inputs.weights[{name}]") for name in names], dtype=float)
    if np.any(weights < 0) or not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ComputeError("weights must be non-negative and sum to 1")
    raw_scenarios = _sequence(inputs.get("scenarios"), "inputs.scenarios")
    if not 1 <= len(raw_scenarios) <= MAX_SCENARIOS:
        raise ComputeError(f"inputs.scenarios must contain 1 to {MAX_SCENARIOS} entries")
    rows = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_scenarios):
        scenario = _mapping(raw, f"inputs.scenarios[{index}]")
        name = str(scenario.get("name") or "")
        shocks = _mapping(scenario.get("asset_shocks"), f"scenario[{name}].asset_shocks")
        if not name or name in seen or set(shocks) != set(names):
            raise ComputeError("scenario names must be unique and every scenario must include every asset")
        shock_values = np.asarray([_finite(shocks[asset], f"scenario[{name}].{asset}") for asset in names], dtype=float)
        if np.any(shock_values <= -1):
            raise ComputeError("asset shocks must be greater than -1")
        portfolio_return = float(weights @ shock_values)
        contributions = weights * shock_values
        rows.append({
            "scenario": name,
            "portfolio_return": portfolio_return,
            "portfolio_loss": max(0.0, -portfolio_return),
            "asset_contributions": {names[i]: float(contributions[i]) for i in range(len(names))},
        })
        seen.add(name)
    rows.sort(key=lambda row: row["portfolio_return"])
    return {
        "mode": "portfolio_stress_test",
        "worst_scenario": rows[0]["scenario"],
        "worst_portfolio_return": rows[0]["portfolio_return"],
        "scenarios": rows,
        "herfindahl_concentration": float(np.sum(weights ** 2)),
        "largest_weight": float(np.max(weights)),
        "decision_support_only": True,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "factor_regression": factor_regression,
    "walk_forward_backtest": walk_forward_backtest,
    "risk_parity_allocation": risk_parity_allocation,
    "portfolio_stress_test": portfolio_stress_test,
}
