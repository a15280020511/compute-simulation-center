
#!/usr/bin/env python3
"""Bounded finance, investment and business-decision operations.

No network, brokerage execution, live quotes or personalized suitability advice.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from compute_runner import ComputeError

MAX_ASSETS = 50
MAX_OBSERVATIONS = 10_000
MAX_CASH_FLOWS = 1_200


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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComputeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _returns_array(value: Any, name: str = "inputs.returns") -> np.ndarray:
    rows = [_finite(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name))]
    if len(rows) < 2 or len(rows) > MAX_OBSERVATIONS:
        raise ComputeError(f"{name} must contain 2 to {MAX_OBSERVATIONS} values")
    array = np.asarray(rows, dtype=float)
    if np.any(array <= -1):
        raise ComputeError(f"{name} values must be greater than -1")
    return array


def performance_metrics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = _returns_array(inputs.get("returns"))
    periods = _integer(inputs.get("periods_per_year", 252), "inputs.periods_per_year", 1, 100_000)
    risk_free = _finite(inputs.get("risk_free_rate", 0.0), "inputs.risk_free_rate")
    confidence = _finite(inputs.get("confidence", 0.95), "inputs.confidence")
    if not 0.5 < confidence < 1:
        raise ComputeError("inputs.confidence must be between 0.5 and 1")
    wealth = np.cumprod(1.0 + values)
    total_return = float(wealth[-1] - 1.0)
    annualized_return = float((wealth[-1] ** (periods / len(values))) - 1.0)
    annualized_volatility = float(np.std(values, ddof=1) * math.sqrt(periods))
    periodic_rf = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    excess = values - periodic_rf
    standard_deviation = float(np.std(excess, ddof=1))
    sharpe = None if standard_deviation == 0 else float(np.mean(excess) / standard_deviation * math.sqrt(periods))
    downside = excess[excess < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
    sortino = None if downside_deviation == 0 else float(np.mean(excess) / downside_deviation * math.sqrt(periods))
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    cutoff = float(np.quantile(values, 1.0 - confidence))
    tail = values[values <= cutoff]
    return {
        "mode": "performance_metrics",
        "observations": int(values.size),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": float(np.min(drawdowns)),
        "historical_var_loss": max(0.0, -cutoff),
        "historical_cvar_loss": max(0.0, -float(np.mean(tail))) if tail.size else 0.0,
        "win_rate": float(np.mean(values > 0)),
        "confidence": confidence,
        "decision_support_only": True,
    }


def _price_frame(value: Any):
    raw = _mapping(value, "inputs.prices")
    if not 2 <= len(raw) <= MAX_ASSETS:
        raise ComputeError(f"inputs.prices must contain 2 to {MAX_ASSETS} assets")
    names = [str(name) for name in raw]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ComputeError("asset names must be non-empty and unique")
    columns: dict[str, list[float]] = {}
    expected = None
    for name, series in raw.items():
        values = [_finite(item, f"inputs.prices[{name}]") for item in _sequence(series, f"inputs.prices[{name}]")]
        if len(values) < 3 or len(values) > MAX_OBSERVATIONS:
            raise ComputeError(f"price series {name} must contain 3 to {MAX_OBSERVATIONS} values")
        if any(item <= 0 for item in values):
            raise ComputeError(f"price series {name} must contain positive values")
        expected = len(values) if expected is None else expected
        if len(values) != expected:
            raise ComputeError("all price series must have equal length")
        columns[str(name)] = values
    import pandas as pd
    return pd.DataFrame(columns)


def portfolio_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from pypfopt import EfficientFrontier, expected_returns, risk_models
    except ImportError as exc:
        raise ComputeError("finance optional dependency is not installed") from exc
    prices = _price_frame(inputs.get("prices"))
    frequency = _integer(inputs.get("periods_per_year", 252), "inputs.periods_per_year", 1, 100_000)
    risk_free = _finite(inputs.get("risk_free_rate", 0.0), "inputs.risk_free_rate")
    raw_bounds = _sequence(inputs.get("weight_bounds", [0.0, 1.0]), "inputs.weight_bounds")
    if len(raw_bounds) != 2:
        raise ComputeError("inputs.weight_bounds must contain [minimum, maximum]")
    lower = _finite(raw_bounds[0], "inputs.weight_bounds[0]")
    upper = _finite(raw_bounds[1], "inputs.weight_bounds[1]")
    if lower > upper or lower * len(prices.columns) > 1 or upper * len(prices.columns) < 1:
        raise ComputeError("weight bounds cannot produce a fully invested portfolio")
    mu = expected_returns.mean_historical_return(prices, frequency=frequency)
    covariance = risk_models.sample_cov(prices, frequency=frequency)
    frontier = EfficientFrontier(mu, covariance, weight_bounds=(lower, upper))
    objective = str(inputs.get("objective") or "max_sharpe")
    if objective == "max_sharpe":
        frontier.max_sharpe(risk_free_rate=risk_free)
    elif objective == "min_volatility":
        frontier.min_volatility()
    elif objective == "efficient_return":
        frontier.efficient_return(_finite(inputs.get("target_return"), "inputs.target_return"))
    elif objective == "efficient_risk":
        frontier.efficient_risk(_finite(inputs.get("target_volatility"), "inputs.target_volatility"))
    else:
        raise ComputeError("inputs.objective must be max_sharpe, min_volatility, efficient_return, or efficient_risk")
    weights = {name: float(value) for name, value in frontier.clean_weights().items()}
    expected_return, volatility, sharpe = frontier.portfolio_performance(
        verbose=False, risk_free_rate=risk_free
    )
    return {
        "mode": "portfolio_optimization",
        "objective": objective,
        "asset_count": len(weights),
        "weights": weights,
        "expected_annual_return": float(expected_return),
        "annual_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "risk_free_rate": risk_free,
        "decision_support_only": True,
    }


def investment_projection(inputs: Mapping[str, Any]) -> dict[str, Any]:
    initial = _finite(inputs.get("initial_principal", 0.0), "inputs.initial_principal")
    contribution = _finite(inputs.get("monthly_contribution", 0.0), "inputs.monthly_contribution")
    annual_return = _finite(inputs.get("annual_return"), "inputs.annual_return")
    annual_fee = _finite(inputs.get("annual_fee_rate", 0.0), "inputs.annual_fee_rate")
    inflation = _finite(inputs.get("annual_inflation_rate", 0.0), "inputs.annual_inflation_rate")
    years = _finite(inputs.get("years"), "inputs.years")
    if initial < 0 or contribution < 0 or years <= 0 or years > 100:
        raise ComputeError("principal/contribution must be non-negative and years must be in (0, 100]")
    months = max(1, int(round(years * 12)))
    net_annual = (1.0 + annual_return) * (1.0 - annual_fee) - 1.0
    if net_annual <= -1:
        raise ComputeError("net annual return must be greater than -1")
    monthly_rate = (1.0 + net_annual) ** (1.0 / 12.0) - 1.0
    balance = initial
    for _ in range(months):
        balance = balance * (1.0 + monthly_rate) + contribution
    contributed = initial + contribution * months
    real_value = balance / ((1.0 + inflation) ** years) if inflation > -1 else None
    return {
        "mode": "investment_projection",
        "months": months,
        "nominal_ending_value": float(balance),
        "inflation_adjusted_ending_value": None if real_value is None else float(real_value),
        "total_contributions": float(contributed),
        "nominal_gain": float(balance - contributed),
        "net_annual_return_after_fee": float(net_annual),
        "decision_support_only": True,
    }


def business_unit_economics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    price = _finite(inputs.get("price_per_unit"), "inputs.price_per_unit")
    variable = _finite(inputs.get("variable_cost_per_unit"), "inputs.variable_cost_per_unit")
    fixed = _finite(inputs.get("fixed_costs"), "inputs.fixed_costs")
    units = _finite(inputs.get("expected_units"), "inputs.expected_units")
    if min(price, variable, fixed, units) < 0 or price <= variable:
        raise ComputeError("values must be non-negative and price_per_unit must exceed variable_cost_per_unit")
    margin = price - variable
    revenue = price * units
    operating_profit = margin * units - fixed
    result: dict[str, Any] = {
        "mode": "business_unit_economics",
        "revenue": revenue,
        "contribution_margin_per_unit": margin,
        "contribution_margin_ratio": margin / price if price else None,
        "break_even_units": fixed / margin,
        "operating_profit": operating_profit,
        "operating_margin": operating_profit / revenue if revenue else None,
        "return_on_fixed_cost": operating_profit / fixed if fixed else None,
        "decision_support_only": True,
    }
    if "customer_acquisition_cost" in inputs:
        cac = _finite(inputs.get("customer_acquisition_cost"), "inputs.customer_acquisition_cost")
        gross = _finite(inputs.get("gross_profit_per_customer"), "inputs.gross_profit_per_customer")
        retention = _finite(inputs.get("retention_months"), "inputs.retention_months")
        if cac < 0 or gross < 0 or retention < 0:
            raise ComputeError("CAC, customer gross profit and retention must be non-negative")
        ltv = gross * retention
        result.update({
            "customer_lifetime_value": ltv,
            "ltv_to_cac": None if cac == 0 else ltv / cac,
            "cac_payback_months": None if gross == 0 else cac / gross,
        })
    return result


def _npv(cash_flows: np.ndarray, rate: float) -> float:
    return float(sum(value / ((1.0 + rate) ** index) for index, value in enumerate(cash_flows)))


def _irr(cash_flows: np.ndarray) -> float | None:
    grid = np.concatenate((np.linspace(-0.999, 1.0, 800), np.geomspace(2.0, 101.0, 300) - 1.0))
    previous_rate = float(grid[0])
    previous_value = _npv(cash_flows, previous_rate)
    for rate_value in grid[1:]:
        rate = float(rate_value)
        current = _npv(cash_flows, rate)
        if current == 0:
            return rate
        if previous_value * current < 0:
            try:
                return float(brentq(lambda item: _npv(cash_flows, item), previous_rate, rate))
            except ValueError:
                return None
        previous_rate, previous_value = rate, current
    return None


def capital_budgeting(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = [_finite(item, f"inputs.cash_flows[{index}]") for index, item in enumerate(_sequence(inputs.get("cash_flows"), "inputs.cash_flows"))]
    if not 2 <= len(rows) <= MAX_CASH_FLOWS:
        raise ComputeError(f"inputs.cash_flows must contain 2 to {MAX_CASH_FLOWS} values")
    discount = _finite(inputs.get("discount_rate"), "inputs.discount_rate")
    if discount <= -1:
        raise ComputeError("inputs.discount_rate must be greater than -1")
    array = np.asarray(rows, dtype=float)
    discounted = np.asarray([value / ((1.0 + discount) ** index) for index, value in enumerate(array)])
    cumulative = np.cumsum(discounted)
    payback = next((index for index, value in enumerate(cumulative) if value >= 0), None)
    return {
        "mode": "capital_budgeting",
        "net_present_value": float(np.sum(discounted)),
        "internal_rate_of_return": _irr(array),
        "discounted_payback_periods": payback,
        "cash_flow_count": len(rows),
        "discount_rate": discount,
        "decision_support_only": True,
    }



def strategy_backtest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    '''Run a bounded fixed-strategy backtest through vectorbt.'''
    try:
        import pandas as pd
        import vectorbt as vbt
    except ImportError as exc:
        raise ComputeError("backtest optional dependency is not installed") from exc

    prices = np.asarray(
        [_finite(item, f"inputs.prices[{index}]")
         for index, item in enumerate(_sequence(inputs.get("prices"), "inputs.prices"))],
        dtype=float,
    )
    if not 20 <= prices.size <= MAX_OBSERVATIONS:
        raise ComputeError(f"inputs.prices must contain 20 to {MAX_OBSERVATIONS} values")
    if np.any(prices <= 0):
        raise ComputeError("inputs.prices must contain positive values")

    strategy = str(inputs.get("strategy") or "buy_and_hold")
    initial_cash = _finite(inputs.get("initial_cash", 100000.0), "inputs.initial_cash")
    fees = _finite(inputs.get("fees", 0.0), "inputs.fees")
    slippage = _finite(inputs.get("slippage", 0.0), "inputs.slippage")
    periods = _integer(inputs.get("periods_per_year", 252), "inputs.periods_per_year", 1, 100000)
    if initial_cash <= 0:
        raise ComputeError("inputs.initial_cash must be positive")
    if not 0 <= fees <= 0.1 or not 0 <= slippage <= 0.1:
        raise ComputeError("inputs.fees and inputs.slippage must be between 0 and 0.1")

    close = pd.Series(prices, dtype=float)
    if strategy == "buy_and_hold":
        entries = pd.Series(False, index=close.index)
        entries.iloc[0] = True
        exits = pd.Series(False, index=close.index)
        strategy_parameters: dict[str, Any] = {}
    elif strategy == "moving_average_crossover":
        fast_window = _integer(inputs.get("fast_window", 10), "inputs.fast_window", 2, 500)
        slow_window = _integer(inputs.get("slow_window", 30), "inputs.slow_window", 3, 1000)
        if fast_window >= slow_window or slow_window >= prices.size:
            raise ComputeError("moving-average windows must satisfy 2 <= fast < slow < price count")
        fast = close.rolling(fast_window, min_periods=fast_window).mean()
        slow = close.rolling(slow_window, min_periods=slow_window).mean()
        active = (fast > slow).fillna(False)
        previous = active.shift(1, fill_value=False)
        entries = active & ~previous
        exits = ~active & previous
        strategy_parameters = {"fast_window": fast_window, "slow_window": slow_window}
    else:
        raise ComputeError("inputs.strategy must be buy_and_hold or moving_average_crossover")

    portfolio = vbt.Portfolio.from_signals(
        close, entries, exits, init_cash=initial_cash, fees=fees, slippage=slippage
    )
    values = np.asarray(portfolio.value(), dtype=float)
    if values.size != prices.size or not np.isfinite(values).all() or np.any(values <= 0):
        raise ComputeError("backtest produced invalid portfolio values")
    periodic_returns = values[1:] / values[:-1] - 1.0
    wealth_ratio = float(values[-1] / values[0])
    annualized_return = float(wealth_ratio ** (periods / max(1, periodic_returns.size)) - 1.0)
    annualized_volatility = (
        float(np.std(periodic_returns, ddof=1) * math.sqrt(periods))
        if periodic_returns.size > 1 else 0.0
    )
    risk_free = _finite(inputs.get("risk_free_rate", 0.0), "inputs.risk_free_rate")
    periodic_rf = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    excess = periodic_returns - periodic_rf
    excess_std = float(np.std(excess, ddof=1)) if excess.size > 1 else 0.0
    sharpe = None if excess_std == 0 else float(np.mean(excess) / excess_std * math.sqrt(periods))
    downside = excess[excess < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
    sortino = None if downside_deviation == 0 else float(np.mean(excess) / downside_deviation * math.sqrt(periods))
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    confidence = _finite(inputs.get("confidence", 0.95), "inputs.confidence")
    if not 0.5 < confidence < 1:
        raise ComputeError("inputs.confidence must be between 0.5 and 1")
    cutoff = float(np.quantile(periodic_returns, 1.0 - confidence))
    tail = periodic_returns[periodic_returns <= cutoff]
    try:
        trade_count = int(portfolio.trades.count())
    except Exception:
        trade_count = 0
    try:
        order_count = int(portfolio.orders.count())
    except Exception:
        order_count = 0
    try:
        win_rate = float(portfolio.trades.win_rate()) if trade_count else None
    except Exception:
        win_rate = None
    return {
        "mode": "strategy_backtest",
        "strategy": strategy,
        "strategy_parameters": strategy_parameters,
        "observations": int(prices.size),
        "initial_cash": initial_cash,
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / initial_cash - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": float(np.min(drawdowns)),
        "historical_var_loss": max(0.0, -cutoff),
        "historical_cvar_loss": max(0.0, -float(np.mean(tail))) if tail.size else 0.0,
        "trade_count": trade_count,
        "order_count": order_count,
        "win_rate": win_rate,
        "fees": fees,
        "slippage": slippage,
        "decision_support_only": True,
        "arbitrary_strategy_code_allowed": False,
    }

def finance_decision_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
        "performance_metrics": performance_metrics,
        "portfolio_optimization": portfolio_optimization,
        "investment_projection": investment_projection,
        "business_unit_economics": business_unit_economics,
        "capital_budgeting": capital_budgeting,
        "strategy_backtest": strategy_backtest,
    }
    handler = handlers.get(mode)
    if handler is None:
        raise ComputeError(f"unsupported finance mode: {mode}")
    return handler(inputs)


OPERATIONS = {"finance_decision_analysis": finance_decision_analysis}
