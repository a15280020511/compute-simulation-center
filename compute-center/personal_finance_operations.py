#!/usr/bin/env python3
"""Bounded offline personal-finance planning modes.

These handlers provide deterministic planning calculations only. They do not
fetch account data, place trades, underwrite insurance, provide tax/legal
advice, or guarantee outcomes.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from compute_runner import ComputeError

MAX_GOALS = 20
MAX_DEBTS = 20
MAX_SCENARIOS = 20
MAX_SIMULATIONS = 10_000
MAX_YEARS = 100


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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


def _monthly_rate(annual_rate: float) -> float:
    if annual_rate <= -1.0:
        raise ComputeError("annual rate must be greater than -1")
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def _base_result(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "network_used": False,
        "brokerage_execution": False,
        "account_access": False,
        "decision_support_only": True,
        "no_guaranteed_outcome": True,
    }


def goal_based_financial_plan(inputs: Mapping[str, Any]) -> dict[str, Any]:
    goals = _sequence(inputs.get("goals"), "inputs.goals")
    if not 1 <= len(goals) <= MAX_GOALS:
        raise ComputeError(f"goals must contain 1 to {MAX_GOALS} entries")
    rows = []
    total_monthly = 0.0
    for index, raw in enumerate(goals):
        goal = _mapping(raw, f"inputs.goals[{index}]")
        name = str(goal.get("name") or f"goal-{index + 1}").strip()
        if not name or len(name) > 100:
            raise ComputeError("goal names must be 1 to 100 characters")
        target = _finite(goal.get("target_amount"), f"goals[{index}].target_amount", 1.0)
        current = _finite(goal.get("current_balance", 0.0), f"goals[{index}].current_balance", 0.0)
        years = _finite(goal.get("years"), f"goals[{index}].years", 1.0 / 12.0, 80.0)
        annual_return = _finite(goal.get("annual_return", 0.0), f"goals[{index}].annual_return", -0.95, 1.0)
        months = max(1, int(round(years * 12.0)))
        rate = _monthly_rate(annual_return)
        future_current = current * (1.0 + rate) ** months
        gap = max(0.0, target - future_current)
        if gap == 0.0:
            monthly = 0.0
        elif abs(rate) < 1e-12:
            monthly = gap / months
        else:
            factor = ((1.0 + rate) ** months - 1.0) / rate
            monthly = gap / factor
        total_monthly += monthly
        rows.append({
            "name": name,
            "target_amount": target,
            "months": months,
            "future_value_of_current_balance": future_current,
            "unfunded_future_value": gap,
            "required_monthly_contribution": monthly,
            "annual_return_assumption": annual_return,
        })
    result = _base_result("goal_based_financial_plan")
    result.update({"goal_count": len(rows), "total_required_monthly_contribution": total_monthly, "goals": rows})
    return result


def retirement_monte_carlo(inputs: Mapping[str, Any]) -> dict[str, Any]:
    current_portfolio = _finite(inputs.get("current_portfolio"), "inputs.current_portfolio", 0.0)
    annual_contribution = _finite(inputs.get("annual_contribution", 0.0), "inputs.annual_contribution", 0.0)
    annual_withdrawal = _finite(inputs.get("annual_withdrawal"), "inputs.annual_withdrawal", 0.0)
    years_to_retirement = _integer(inputs.get("years_to_retirement"), "inputs.years_to_retirement", 0, 80)
    retirement_years = _integer(inputs.get("retirement_years"), "inputs.retirement_years", 1, MAX_YEARS)
    expected_return = _finite(inputs.get("expected_return", 0.04), "inputs.expected_return", -0.5, 0.5)
    volatility = _finite(inputs.get("volatility", 0.12), "inputs.volatility", 0.0, 1.0)
    inflation = _finite(inputs.get("inflation", 0.02), "inputs.inflation", -0.2, 0.3)
    simulations = _integer(inputs.get("simulations", 2_000), "inputs.simulations", 100, MAX_SIMULATIONS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    terminal = np.empty(simulations, dtype=float)
    ruin_years: list[int] = []
    for simulation in range(simulations):
        balance = current_portfolio
        for _ in range(years_to_retirement):
            annual_result = max(-0.95, float(rng.normal(expected_return, volatility)))
            balance = max(0.0, balance * (1.0 + annual_result) + annual_contribution)
        ruined = False
        withdrawal = annual_withdrawal
        for year in range(retirement_years):
            annual_result = max(-0.95, float(rng.normal(expected_return, volatility)))
            balance = balance * (1.0 + annual_result) - withdrawal
            if balance <= 0.0:
                balance = 0.0
                ruin_years.append(year + 1)
                ruined = True
                break
            withdrawal *= 1.0 + inflation
        if not ruined:
            terminal[simulation] = balance
        else:
            terminal[simulation] = 0.0
    success_probability = float(np.mean(terminal > 0.0))
    result = _base_result("retirement_monte_carlo")
    result.update({
        "simulations": simulations,
        "seed": seed,
        "success_probability": success_probability,
        "ruin_probability": 1.0 - success_probability,
        "terminal_wealth_percentiles": {
            "p10": float(np.percentile(terminal, 10)),
            "p50": float(np.percentile(terminal, 50)),
            "p90": float(np.percentile(terminal, 90)),
        },
        "median_ruin_year": float(np.median(ruin_years)) if ruin_years else None,
        "return_model": "independent annual normal draws clipped at -95 percent",
        "model_risk_warning": "results depend on return, volatility, inflation and independence assumptions",
    })
    return result


def _withdrawal_path(initial_balance: float, annual_withdrawal: float, returns: Sequence[float]) -> dict[str, Any]:
    balance = initial_balance
    trajectory = []
    depletion_year = None
    for year, annual_return in enumerate(returns, 1):
        balance = balance * (1.0 + annual_return) - annual_withdrawal
        if balance <= 0.0:
            balance = 0.0
            depletion_year = year
        trajectory.append(balance)
        if depletion_year is not None:
            trajectory.extend([0.0] * (len(returns) - year))
            break
    return {"ending_balance": balance, "depletion_year": depletion_year, "trajectory": trajectory}


def sequence_of_returns_risk(inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray(_sequence(inputs.get("returns"), "inputs.returns"), dtype=float)
    if values.ndim != 1 or not 2 <= values.size <= 200 or not np.all(np.isfinite(values)) or np.any(values <= -1.0):
        raise ComputeError("returns must contain 2 to 200 finite values greater than -1")
    initial = _finite(inputs.get("initial_balance"), "inputs.initial_balance", 1.0)
    withdrawal = _finite(inputs.get("annual_withdrawal"), "inputs.annual_withdrawal", 0.0)
    cases = {
        "observed_order": _withdrawal_path(initial, withdrawal, values.tolist()),
        "bad_returns_first": _withdrawal_path(initial, withdrawal, sorted(values.tolist())),
        "good_returns_first": _withdrawal_path(initial, withdrawal, sorted(values.tolist(), reverse=True)),
    }
    endings = [row["ending_balance"] for row in cases.values()]
    result = _base_result("sequence_of_returns_risk")
    result.update({
        "years": int(values.size),
        "cases": cases,
        "ending_balance_range": float(max(endings) - min(endings)),
        "same_return_multiset": True,
        "interpretation": "dispersion is caused only by return ordering under withdrawals",
    })
    return result


def liability_aware_allocation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    expected = np.asarray(_sequence(inputs.get("expected_returns"), "inputs.expected_returns"), dtype=float)
    covariance = np.asarray(_sequence(inputs.get("covariance"), "inputs.covariance"), dtype=float)
    durations = np.asarray(_sequence(inputs.get("asset_durations"), "inputs.asset_durations"), dtype=float)
    if expected.ndim != 1 or not 2 <= expected.size <= 30 or not np.all(np.isfinite(expected)):
        raise ComputeError("expected_returns must contain 2 to 30 finite values")
    if covariance.shape != (expected.size, expected.size) or not np.all(np.isfinite(covariance)):
        raise ComputeError("covariance dimensions must match expected_returns")
    if durations.shape != expected.shape or np.any(durations < 0.0) or not np.all(np.isfinite(durations)):
        raise ComputeError("asset_durations must contain one non-negative value per asset")
    covariance = (covariance + covariance.T) / 2.0
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-8:
        raise ComputeError("covariance must be positive semidefinite")
    liabilities = _sequence(inputs.get("liabilities"), "inputs.liabilities")
    if not 1 <= len(liabilities) <= 50:
        raise ComputeError("liabilities must contain 1 to 50 entries")
    discount_rate = _finite(inputs.get("discount_rate", 0.03), "inputs.discount_rate", -0.5, 0.5)
    current_assets = _finite(inputs.get("current_assets"), "inputs.current_assets", 1.0)
    pv_rows = []
    for index, raw in enumerate(liabilities):
        row = _mapping(raw, f"inputs.liabilities[{index}]")
        amount = _finite(row.get("amount"), f"liabilities[{index}].amount", 0.0)
        years = _finite(row.get("years"), f"liabilities[{index}].years", 0.0, 100.0)
        present_value = amount / (1.0 + discount_rate) ** years
        pv_rows.append((present_value, years))
    liability_pv = float(sum(item[0] for item in pv_rows))
    liability_duration = float(sum(pv * years for pv, years in pv_rows) / liability_pv) if liability_pv > 0 else 0.0
    max_weight = _finite(inputs.get("max_weight", 1.0), "inputs.max_weight", 1.0 / expected.size, 1.0)
    target_return = _finite(inputs.get("target_return", 0.0), "inputs.target_return", -0.5, 0.5)
    duration_penalty = _finite(inputs.get("duration_penalty", 0.1), "inputs.duration_penalty", 0.0, 1_000.0)

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ covariance @ weights)
        duration_gap = float(weights @ durations - liability_duration)
        return variance + duration_penalty * duration_gap**2

    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "ineq", "fun": lambda w: float(w @ expected - target_return)},
    ]
    solution = minimize(
        objective,
        np.full(expected.size, 1.0 / expected.size),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * expected.size,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise ComputeError(f"liability-aware allocation is infeasible: {solution.message}")
    weights = np.maximum(solution.x, 0.0)
    weights /= weights.sum()
    portfolio_return = float(weights @ expected)
    portfolio_duration = float(weights @ durations)
    result = _base_result("liability_aware_allocation")
    result.update({
        "weights": weights.tolist(),
        "expected_portfolio_return": portfolio_return,
        "portfolio_duration": portfolio_duration,
        "liability_present_value": liability_pv,
        "liability_duration": liability_duration,
        "duration_gap": portfolio_duration - liability_duration,
        "funding_ratio": current_assets / liability_pv if liability_pv > 0 else None,
        "target_return": target_return,
        "solver": "SciPy SLSQP",
    })
    return result


def emergency_fund_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    essential = _finite(inputs.get("monthly_essential_expenses"), "inputs.monthly_essential_expenses", 0.0)
    reserve = _finite(inputs.get("current_liquid_reserve", 0.0), "inputs.current_liquid_reserve", 0.0)
    target_months = _integer(inputs.get("target_months", 6), "inputs.target_months", 1, 36)
    monthly_saving = _finite(inputs.get("monthly_saving_capacity", 0.0), "inputs.monthly_saving_capacity", 0.0)
    target = essential * target_months
    gap = max(0.0, target - reserve)
    months_to_target = math.ceil(gap / monthly_saving) if gap > 0.0 and monthly_saving > 0.0 else (0 if gap == 0.0 else None)
    result = _base_result("emergency_fund_analysis")
    result.update({
        "target_reserve": target,
        "current_coverage_months": reserve / essential if essential > 0.0 else None,
        "target_months": target_months,
        "funding_gap": gap,
        "months_to_target": months_to_target,
        "target_is_user_supplied": True,
    })
    return result


def _simulate_debt_strategy(debts: list[dict[str, float]], extra: float, strategy: str, max_months: int) -> dict[str, Any]:
    balances = np.asarray([row["balance"] for row in debts], dtype=float)
    rates = np.asarray([row["annual_rate"] / 12.0 for row in debts], dtype=float)
    minimums = np.asarray([row["minimum_payment"] for row in debts], dtype=float)
    total_interest = 0.0
    for month in range(1, max_months + 1):
        if float(np.sum(balances)) <= 1e-8:
            return {"status": "PAID", "months": month - 1, "total_interest": total_interest}
        interest = balances * rates
        balances += interest
        total_interest += float(np.sum(interest))
        active = balances > 1e-8
        payments = np.minimum(balances, minimums * active)
        balances -= payments
        remaining = extra
        while remaining > 1e-10 and np.any(balances > 1e-8):
            active_indices = np.flatnonzero(balances > 1e-8)
            if strategy == "avalanche":
                target = int(active_indices[np.argmax(rates[active_indices])])
            else:
                target = int(active_indices[np.argmin(balances[active_indices])])
            payment = min(remaining, float(balances[target]))
            balances[target] -= payment
            remaining -= payment
    return {"status": "NOT_REPAID", "months": max_months, "total_interest": total_interest, "remaining_balance": float(np.sum(balances))}


def debt_repayment_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw_debts = _sequence(inputs.get("debts"), "inputs.debts")
    if not 1 <= len(raw_debts) <= MAX_DEBTS:
        raise ComputeError(f"debts must contain 1 to {MAX_DEBTS} entries")
    debts = []
    for index, raw in enumerate(raw_debts):
        debt = _mapping(raw, f"inputs.debts[{index}]")
        debts.append({
            "balance": _finite(debt.get("balance"), f"debts[{index}].balance", 0.01),
            "annual_rate": _finite(debt.get("annual_rate"), f"debts[{index}].annual_rate", 0.0, 2.0),
            "minimum_payment": _finite(debt.get("minimum_payment"), f"debts[{index}].minimum_payment", 0.0),
        })
    extra = _finite(inputs.get("extra_monthly_payment", 0.0), "inputs.extra_monthly_payment", 0.0)
    max_months = _integer(inputs.get("max_months", 600), "inputs.max_months", 1, 1_200)
    avalanche = _simulate_debt_strategy(debts, extra, "avalanche", max_months)
    snowball = _simulate_debt_strategy(debts, extra, "snowball", max_months)
    preferred = "avalanche" if avalanche["total_interest"] <= snowball["total_interest"] else "snowball"
    result = _base_result("debt_repayment_optimization")
    result.update({
        "strategies": {"avalanche": avalanche, "snowball": snowball},
        "lowest_interest_strategy": preferred,
        "extra_monthly_payment": extra,
        "assumption": "minimum payments continue and all extra cash is immediately redirected",
    })
    return result


def _simulate_withdrawal_strategy(
    returns: np.ndarray,
    initial: float,
    initial_spending: float,
    inflation: float,
    strategy: str,
    percentage: float,
    lower_guardrail: float,
    upper_guardrail: float,
) -> dict[str, Any]:
    balance = initial
    spending = initial_spending
    withdrawals = []
    depletion_year = None
    for year, annual_return in enumerate(returns, 1):
        if strategy == "percentage":
            spending = balance * percentage
        elif strategy == "guardrails" and year > 1:
            current_rate = spending / max(balance, 1e-12)
            if current_rate > upper_guardrail:
                spending *= 0.9
            elif current_rate < lower_guardrail:
                spending *= 1.1
            spending *= 1.0 + inflation
        elif year > 1:
            spending *= 1.0 + inflation
        withdrawals.append(spending)
        balance = balance * (1.0 + float(annual_return)) - spending
        if balance <= 0.0:
            balance = 0.0
            depletion_year = year
            break
    return {
        "ending_balance": balance,
        "depletion_year": depletion_year,
        "total_withdrawals": float(sum(withdrawals)),
        "years_funded": len(withdrawals),
    }


def withdrawal_strategy(inputs: Mapping[str, Any]) -> dict[str, Any]:
    returns = np.asarray(_sequence(inputs.get("returns"), "inputs.returns"), dtype=float)
    if returns.ndim != 1 or not 1 <= returns.size <= 200 or not np.all(np.isfinite(returns)) or np.any(returns <= -1.0):
        raise ComputeError("returns must contain 1 to 200 finite values greater than -1")
    initial = _finite(inputs.get("initial_portfolio"), "inputs.initial_portfolio", 1.0)
    spending = _finite(inputs.get("annual_spending"), "inputs.annual_spending", 0.0)
    inflation = _finite(inputs.get("inflation", 0.02), "inputs.inflation", -0.2, 0.3)
    percentage = _finite(inputs.get("percentage_rate", spending / initial), "inputs.percentage_rate", 0.0, 1.0)
    lower = _finite(inputs.get("lower_guardrail", 0.03), "inputs.lower_guardrail", 0.0, 1.0)
    upper = _finite(inputs.get("upper_guardrail", 0.06), "inputs.upper_guardrail", lower, 1.0)
    cases = {
        name: _simulate_withdrawal_strategy(returns, initial, spending, inflation, name, percentage, lower, upper)
        for name in ("fixed_real", "percentage", "guardrails")
    }
    result = _base_result("withdrawal_strategy")
    result.update({"years": int(returns.size), "strategies": cases, "comparison_only": True})
    return result


def _annuity_present_value(payment: float, years: int, discount_rate: float) -> float:
    if years <= 0 or payment <= 0.0:
        return 0.0
    if abs(discount_rate) < 1e-12:
        return payment * years
    return payment * (1.0 - (1.0 + discount_rate) ** (-years)) / discount_rate


def insurance_protection_gap(inputs: Mapping[str, Any]) -> dict[str, Any]:
    liabilities = _finite(inputs.get("outstanding_liabilities", 0.0), "inputs.outstanding_liabilities", 0.0)
    final_expenses = _finite(inputs.get("final_expenses", 0.0), "inputs.final_expenses", 0.0)
    education = _finite(inputs.get("education_funding_need", 0.0), "inputs.education_funding_need", 0.0)
    annual_support = _finite(inputs.get("annual_dependent_support", 0.0), "inputs.annual_dependent_support", 0.0)
    support_years = _integer(inputs.get("support_years", 0), "inputs.support_years", 0, 80)
    discount_rate = _finite(inputs.get("discount_rate", 0.03), "inputs.discount_rate", -0.5, 0.5)
    liquid_assets = _finite(inputs.get("available_liquid_assets", 0.0), "inputs.available_liquid_assets", 0.0)
    existing_coverage = _finite(inputs.get("existing_coverage", 0.0), "inputs.existing_coverage", 0.0)
    support_pv = _annuity_present_value(annual_support, support_years, discount_rate)
    gross_need = liabilities + final_expenses + education + support_pv
    resources = liquid_assets + existing_coverage
    result = _base_result("insurance_protection_gap")
    result.update({
        "gross_protection_need": gross_need,
        "dependent_support_present_value": support_pv,
        "available_resources": resources,
        "protection_gap": max(0.0, gross_need - resources),
        "surplus": max(0.0, resources - gross_need),
        "underwriting_performed": False,
        "product_recommendation_provided": False,
    })
    return result


def household_cashflow_stress(inputs: Mapping[str, Any]) -> dict[str, Any]:
    monthly_income = _finite(inputs.get("monthly_income"), "inputs.monthly_income", 0.0)
    monthly_expenses = _finite(inputs.get("monthly_expenses"), "inputs.monthly_expenses", 0.0)
    reserve = _finite(inputs.get("liquid_reserve", 0.0), "inputs.liquid_reserve", 0.0)
    raw_scenarios = _sequence(inputs.get("scenarios"), "inputs.scenarios")
    if not 1 <= len(raw_scenarios) <= MAX_SCENARIOS:
        raise ComputeError(f"scenarios must contain 1 to {MAX_SCENARIOS} entries")
    rows = []
    for index, raw in enumerate(raw_scenarios):
        scenario = _mapping(raw, f"inputs.scenarios[{index}]")
        name = str(scenario.get("name") or f"scenario-{index + 1}").strip()
        duration = _integer(scenario.get("duration_months"), f"scenarios[{index}].duration_months", 1, 120)
        income_change = _finite(scenario.get("income_change_pct", 0.0), f"scenarios[{index}].income_change_pct", -1.0, 10.0)
        expense_change = _finite(scenario.get("expense_change_pct", 0.0), f"scenarios[{index}].expense_change_pct", -1.0, 10.0)
        one_time_cost = _finite(scenario.get("one_time_cost", 0.0), f"scenarios[{index}].one_time_cost", 0.0)
        balance = reserve - one_time_cost
        minimum_balance = balance
        insolvency_month = 0 if balance < 0.0 else None
        stressed_income = monthly_income * (1.0 + income_change)
        stressed_expenses = monthly_expenses * (1.0 + expense_change)
        for month in range(1, duration + 1):
            balance += stressed_income - stressed_expenses
            minimum_balance = min(minimum_balance, balance)
            if balance < 0.0 and insolvency_month is None:
                insolvency_month = month
        rows.append({
            "name": name,
            "duration_months": duration,
            "ending_reserve": balance,
            "minimum_reserve": minimum_balance,
            "insolvency_month": insolvency_month,
            "monthly_stressed_surplus": stressed_income - stressed_expenses,
        })
    result = _base_result("household_cashflow_stress")
    result.update({
        "scenario_count": len(rows),
        "scenarios": rows,
        "scenarios_with_insolvency": sum(row["insolvency_month"] is not None for row in rows),
    })
    return result


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "goal_based_financial_plan": goal_based_financial_plan,
    "retirement_monte_carlo": retirement_monte_carlo,
    "sequence_of_returns_risk": sequence_of_returns_risk,
    "liability_aware_allocation": liability_aware_allocation,
    "emergency_fund_analysis": emergency_fund_analysis,
    "debt_repayment_optimization": debt_repayment_optimization,
    "withdrawal_strategy": withdrawal_strategy,
    "insurance_protection_gap": insurance_protection_gap,
    "household_cashflow_stress": household_cashflow_stress,
}
