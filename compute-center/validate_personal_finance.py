#!/usr/bin/env python3
"""Execute one fixed offline fixture for each personal-finance mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from personal_finance_operations import HANDLERS


def fixture(mode: str) -> dict:
    if mode == "goal_based_financial_plan":
        return {
            "goals": [
                {"name": "education", "target_amount": 120000.0, "current_balance": 20000.0, "years": 6.0, "annual_return": 0.04},
                {"name": "housing", "target_amount": 200000.0, "current_balance": 50000.0, "years": 8.0, "annual_return": 0.035},
            ]
        }
    if mode == "retirement_monte_carlo":
        return {
            "current_portfolio": 300000.0,
            "annual_contribution": 30000.0,
            "annual_withdrawal": 40000.0,
            "years_to_retirement": 10,
            "retirement_years": 25,
            "expected_return": 0.05,
            "volatility": 0.12,
            "inflation": 0.02,
            "simulations": 300,
            "seed": 17,
        }
    if mode == "sequence_of_returns_risk":
        return {"returns": [0.25, -0.20, 0.15, -0.10, 0.08], "initial_balance": 100000.0, "annual_withdrawal": 18000.0}
    if mode == "liability_aware_allocation":
        return {
            "expected_returns": [0.05, 0.03, 0.02],
            "covariance": [[0.04, 0.01, 0.005], [0.01, 0.02, 0.004], [0.005, 0.004, 0.01]],
            "asset_durations": [2.0, 7.0, 12.0],
            "liabilities": [{"amount": 50000.0, "years": 5.0}, {"amount": 80000.0, "years": 10.0}],
            "current_assets": 150000.0,
            "discount_rate": 0.03,
            "target_return": 0.02,
            "max_weight": 0.8,
        }
    if mode == "emergency_fund_analysis":
        return {"monthly_essential_expenses": 5000.0, "current_liquid_reserve": 10000.0, "target_months": 6, "monthly_saving_capacity": 2000.0}
    if mode == "debt_repayment_optimization":
        return {
            "debts": [
                {"balance": 10000.0, "annual_rate": 0.18, "minimum_payment": 300.0},
                {"balance": 5000.0, "annual_rate": 0.08, "minimum_payment": 150.0},
            ],
            "extra_monthly_payment": 500.0,
            "max_months": 240,
        }
    if mode == "withdrawal_strategy":
        return {"returns": [0.06, -0.12, 0.08, 0.04, -0.03, 0.07], "initial_portfolio": 500000.0, "annual_spending": 24000.0, "inflation": 0.02}
    if mode == "insurance_protection_gap":
        return {
            "outstanding_liabilities": 200000.0,
            "final_expenses": 20000.0,
            "education_funding_need": 100000.0,
            "annual_dependent_support": 50000.0,
            "support_years": 10,
            "discount_rate": 0.03,
            "available_liquid_assets": 100000.0,
            "existing_coverage": 200000.0,
        }
    if mode == "household_cashflow_stress":
        return {
            "monthly_income": 10000.0,
            "monthly_expenses": 9000.0,
            "liquid_reserve": 15000.0,
            "scenarios": [
                {"name": "income-loss", "duration_months": 12, "income_change_pct": -0.8, "expense_change_pct": 0.1, "one_time_cost": 5000.0},
                {"name": "expense-shock", "duration_months": 6, "income_change_pct": 0.0, "expense_change_pct": 0.4, "one_time_cost": 10000.0},
            ],
        }
    raise KeyError(mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(HANDLERS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = HANDLERS[args.mode](fixture(args.mode))
    if result.get("mode") != args.mode:
        raise RuntimeError("handler returned the wrong mode")
    receipt = {
        "status": "PASS",
        "mode": args.mode,
        "network_used": bool(result.get("network_used", False)),
        "model_calls": 0,
        "brokerage_execution": bool(result.get("brokerage_execution", False)),
        "account_access": bool(result.get("account_access", False)),
        "arbitrary_code_used": False,
        "result": result,
    }
    if receipt["network_used"] or receipt["brokerage_execution"] or receipt["account_access"]:
        raise RuntimeError("personal finance mode violated runtime boundaries")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
