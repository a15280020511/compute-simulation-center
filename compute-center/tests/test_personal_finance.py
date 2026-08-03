from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personal_finance_operations import (  # noqa: E402
    HANDLERS,
    debt_repayment_optimization,
    emergency_fund_analysis,
    goal_based_financial_plan,
    household_cashflow_stress,
    insurance_protection_gap,
    liability_aware_allocation,
    retirement_monte_carlo,
    sequence_of_returns_risk,
    withdrawal_strategy,
)


class PersonalFinanceModeTests(unittest.TestCase):
    def test_fixed_mode_catalog_and_boundaries(self) -> None:
        self.assertEqual(len(HANDLERS), 9)
        for name, handler in HANDLERS.items():
            self.assertTrue(callable(handler), name)

    def test_goal_plan_computes_required_contribution(self) -> None:
        result = goal_based_financial_plan({
            "goals": [{
                "name": "education",
                "target_amount": 100000.0,
                "current_balance": 20000.0,
                "years": 5.0,
                "annual_return": 0.04,
            }]
        })
        self.assertGreater(result["total_required_monthly_contribution"], 0.0)
        self.assertFalse(result["network_used"])

    def test_retirement_simulation_is_seeded_and_bounded(self) -> None:
        inputs = {
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
        first = retirement_monte_carlo(inputs)
        second = retirement_monte_carlo(inputs)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["success_probability"], 0.0)
        self.assertLessEqual(first["success_probability"], 1.0)

    def test_sequence_risk_changes_ending_wealth(self) -> None:
        result = sequence_of_returns_risk({
            "returns": [0.25, -0.20, 0.15, -0.10, 0.08],
            "initial_balance": 100000.0,
            "annual_withdrawal": 18000.0,
        })
        self.assertGreater(result["ending_balance_range"], 0.0)
        self.assertTrue(result["same_return_multiset"])

    def test_liability_allocation_matches_dimensions(self) -> None:
        result = liability_aware_allocation({
            "expected_returns": [0.05, 0.03, 0.02],
            "covariance": [
                [0.04, 0.01, 0.005],
                [0.01, 0.02, 0.004],
                [0.005, 0.004, 0.01],
            ],
            "asset_durations": [2.0, 7.0, 12.0],
            "liabilities": [
                {"amount": 50000.0, "years": 5.0},
                {"amount": 80000.0, "years": 10.0},
            ],
            "current_assets": 150000.0,
            "discount_rate": 0.03,
            "target_return": 0.02,
            "max_weight": 0.8,
        })
        self.assertEqual(len(result["weights"]), 3)
        self.assertAlmostEqual(sum(result["weights"]), 1.0, places=7)
        self.assertFalse(result["brokerage_execution"])

    def test_emergency_fund_reports_gap_and_time(self) -> None:
        result = emergency_fund_analysis({
            "monthly_essential_expenses": 5000.0,
            "current_liquid_reserve": 10000.0,
            "target_months": 6,
            "monthly_saving_capacity": 2000.0,
        })
        self.assertEqual(result["funding_gap"], 20000.0)
        self.assertEqual(result["months_to_target"], 10)

    def test_debt_avalanche_and_snowball_complete(self) -> None:
        result = debt_repayment_optimization({
            "debts": [
                {"balance": 10000.0, "annual_rate": 0.18, "minimum_payment": 300.0},
                {"balance": 5000.0, "annual_rate": 0.08, "minimum_payment": 150.0},
            ],
            "extra_monthly_payment": 500.0,
            "max_months": 240,
        })
        self.assertEqual(result["strategies"]["avalanche"]["status"], "PAID")
        self.assertEqual(result["strategies"]["snowball"]["status"], "PAID")
        self.assertLessEqual(
            result["strategies"]["avalanche"]["total_interest"],
            result["strategies"]["snowball"]["total_interest"] + 1e-8,
        )

    def test_withdrawal_strategies_are_comparable(self) -> None:
        result = withdrawal_strategy({
            "returns": [0.06, -0.12, 0.08, 0.04, -0.03, 0.07],
            "initial_portfolio": 500000.0,
            "annual_spending": 24000.0,
            "inflation": 0.02,
        })
        self.assertEqual(set(result["strategies"]), {"fixed_real", "percentage", "guardrails"})
        self.assertTrue(result["comparison_only"])

    def test_insurance_gap_is_not_underwriting(self) -> None:
        result = insurance_protection_gap({
            "outstanding_liabilities": 200000.0,
            "final_expenses": 20000.0,
            "education_funding_need": 100000.0,
            "annual_dependent_support": 50000.0,
            "support_years": 10,
            "discount_rate": 0.03,
            "available_liquid_assets": 100000.0,
            "existing_coverage": 200000.0,
        })
        self.assertGreaterEqual(result["protection_gap"], 0.0)
        self.assertFalse(result["underwriting_performed"])
        self.assertFalse(result["product_recommendation_provided"])

    def test_cashflow_stress_detects_insolvency(self) -> None:
        result = household_cashflow_stress({
            "monthly_income": 10000.0,
            "monthly_expenses": 9000.0,
            "liquid_reserve": 15000.0,
            "scenarios": [
                {
                    "name": "income-loss",
                    "duration_months": 12,
                    "income_change_pct": -0.8,
                    "expense_change_pct": 0.1,
                    "one_time_cost": 5000.0,
                }
            ],
        })
        self.assertEqual(result["scenarios_with_insolvency"], 1)
        self.assertIsNotNone(result["scenarios"][0]["insolvency_month"])


if __name__ == "__main__":
    unittest.main()
