
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finance_operations import finance_decision_analysis  # noqa: E402


class FinanceOperationTests(unittest.TestCase):
    def test_performance_and_business_metrics(self):
        result = finance_decision_analysis(
            {
                "mode": "performance_metrics",
                "returns": [0.01, -0.02, 0.015, 0.005, -0.004],
                "periods_per_year": 252,
                "risk_free_rate": 0.02,
            }
        )
        self.assertEqual(result["mode"], "performance_metrics")
        self.assertIn("maximum_drawdown", result)
        business = finance_decision_analysis(
            {
                "mode": "business_unit_economics",
                "price_per_unit": 100,
                "variable_cost_per_unit": 60,
                "fixed_costs": 10000,
                "expected_units": 400,
                "customer_acquisition_cost": 50,
                "gross_profit_per_customer": 20,
                "retention_months": 12,
            }
        )
        self.assertEqual(business["break_even_units"], 250)
        self.assertEqual(business["ltv_to_cac"], 4.8)

    def test_projection_and_capital_budgeting(self):
        projection = finance_decision_analysis(
            {
                "mode": "investment_projection",
                "initial_principal": 10000,
                "monthly_contribution": 1000,
                "annual_return": 0.08,
                "annual_fee_rate": 0.01,
                "annual_inflation_rate": 0.02,
                "years": 5,
            }
        )
        self.assertGreater(projection["nominal_ending_value"], projection["total_contributions"])
        capital = finance_decision_analysis(
            {
                "mode": "capital_budgeting",
                "cash_flows": [-1000, 400, 400, 400],
                "discount_rate": 0.08,
            }
        )
        self.assertGreater(capital["net_present_value"], 0)
        self.assertIsNotNone(capital["internal_rate_of_return"])

    def test_fixed_strategy_backtest(self):
        prices = [100 + index * 0.4 + (index % 5) * 0.2 for index in range(80)]
        result = finance_decision_analysis(
            {
                "mode": "strategy_backtest",
                "prices": prices,
                "strategy": "moving_average_crossover",
                "fast_window": 5,
                "slow_window": 15,
                "initial_cash": 100000,
                "fees": 0.001,
                "slippage": 0.0005,
                "periods_per_year": 252,
            }
        )
        self.assertEqual(result["mode"], "strategy_backtest")
        self.assertEqual(result["strategy"], "moving_average_crossover")
        self.assertGreater(result["final_value"], 0)
        self.assertFalse(result["arbitrary_strategy_code_allowed"])

    def test_portfolio_optimization(self):
        result = finance_decision_analysis(
            {
                "mode": "portfolio_optimization",
                "prices": {
                    "A": [100, 101, 103, 102, 105, 107, 109, 110],
                    "B": [100, 100.5, 101, 101.5, 102, 102.5, 103, 104],
                    "C": [100, 99, 100, 102, 101, 104, 103, 106],
                },
                "objective": "min_volatility",
            }
        )
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
