from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assurance_operations import HANDLERS as ASSURANCE_HANDLERS  # noqa: E402
from decision_intelligence_gateway import (  # noqa: E402
    ALL_SUPPORTED_MODES,
    SUPPORTED_MODES,
    finance_decision_analysis,
)
from operations_research_modes import assignment_optimization, mixed_integer_optimization, vehicle_routing  # noqa: E402
from professional_forecasting_operations import (  # noqa: E402
    exponential_smoothing_forecast,
    sarimax_forecast,
    sobol_sensitivity,
    vector_autoregression_forecast,
)
from quantitative_operations import (  # noqa: E402
    factor_regression,
    portfolio_stress_test,
    risk_parity_allocation,
    walk_forward_backtest,
)
from strategic_intelligence_operations import (  # noqa: E402
    competing_hypotheses,
    indicators_and_warnings,
    minimax_regret,
    value_of_information,
    weighted_mcda,
)
from think_tank_operations import SUPPORTED_MODES as THINK_TANK_MODES  # noqa: E402


class StrategicIntelligenceTests(unittest.TestCase):
    def test_all_new_modes_are_allowlisted(self):
        expected = {
            "factor_regression", "walk_forward_backtest", "risk_parity_allocation",
            "portfolio_stress_test", "sarimax_forecast", "exponential_smoothing_forecast",
            "vector_autoregression_forecast", "sobol_sensitivity", "mixed_integer_optimization",
            "assignment_optimization", "vehicle_routing", "weighted_mcda", "minimax_regret",
            "value_of_information", "competing_hypotheses", "indicators_and_warnings",
        }
        self.assertTrue(expected <= set(SUPPORTED_MODES))
        self.assertTrue(set(THINK_TANK_MODES) <= set(ALL_SUPPORTED_MODES))
        self.assertTrue(set(ASSURANCE_HANDLERS) <= set(ALL_SUPPORTED_MODES))
        self.assertTrue(set(SUPPORTED_MODES).isdisjoint(set(THINK_TANK_MODES)))
        self.assertTrue(set(SUPPORTED_MODES).isdisjoint(set(ASSURANCE_HANDLERS)))
        self.assertTrue(set(THINK_TANK_MODES).isdisjoint(set(ASSURANCE_HANDLERS)))
        self.assertEqual(len(SUPPORTED_MODES), 22)
        self.assertEqual(len(THINK_TANK_MODES), 53)
        self.assertEqual(len(ASSURANCE_HANDLERS), 8)
        self.assertEqual(len(ALL_SUPPORTED_MODES), 83)

    def test_weighted_mcda_ranks_alternatives(self):
        result = weighted_mcda({
            "criteria": [
                {"name": "benefit", "weight": 0.4, "direction": "benefit"},
                {"name": "cost", "weight": 0.6, "direction": "cost"},
            ],
            "alternatives": [
                {"name": "A", "values": {"benefit": 80, "cost": 60}},
                {"name": "B", "values": {"benefit": 70, "cost": 20}},
            ],
        })
        self.assertEqual(result["recommended_alternative"], "B")

    def test_minimax_regret_and_value_of_information(self):
        regret = minimax_regret({
            "actions": ["A", "B"], "scenarios": ["low", "high"],
            "payoffs": [[10, 30], [18, 25]], "objective": "maximize",
        })
        self.assertEqual(regret["robust_action"], "B")
        voi = value_of_information({
            "actions": ["A", "B"], "scenarios": ["low", "high"],
            "probabilities": [0.5, 0.5], "payoffs": [[10, 30], [18, 25]],
        })
        self.assertAlmostEqual(voi["expected_value_of_perfect_information"], 2.5)

    def test_competing_hypotheses_and_warning_indicators(self):
        ach = competing_hypotheses({
            "hypotheses": ["H1", "H2"],
            "evidence": [
                {"id": "E1", "reliability": 0.9, "diagnosticity": 1.0, "ratings": {"H1": 2, "H2": -2}},
                {"id": "E2", "reliability": 0.5, "diagnosticity": 0.5, "ratings": {"H1": 0, "H2": 1}},
            ],
        })
        self.assertEqual(ach["leading_hypothesis"], "H1")
        warnings = indicators_and_warnings({
            "indicators": [
                {"name": "liquidity", "current": 20, "warning_threshold": 30, "critical_threshold": 15, "direction": "lower_is_worse", "reliability": 1, "importance": 1},
                {"name": "defaults", "current": 5, "warning_threshold": 3, "critical_threshold": 5, "direction": "higher_is_worse", "reliability": 1, "importance": 1},
            ]
        })
        self.assertEqual(warnings["critical_count"], 1)

    def test_gateway_delegates_legacy_and_new_modes(self):
        legacy = finance_decision_analysis({"mode": "performance_metrics", "returns": [0.01, -0.01, 0.02]})
        self.assertEqual(legacy["mode"], "performance_metrics")
        self.assertTrue(legacy["no_guaranteed_profit"])
        new = finance_decision_analysis({
            "mode": "portfolio_stress_test",
            "weights": {"equity": 0.6, "bond": 0.4},
            "scenarios": [{"name": "shock", "asset_shocks": {"equity": -0.3, "bond": 0.05}}],
        })
        self.assertTrue(new["no_guaranteed_profit"])
        self.assertFalse(new["brokerage_execution"])
        self.assertFalse(new["arbitrary_code_allowed"])


class QuantitativeTests(unittest.TestCase):
    def test_walk_forward_has_out_of_sample_controls(self):
        prices = [100 + 0.4 * i + (2 if i % 15 < 8 else -2) for i in range(160)]
        result = walk_forward_backtest({
            "prices": prices, "fast_window": 5, "slow_window": 20,
            "validation_window": 20, "fee_rate": 0.001, "slippage_rate": 0.001,
        })
        self.assertGreaterEqual(result["fold_count"], 5)
        self.assertIn("lookahead_bias_control", result)
        self.assertFalse(result["arbitrary_strategy_code_allowed"])

    def test_risk_parity_and_stress_test(self):
        result = risk_parity_allocation({
            "returns_by_asset": {
                "A": [0.01, -0.01, 0.02, 0.0, 0.015, -0.005, 0.01, 0.005, -0.002, 0.012],
                "B": [0.003, 0.002, -0.001, 0.004, 0.003, 0.002, -0.002, 0.004, 0.001, 0.003],
            }
        })
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=6)
        stress = portfolio_stress_test({
            "weights": result["weights"],
            "scenarios": [{"name": "crash", "asset_shocks": {"A": -0.4, "B": -0.05}}],
        })
        self.assertEqual(stress["worst_scenario"], "crash")


class OptionalEngineIntegrationTests(unittest.TestCase):
    def test_statsmodels_factor_and_forecast_modes(self):
        factor = [(value - 20) / 1000 for value in range(1, 41)]
        asset = [0.001 + 1.5 * value for value in factor]
        regression = factor_regression({
            "asset_returns": asset, "factors": {"market": factor}, "covariance_type": "HC1",
        })
        self.assertAlmostEqual(regression["parameters"]["market"]["coefficient"], 1.5, places=5)
        data = [10 + 0.4 * index + (1.5 if index % 12 < 6 else -1.5) for index in range(60)]
        smoothing = exponential_smoothing_forecast({"data": data, "horizon": 4, "holdout": 6, "trend": "add"})
        self.assertEqual(len(smoothing["forecast"]), 4)
        sarimax = sarimax_forecast({"data": data, "horizon": 3, "holdout": 5, "order": [1, 1, 0]})
        self.assertEqual(len(sarimax["forecast"]), 3)
        series = {
            "demand": [50 + index * 0.3 + (index % 4) for index in range(50)],
            "price": [20 + index * 0.1 + ((index + 1) % 3) for index in range(50)],
        }
        var = vector_autoregression_forecast({"series": series, "horizon": 3, "holdout": 4, "max_lags": 2})
        self.assertEqual(len(var["forecast"]), 3)

    def test_salib_sobol_mode(self):
        result = sobol_sensitivity({
            "parameters": [
                {"name": "demand", "minimum": 80, "maximum": 120},
                {"name": "margin", "minimum": 5, "maximum": 15},
            ],
            "base_samples": 64, "seed": 7,
            "model": {
                "intercept": 0,
                "linear": {"demand": 1.0, "margin": 3.0},
                "interactions": [{"left": "demand", "right": "margin", "coefficient": 0.02}],
            },
        })
        self.assertEqual(len(result["ranking"]), 2)
        self.assertEqual(result["seed"], 7)

    def test_ortools_modes(self):
        mip = mixed_integer_optimization({
            "variables": [
                {"name": "x", "type": "integer", "lower": 0, "upper": 10, "objective": 3},
                {"name": "y", "type": "integer", "lower": 0, "upper": 10, "objective": 2},
            ],
            "constraints": [{"coefficients": {"x": 1, "y": 1}, "relation": "<=", "rhs": 4}],
            "maximize": True, "time_limit_seconds": 5,
        })
        self.assertAlmostEqual(mip["objective_value"], 12.0)
        assignment = assignment_optimization({
            "workers": ["w1", "w2"], "tasks": ["t1", "t2"], "costs": [[1, 8], [7, 2]],
        })
        self.assertEqual({(row["worker"], row["task"]) for row in assignment["assignments"]}, {("w1", "t1"), ("w2", "t2")})
        routing = vehicle_routing({
            "distance_matrix": [[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]],
            "vehicle_count": 1, "depot": 0, "time_limit_seconds": 2,
        })
        self.assertEqual(routing["routes"][0]["route"][0], 0)
        self.assertEqual(routing["routes"][0]["route"][-1], 0)
        self.assertGreater(routing["total_distance"], 0)


if __name__ == "__main__":
    unittest.main()
