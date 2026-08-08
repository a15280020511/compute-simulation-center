from __future__ import annotations

import unittest

from dynamic_family_router import FAMILY_BY_OPERATION, FAMILY_BY_OPERATION_MODE


class DynamicCalibrationRouterCoexistenceTests(unittest.TestCase):
    def test_calibration_does_not_replace_existing_dynamic_routes(self) -> None:
        expected_modes = {
            ("finance_decision_analysis", "indirect_intelligence_analysis"): "indirect-intelligence",
            ("finance_decision_analysis", "bounded_linear_kalman_filter"): "state-estimation",
            ("finance_decision_analysis", "mixed_integer_optimization"): "optimization",
            ("finance_decision_analysis", "open_spiel_policy_evaluation"): "game-theory",
            ("finance_decision_analysis", "evidently_data_drift"): "drift",
            ("finance_decision_analysis", "policy_microsimulation"): "policy-simulation",
            ("finance_decision_analysis", "control_step_response"): "control-response",
            ("finance_decision_analysis", "lmfit_exponential_calibration"): "calibration",
        }
        for key, family in expected_modes.items():
            self.assertEqual(FAMILY_BY_OPERATION_MODE.get(key), family)

        expected_operations = {
            "scenario_compare": "scenario-decision",
            "time_series_forecast": "time-series",
            "causal_policy_evaluation": "causal-policy",
            "bayesian_network_inference": "bayesian-network",
            "descriptive_statistics": "reliability",
            "system_dynamics_simulation": "system-dynamics",
        }
        for operation, family in expected_operations.items():
            self.assertEqual(FAMILY_BY_OPERATION.get(operation), family)


if __name__ == "__main__":
    unittest.main()
