from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compute_runner  # noqa: E402
import tool_registry  # noqa: E402


class ToolRegistryTests(unittest.TestCase):
    def test_registry_modules_match_declared_operations(self):
        operations = tool_registry.load_registered_operations()
        self.assertEqual(len(operations), 23)
        for name in (
            "finance_decision_analysis", "agent_based_simulation", "missing_data_analysis",
            "system_dynamics_simulation", "crisis_early_warning", "information_diffusion_analysis",
            "causal_policy_evaluation", "bayesian_network_inference", "sector_model_analysis",
            "strategic_policy_analysis", "transport_forecast_analysis",
        ):
            self.assertIn(name, operations)

    def test_operation_specific_requirements(self):
        cases = {
            "finance_decision_analysis": ("strategy_backtest", "requirements-finance.txt"),
            "agent_based_simulation": ("network_contagion", "requirements-mesa.txt"),
            "missing_data_analysis": ("mice_multiple_imputation", "requirements-missing-data.txt"),
            "information_diffusion_analysis": ("sir_information_spread", "requirements-diffusion.txt"),
            "causal_policy_evaluation": ("backdoor_adjustment", "requirements-causal.txt"),
            "bayesian_network_inference": ("fixed_network_inference", "requirements-bayesian-network.txt"),
            "sector_model_analysis": ("pypsa_linear_power_flow", "requirements-sector-energy.txt"),
            "strategic_policy_analysis": ("open_spiel_policy_evaluation", "requirements-strategy-open-spiel.txt"),
            "transport_forecast_analysis": ("sumo_micro_simulation", "requirements-sumo.txt"),
        }
        for operation, (mode, requirement) in cases.items():
            observed = tool_registry.requirement_files_for_ticket({"operation": operation, "inputs": {"mode": mode}})
            self.assertEqual([Path(item).name for item in observed], [requirement])

    def test_registry_registers_without_conflicts(self):
        target = dict(compute_runner.OPERATIONS)
        tool_registry.register_into(target)
        self.assertEqual(len(target), 29)


if __name__ == "__main__":
    unittest.main()
