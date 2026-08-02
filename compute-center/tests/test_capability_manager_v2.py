from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capability_manager  # noqa: E402


class CapabilityManagerV2Tests(unittest.TestCase):
    def test_registry_is_v2_managed_and_closed(self):
        registry = capability_manager.load_registry()
        self.assertEqual(registry["manager_version"], 2)
        self.assertFalse(registry["arbitrary_modules_allowed"])
        self.assertFalse(registry["arbitrary_requirements_allowed"])
        groups = capability_manager.validated_groups()
        self.assertEqual(len(groups), 11)
        self.assertEqual(len({row["id"] for row in groups}), len(groups))

    def test_runtime_plan_is_bounded_and_offline(self):
        plan = capability_manager.runtime_plan({
            "operation": "finance_decision_analysis",
            "inputs": {"mode": "vehicle_routing"},
        })
        self.assertTrue(plan["managed"])
        self.assertEqual(plan["capability_pack"], "decision-intelligence")
        self.assertEqual(plan["network_policy"], "deny")
        self.assertFalse(plan["arbitrary_code_allowed"])
        self.assertFalse(plan["deterministic"])
        self.assertEqual([Path(item).name for item in plan["requirements"]], ["requirements-finance.txt"])
        self.assertEqual(plan["limits"]["max_nodes"], 200)

    def test_sector_mode_selects_one_pinned_bundle(self):
        plan = capability_manager.runtime_plan({
            "operation": "sector_model_analysis",
            "inputs": {"mode": "pypsa_linear_power_flow"},
        })
        self.assertTrue(plan["managed"])
        self.assertEqual(plan["capability_pack"], "sector-models")
        self.assertEqual(plan["network_policy"], "deny")
        self.assertFalse(plan["arbitrary_code_allowed"])
        self.assertEqual([Path(item).name for item in plan["requirements"]], ["requirements-sector-energy.txt"])

    def test_all_registered_operations_load(self):
        operations = capability_manager.load_registered_operations()
        self.assertEqual(len(operations), 21)
        for name in (
            "finance_decision_analysis", "missing_data_analysis",
            "system_dynamics_simulation", "crisis_early_warning",
            "information_diffusion_analysis", "causal_policy_evaluation",
            "bayesian_network_inference", "sector_model_analysis",
        ):
            self.assertIn(name, operations)


if __name__ == "__main__":
    unittest.main()
