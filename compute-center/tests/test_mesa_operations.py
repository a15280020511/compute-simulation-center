from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MESA_AVAILABLE = importlib.util.find_spec("mesa") is not None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dispatch = load_module("compute_dispatch_mesa", ROOT / "compute_dispatch.py")
mesa_operations = load_module("mesa_operations_tested", ROOT / "mesa_operations.py")


@unittest.skipUnless(MESA_AVAILABLE, "Mesa optional engine is not installed in the core environment")
class MesaOperationTests(unittest.TestCase):
    def execute(self, inputs: dict) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return dispatch.run_ticket({"task_id": "mesa-agent-based-simulation-001", "operation": "agent_based_simulation", "inputs": inputs}, Path(directory.name))["results"]

    def test_registry_exposes_mesa_operation(self) -> None:
        self.assertIn("agent_based_simulation", dispatch.OPERATIONS)
        self.assertEqual(len(dispatch.OPERATIONS), 28)

    def test_worker_choice_is_reproducible(self) -> None:
        inputs = {"mode": "heterogeneous_worker_choice", "agent_count": 80, "steps": 20, "seed": 20260728, "learning_rate": 0.2, "choice_sensitivity": 2.0, "switching_cost": 0.5, "preference_standard_deviation": 0.4, "reward_standard_deviation": 0.2, "options": [{"name": "zone_a", "base_reward": 20, "cost": 5, "capacity": 35, "congestion_penalty": 8}, {"name": "zone_b", "base_reward": 18, "cost": 3, "capacity": 50, "congestion_penalty": 4}]}
        first = self.execute(inputs); second = self.execute(inputs)
        self.assertEqual(first, second)
        self.assertEqual(first["engine"]["name"], "mesa")
        self.assertAlmostEqual(sum(first["final_shares"]), 1.0)

    def test_network_contagion_is_bounded(self) -> None:
        result = self.execute({"mode": "network_contagion", "agent_count": 120, "steps": 30, "seed": 7, "average_degree": 6, "initial_adoption_rate": 0.05, "threshold_mean": 0.3, "threshold_standard_deviation": 0.08, "external_influence": 0.03, "recovery_rate": 0.0})
        self.assertGreaterEqual(result["final_adoption_rate"], 0.0)
        self.assertLessEqual(result["final_adoption_rate"], 1.0)
        self.assertLessEqual(result["edge_count"], mesa_operations.MAX_EDGES)

    def test_resource_competition(self) -> None:
        result = self.execute({"mode": "resource_competition", "agent_count": 60, "steps": 15, "seed": 11, "demand_mean": 0.8, "demand_standard_deviation": 0.1, "resources": [{"name": "resource_a", "initial_stock": 100, "capacity": 100, "regeneration": 20, "unit_value": 2}, {"name": "resource_b", "initial_stock": 80, "capacity": 80, "regeneration": 15, "unit_value": 3}]})
        self.assertEqual(len(result["resources"]), 2)
        self.assertGreaterEqual(result["agent_reward"]["gini"], 0.0)

    def test_all_social_behavior_modes_are_reproducible_and_bounded(self) -> None:
        modes = ["prospect_theory_choice", "bounded_rational_adoption", "trust_update", "social_norm_compliance", "risk_perception", "fatigue_and_adaptation", "institutional_confidence", "group_identity_choice"]
        for mode in modes:
            inputs = {"mode": mode, "agent_count": 40, "steps": 8, "seed": 19, "initial_mean": 0.5, "initial_standard_deviation": 0.1}
            first = self.execute(inputs); second = self.execute(inputs)
            self.assertEqual(first, second, mode)
            self.assertGreaterEqual(first["result_distribution"]["mean"], 0.0)
            self.assertLessEqual(first["result_distribution"]["mean"], 1.0)
            self.assertIn("interpretation_boundary", first)

    def test_unknown_mode_and_limits_are_rejected(self) -> None:
        with self.assertRaises(dispatch.ComputeError):
            dispatch.OPERATIONS["agent_based_simulation"]({"mode": "run_python"})
        with self.assertRaises(dispatch.ComputeError):
            dispatch.OPERATIONS["agent_based_simulation"]({"mode": "network_contagion", "agent_count": mesa_operations.MAX_AGENTS + 1, "steps": 1, "seed": 0})


if __name__ == "__main__":
    unittest.main()
