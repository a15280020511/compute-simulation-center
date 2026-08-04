from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dispatch = load_module("compute_dispatch_advanced", ROOT / "compute_dispatch.py")


class LightweightSimulationSuiteTests(unittest.TestCase):
    def execute(self, operation: str, inputs: dict) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return dispatch.run_ticket({"task_id": f"advanced-{operation}-001", "operation": operation, "inputs": inputs}, Path(directory.name))

    def test_operation_registry_contains_fixed_suite(self) -> None:
        expected = {"discrete_event_simulation", "repeated_game", "agent_evolution", "time_series_forecast", "causal_screening", "nonlinear_dynamics", "pattern_discovery", "assumption_validation", "markov_simulation", "gis_spatial_analysis", "bayesian_inference", "econometric_analysis", "agent_based_simulation", "finance_decision_analysis", "missing_data_analysis", "system_dynamics_simulation", "crisis_early_warning", "information_diffusion_analysis", "causal_policy_evaluation", "bayesian_network_inference", "sector_model_analysis", "strategic_policy_analysis", "transport_forecast_analysis"}
        self.assertTrue(expected.issubset(dispatch.OPERATIONS))
        schema = json.loads((ROOT / "compute-ticket.schema.json").read_text(encoding="utf-8"))
        declared = set(schema["properties"]["operation"]["enum"])
        self.assertTrue(declared.issubset(set(dispatch.OPERATIONS)))
        self.assertIn("symbolic_mathematics", dispatch.OPERATIONS)
        self.assertNotIn("symbolic_mathematics", declared)

    def test_discrete_event_simulation_is_reproducible(self) -> None:
        inputs = {"seed": 7, "entities": 200, "arrival": {"distribution": "exponential", "mean": 1.2}, "stages": [{"name": "intake", "capacity": 2, "service": {"distribution": "triangular", "minimum": 0.5, "mode": 0.8, "maximum": 1.5}}, {"name": "service", "capacity": 3, "service": {"distribution": "constant", "value": 1.0}}]}
        first = self.execute("discrete_event_simulation", inputs)
        second = self.execute("discrete_event_simulation", inputs)
        self.assertEqual(first["results"], second["results"])
        self.assertEqual(first["results"]["entities_completed"], 200)
        self.assertEqual(first["results"]["engine"]["name"], "simpy")

    def test_repeated_game_fixed_and_adaptive(self) -> None:
        result = self.execute("repeated_game", {"seed": 11, "rounds": 100, "trials": 20, "red_payoffs": [[3, 0], [5, 1]], "blue_payoffs": [[3, 5], [0, 1]], "red_policy": {"type": "epsilon_best_response", "epsilon": 0.05}, "blue_policy": {"type": "fixed", "probabilities": [0.6, 0.4]}})
        self.assertIn("red_mean_total_payoff", result["results"])
        self.assertIn("blue_mean_total_payoff", result["results"])


if __name__ == "__main__":
    unittest.main()
