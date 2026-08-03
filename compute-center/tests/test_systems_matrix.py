from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import systems_matrix  # noqa: E402
import tool_registry  # noqa: E402


class SystemsComputationMatrixTests(unittest.TestCase):
    def test_matrix_covers_every_public_operation(self):
        matrix = systems_matrix.load_systems_matrix()
        expected = systems_matrix.capability_operation_ids()
        self.assertEqual(matrix["schema_version"], "compute-systems-matrix-v1")
        self.assertEqual(len(matrix["routes"]), len(expected))
        self.assertEqual(set(matrix["routes"]), expected)

    def test_decision_routes_require_assumptions_and_stress(self):
        matrix = systems_matrix.load_systems_matrix()
        for operation, route in matrix["routes"].items():
            if route["system_level"] == "decision":
                self.assertIn("assumption_register", route["required_gates"], operation)
                self.assertIn("stress_test", route["required_gates"], operation)

    def test_runtime_plan_contains_systems_route(self):
        plan = tool_registry.managed_runtime_plan({
            "operation": "finance_decision_analysis",
            "inputs": {"mode": "vehicle_routing"},
            "quality_profile": {"decision_class": "high-stakes"},
        })
        route = plan["systems_route"]
        self.assertEqual(route["operation"], "finance_decision_analysis")
        self.assertEqual(route["mode"], "vehicle_routing")
        self.assertEqual(route["runtime_network_policy"], "deny")
        self.assertEqual(
            route["publication_gate"],
            "independent-validation-and-human-approval-required",
        )
        self.assertFalse(route["arbitrary_code_allowed"])

    def test_observation_route_stays_minimal(self):
        route = systems_matrix.route_for_operation("descriptive_statistics")
        self.assertEqual(route["system_level"], "observation")
        self.assertEqual(route["required_gates"], ["input_quality"])


if __name__ == "__main__":
    unittest.main()
