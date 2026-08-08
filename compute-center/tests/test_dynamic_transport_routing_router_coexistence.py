from __future__ import annotations

import unittest

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family

PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "transport-routing router coexistence regression",
    "upstream_refs": [],
}


class DynamicTransportRoutingRouterCoexistenceTests(unittest.TestCase):
    def test_transport_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "router-transport-routing",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "aequilibrae_shortest_path",
                "links": [
                    {"a_node": 1, "b_node": 2, "cost": 1.0},
                    {"a_node": 2, "b_node": 3, "cost": 1.0},
                    {"a_node": 1, "b_node": 3, "cost": 3.0},
                ],
                "origin": 1,
                "destination": 3,
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "transport-routing")
        self.assertEqual(
            family_runtime_metadata(ticket)["requirements"],
            ["requirements-ortools.txt", "requirements-global-aequilibrae.txt"],
        )

    def test_global_sensitivity_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-sensitivity-after-transport",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "sobol_sensitivity",
                "parameters": [
                    {"name": "x", "minimum": -1.0, "maximum": 1.0},
                    {"name": "y", "minimum": -1.0, "maximum": 1.0},
                ],
                "model": {"intercept": 0.0, "linear": {"x": 1.0, "y": 2.0}, "quadratic": {}, "interactions": []},
                "base_samples": 256,
                "seed": 0,
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "global-sensitivity")
        self.assertIn("requirements-global-salib.txt", family_runtime_metadata(ticket)["requirements"])

    def test_conformal_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-conformal-after-transport",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "mapie_conformal_interval",
                "train_x": [[float(index)] for index in range(20)],
                "train_y": [2.0 * index + 1.0 for index in range(20)],
                "predict_x": [[20.0], [21.0]],
                "confidence": 0.9,
                "cv": 5,
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "conformal-prediction")
        self.assertIn("requirements-global-mapie.txt", family_runtime_metadata(ticket)["requirements"])


if __name__ == "__main__":
    unittest.main()
