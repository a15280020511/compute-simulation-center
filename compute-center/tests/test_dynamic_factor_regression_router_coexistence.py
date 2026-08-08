from __future__ import annotations

import unittest

from dynamic_family_router import resolve_dynamic_family

PIPELINE = {"pipeline_id": "dynamic-auto-v1", "stage_id": "dynamic", "sequence_reason": "router coexistence", "upstream_refs": []}


class DynamicFactorRegressionRouterCoexistenceTests(unittest.TestCase):
    def test_factor_regression_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "factor-route",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "factor_regression",
                "asset_returns": [float(i) / 100.0 for i in range(10)],
                "factors": {"market": [float(i) / 200.0 for i in range(10)]},
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "factor-regression")

    def test_transport_route_remains_available(self) -> None:
        ticket = {
            "task_id": "transport-route",
            "operation": "finance_decision_analysis",
            "inputs": {"mode": "aequilibrae_shortest_path", "links": [{"a_node": 1, "b_node": 2, "cost": 1.0}], "origin": 1, "destination": 2},
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "transport-routing")

    def test_conformal_route_remains_available(self) -> None:
        train_x = [[float(i), float(i % 3)] for i in range(20)]
        ticket = {
            "task_id": "conformal-route",
            "operation": "finance_decision_analysis",
            "inputs": {"mode": "mapie_conformal_interval", "train_x": train_x, "train_y": [row[0] for row in train_x], "predict_x": [[21.0, 0.0]]},
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "conformal-prediction")


if __name__ == "__main__":
    unittest.main()
