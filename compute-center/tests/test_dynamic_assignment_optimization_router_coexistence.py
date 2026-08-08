from __future__ import annotations

import unittest

from dynamic_family_router import resolve_dynamic_family

PIPELINE = {"pipeline_id": "dynamic-auto-v1", "stage_id": "dynamic", "sequence_reason": "router coexistence", "upstream_refs": []}


class DynamicAssignmentOptimizationRouterCoexistenceTests(unittest.TestCase):
    def test_assignment_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "assignment-route", "operation": "finance_decision_analysis",
            "inputs": {"mode": "assignment_optimization", "workers": ["A", "B"], "tasks": ["T1"], "costs": [[1.0], [2.0]], "require_all_tasks": True},
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "assignment-optimization")

    def test_factor_regression_route_remains_available(self) -> None:
        ticket = {
            "task_id": "factor-route", "operation": "finance_decision_analysis",
            "inputs": {"mode": "factor_regression", "asset_returns": [float(i) / 100 for i in range(10)], "factors": {"market": [float(i) / 200 for i in range(10)]}},
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "factor-regression")

    def test_transport_route_remains_available(self) -> None:
        ticket = {
            "task_id": "transport-route", "operation": "finance_decision_analysis",
            "inputs": {"mode": "aequilibrae_shortest_path", "links": [{"a_node": 1, "b_node": 2, "cost": 1.0}], "origin": 1, "destination": 2},
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "transport-routing")


if __name__ == "__main__":
    unittest.main()
