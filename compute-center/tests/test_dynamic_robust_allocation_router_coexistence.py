from __future__ import annotations

import unittest

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "robust-allocation router coexistence regression",
    "upstream_refs": [],
}


class DynamicRobustAllocationRouterCoexistenceTests(unittest.TestCase):
    def test_robust_allocation_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "router-robust-allocation",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "rsome_robust_allocation",
                "scenario_returns": [[0.1, 0.0], [0.0, 0.1]],
                "asset_names": ["A", "B"],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "robust-allocation")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-rsome.txt"])

    def test_process_mining_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-process-after-robust",
            "operation": "finance_decision_analysis",
            "inputs": {"mode": "pm4py_directly_follows", "cases": [{"case_id": "c1", "activities": ["A", "B"]}]},
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "process-mining")
        self.assertIn("requirements-global-pm4py.txt", family_runtime_metadata(ticket)["requirements"])

    def test_optimization_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-optimization-after-robust",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "mixed_integer_optimization",
                "variables": [{"name": "x", "type": "continuous", "lower_bound": 0.0, "upper_bound": 1.0, "objective_coefficient": 1.0}],
                "constraints": [],
                "maximize": True,
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "optimization")
        self.assertIn("requirements-thinktank-decision.txt", family_runtime_metadata(ticket)["requirements"])


if __name__ == "__main__":
    unittest.main()
