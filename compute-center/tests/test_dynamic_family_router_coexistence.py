from __future__ import annotations

import unittest

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "router coexistence regression",
    "upstream_refs": [],
}


class DynamicFamilyRouterCoexistenceTests(unittest.TestCase):
    def test_indirect_intelligence_mode_specific_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-indirect-coexistence",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "indirect_intelligence_analysis",
                "hypothesis": "技术A已经进入实际应用",
                "evidence": [
                    {
                        "evref": "ev-1",
                        "analysis_class": "DIRECT",
                        "stance": "support",
                        "reliability": 0.8,
                    }
                ],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "indirect-intelligence")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["family"], "indirect-intelligence")
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(len(metadata["requirements"]), 11)
        self.assertIn("requirements-intelligence-splink.txt", metadata["requirements"])
        self.assertIn("requirements-bayesian-network.txt", metadata["requirements"])

    def test_bayesian_operation_route_coexists_with_mode_specific_route(self) -> None:
        ticket = {
            "task_id": "router-bayesian-coexistence",
            "operation": "bayesian_network_inference",
            "inputs": {
                "mode": "bayesian_parameter_estimation",
                "edges": [["A", "B"]],
                "data": {
                    "A": [0, 1] * 10,
                    "B": [0, 1] * 10,
                },
                "query_variables": ["B"],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "bayesian-network")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["family"], "bayesian-network")
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-bayesian-network.txt"])

    def test_state_estimation_mode_specific_route_coexists_with_indirect_route(self) -> None:
        ticket = {
            "task_id": "router-state-estimation-coexistence",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "bounded_linear_kalman_filter",
                "transition_matrix": [[1.0]],
                "observation_matrix": [[1.0]],
                "process_covariance": [[0.05]],
                "observation_covariance": [[0.2]],
                "initial_covariance": [[1.0]],
                "initial_state": [0.0],
                "observations": [[1.0], [1.4], [1.9], [2.3]],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "state-estimation")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["family"], "state-estimation")
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], [])
        self.assertEqual(
            metadata["entry_contract"],
            "finance_decision_analysis:bounded_linear_kalman_filter",
        )


if __name__ == "__main__":
    unittest.main()
