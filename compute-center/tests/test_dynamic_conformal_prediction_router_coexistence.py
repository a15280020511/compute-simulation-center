from __future__ import annotations

import unittest

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "conformal-prediction router coexistence regression",
    "upstream_refs": [],
}


class DynamicConformalPredictionRouterCoexistenceTests(unittest.TestCase):
    def test_conformal_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "router-conformal",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "mapie_conformal_interval",
                "train_x": [[float(index)] for index in range(20)],
                "train_y": [2.0 * index + 1.0 for index in range(20)],
                "predict_x": [[20.0], [21.0], [22.0], [23.0]],
                "confidence": 0.9,
                "cv": 5,
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "conformal-prediction")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-mapie.txt"])

    def test_robust_allocation_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-robust-after-conformal",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "rsome_robust_allocation",
                "scenario_returns": [[0.1, 0.0], [0.0, 0.1]],
                "asset_names": ["A", "B"],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "robust-allocation")
        self.assertIn("requirements-global-rsome.txt", family_runtime_metadata(ticket)["requirements"])

    def test_process_mining_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-process-after-conformal",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "pm4py_directly_follows",
                "cases": [{"case_id": "c1", "activities": ["A", "B"]}],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "process-mining")
        self.assertIn("requirements-global-pm4py.txt", family_runtime_metadata(ticket)["requirements"])


if __name__ == "__main__":
    unittest.main()
