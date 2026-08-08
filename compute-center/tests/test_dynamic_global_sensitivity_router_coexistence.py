from __future__ import annotations

import unittest

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "global-sensitivity router coexistence regression",
    "upstream_refs": [],
}


class DynamicGlobalSensitivityRouterCoexistenceTests(unittest.TestCase):
    def test_global_sensitivity_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "router-global-sensitivity",
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
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-salib.txt"])

    def test_conformal_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-conformal-after-sensitivity",
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
        self.assertIn("requirements-global-mapie.txt", family_runtime_metadata(ticket)["requirements"])

    def test_robust_allocation_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-robust-after-sensitivity",
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


if __name__ == "__main__":
    unittest.main()
