from __future__ import annotations

import math
import unittest

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family


PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "process-mining router coexistence regression",
    "upstream_refs": [],
}


class DynamicProcessMiningRouterCoexistenceTests(unittest.TestCase):
    def test_process_mining_route_is_mode_specific(self) -> None:
        ticket = {
            "task_id": "router-process-mining",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "pm4py_directly_follows",
                "cases": [{"case_id": "c1", "activities": ["A", "B"]}],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "process-mining")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-pm4py.txt"])

    def test_calibration_route_remains_available(self) -> None:
        x = [0.5 * index for index in range(5)]
        ticket = {
            "task_id": "router-calibration-after-process",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "lmfit_exponential_calibration",
                "x": x,
                "y": [3.0 * math.exp(-0.4 * value) + 2.0 for value in x],
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "calibration")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-lmfit.txt"])

    def test_control_route_remains_available(self) -> None:
        ticket = {
            "task_id": "router-control-after-process",
            "operation": "finance_decision_analysis",
            "inputs": {
                "mode": "control_step_response",
                "numerator": [1.0],
                "denominator": [1.0, 1.0],
                "points": 101,
            },
            "pipeline": dict(PIPELINE),
        }
        self.assertEqual(resolve_dynamic_family(ticket), "control-response")
        metadata = family_runtime_metadata(ticket)
        self.assertIn("requirements-global-control.txt", metadata["requirements"])


if __name__ == "__main__":
    unittest.main()
