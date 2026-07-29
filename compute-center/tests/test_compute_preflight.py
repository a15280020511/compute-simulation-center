from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compute_preflight", ROOT / "compute_preflight.py"
)
assert SPEC and SPEC.loader
compute_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute_preflight)


class ComputePreflightTests(unittest.TestCase):
    def base_ticket(self) -> dict:
        return {
            "task_id": "preflight-test-0001",
            "operation": "descriptive_statistics",
            "inputs": {"values": [1, 2, 3]},
            "data_context": {
                "variables": [
                    {
                        "name": "values",
                        "required": True,
                        "source_type": "user_provided",
                        "confidence": "high",
                        "sample_size": 3,
                        "missing": False,
                        "replacement_strategy": "none",
                        "characteristics": {
                            "skewed": False,
                            "outlier_rate": 0,
                            "weights_available": False,
                            "group_dimensions": [],
                            "time_series": False,
                        },
                    }
                ]
            },
        }

    def test_ready_ticket_is_allowed(self) -> None:
        result = compute_preflight.assess(self.base_ticket())
        self.assertEqual(result["status"], "DATA_READY")
        self.assertTrue(result["execution_allowed"])
        self.assertEqual(result["security"]["model_calls"], 0)
        self.assertFalse(result["security"]["external_data_fetch_used"])

    def test_missing_required_data_blocks(self) -> None:
        ticket = self.base_ticket()
        ticket["data_context"]["variables"][0]["missing"] = True
        result = compute_preflight.assess(ticket)
        self.assertEqual(result["status"], "DATA_INSUFFICIENT")
        self.assertFalse(result["execution_allowed"])
        self.assertIn(
            "REQUIRED_DATA_MISSING",
            {item["code"] for item in result["issues"]},
        )

    def test_low_confidence_assumption_requires_user_approval(self) -> None:
        ticket = self.base_ticket()
        ticket["preflight_policy"] = {"max_assumption_ratio": 1.0}
        ticket["data_context"]["variables"][0].update(
            {
                "source_type": "gpts_assumption",
                "confidence": "low",
            }
        )
        ticket["assumptions"] = [
            {
                "name": "values",
                "value": 2,
                "basis": "missing observation",
                "confidence": "low",
                "source_type": "gpts_assumption",
                "sensitivity_range": {"minimum": 1, "maximum": 3},
                "invalid_when": "observed data becomes available",
                "approved_by": "not_approved",
            }
        ]
        result = compute_preflight.assess(ticket)
        self.assertEqual(result["status"], "USER_APPROVAL_REQUIRED")
        self.assertFalse(result["execution_allowed"])
        self.assertIn("monte_carlo", result["recommended_operations"])

    def test_representative_value_rules(self) -> None:
        ticket = self.base_ticket()
        ticket["data_context"]["variables"][0]["characteristics"]["skewed"] = True
        result = compute_preflight.assess(ticket)
        recommendation = result["representative_value_recommendations"][0]
        self.assertEqual(recommendation["method"], "median")
        self.assertFalse(recommendation["computed_by_preflight"])

    def test_invalid_probability_vector_blocks(self) -> None:
        ticket = self.base_ticket()
        ticket["inputs"] = {"probabilities": [0.8, 0.8]}
        result = compute_preflight.assess(ticket)
        self.assertEqual(result["status"], "DATA_INSUFFICIENT")
        self.assertIn(
            "INVALID_PROBABILITY_VECTOR",
            {item["code"] for item in result["issues"]},
        )


if __name__ == "__main__":
    unittest.main()
