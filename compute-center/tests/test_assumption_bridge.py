from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assumption_library import assess_assumptions


class LightweightAssumptionBridgeTests(unittest.TestCase):
    def test_lightweight_assumption_is_counted_and_hashed(self):
        ticket = {
            "task_id": "assumption-bridge-basic",
            "operation": "scenario_compare",
            "objective": "Verify lightweight assumption governance.",
            "quality_profile": {"decision_class": "exploratory"},
            "assumptions": [
                {
                    "name": "alliance persistence",
                    "value": "high",
                    "basis": "Declared test fixture.",
                    "confidence": "medium",
                }
            ],
        }
        first = assess_assumptions(ticket)
        second = assess_assumptions(ticket)
        self.assertEqual(first, second)
        self.assertEqual(first["inline_assumption_count"], 1)
        self.assertEqual(first["registered_assumption_count"], 0)
        self.assertEqual(first["lightweight_assumption_count"], 1)
        self.assertEqual(first["resolved_assumption_count"], 1)
        self.assertEqual(len(first["resolved_snapshot_sha256"]), 64)
        self.assertFalse(
            any(row["code"] == "NO_EXPLICIT_ASSUMPTIONS" for row in first["issues"])
        )
        resolved = first["resolved_assumptions"][0]
        self.assertTrue(resolved["assumption_id"].startswith("ticket-assumption-1-"))
        self.assertEqual(resolved["source_type"], "gpts_assumption")
        self.assertEqual(resolved["status"], "proposed")

    def test_low_confidence_lightweight_assumption_gets_uncertainty_treatment(self):
        report = assess_assumptions(
            {
                "task_id": "assumption-bridge-low-confidence",
                "operation": "monte_carlo",
                "quality_profile": {"decision_class": "exploratory"},
                "assumptions": [
                    {
                        "name": "unknown response",
                        "value": "unresolved",
                        "basis": "Bounded elicitation fixture.",
                        "confidence": "low",
                    }
                ],
            }
        )
        resolved = report["resolved_assumptions"][0]
        self.assertEqual(resolved["distribution"], "scenario_set")
        self.assertTrue(resolved["sensitivity_required"])
        self.assertEqual(resolved["uncertainty_type"], "mixed")

    def test_sensitivity_range_is_normalized_to_uniform_bounds(self):
        report = assess_assumptions(
            {
                "task_id": "assumption-bridge-range",
                "operation": "sensitivity_analysis",
                "quality_profile": {"decision_class": "exploratory"},
                "assumptions": [
                    {
                        "name": "demand range",
                        "basis": "Fixture range.",
                        "confidence": "medium",
                        "sensitivity_range": {"minimum": 0.2, "maximum": 0.8},
                    }
                ],
            }
        )
        resolved = report["resolved_assumptions"][0]
        self.assertEqual(resolved["distribution"], "uniform")
        self.assertEqual(resolved["minimum"], 0.2)
        self.assertEqual(resolved["maximum"], 0.8)

    def test_registered_and_lightweight_assumptions_are_merged(self):
        registered = {
            "assumption_id": "registered-001",
            "type": "parameter",
            "statement": "A registered assumption.",
            "source_type": "historical",
            "basis": "Frozen fixture.",
            "confidence": "high",
            "calibration_status": "calibrated",
            "uncertainty_type": "deterministic",
            "criticality": "medium",
            "evidence_strength": "moderate",
            "status": "approved",
        }
        report = assess_assumptions(
            {
                "task_id": "assumption-bridge-merge",
                "operation": "descriptive_statistics",
                "quality_profile": {"decision_class": "formal"},
                "assumption_register": [registered],
                "assumptions": [
                    {
                        "name": "lightweight row",
                        "value": True,
                        "basis": "Fixture.",
                        "confidence": "high",
                    }
                ],
            }
        )
        self.assertEqual(report["registered_assumption_count"], 1)
        self.assertEqual(report["lightweight_assumption_count"], 1)
        self.assertEqual(report["inline_assumption_count"], 2)
        self.assertEqual(report["resolved_assumption_count"], 2)
        self.assertEqual(len({row["assumption_id"] for row in report["resolved_assumptions"]}), 2)

    def test_high_stakes_unapproved_lightweight_assumption_blocks(self):
        report = assess_assumptions(
            {
                "task_id": "assumption-bridge-high-stakes",
                "operation": "monte_carlo",
                "quality_profile": {"decision_class": "high_stakes"},
                "assumptions": [
                    {
                        "name": "critical unknown",
                        "value": 1,
                        "basis": "Fixture.",
                        "confidence": "high",
                    }
                ],
            }
        )
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(
            any(
                row["code"] == "UNAPPROVED_LIGHTWEIGHT_ASSUMPTION"
                and row["blocking"]
                for row in report["issues"]
            )
        )

    def test_maximum_lightweight_assumption_load_is_deterministic(self):
        assumptions = [
            {
                "name": f"assumption-{index:03d}",
                "value": index,
                "basis": "Maximum ticket-load fixture.",
                "confidence": "medium",
                "approved_by": "gpts_policy",
            }
            for index in range(100)
        ]
        ticket = {
            "task_id": "assumption-bridge-max-load",
            "operation": "scenario_compare",
            "quality_profile": {"decision_class": "formal"},
            "assumptions": assumptions,
        }
        first = assess_assumptions(ticket)
        second = assess_assumptions(ticket)
        self.assertEqual(first, second)
        self.assertEqual(first["lightweight_assumption_count"], 100)
        self.assertEqual(first["resolved_assumption_count"], 100)
        self.assertEqual(
            len({row["assumption_id"] for row in first["resolved_assumptions"]}),
            100,
        )


if __name__ == "__main__":
    unittest.main()
