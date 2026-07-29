#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from quality_gate import build_quality_report, evaluate_feedback  # noqa: E402


class QualityGateTests(unittest.TestCase):
    def preflight(self, assumption_ratio=0.0):
        return {
            "status": "DATA_READY",
            "policy": {"enforcement": "strict"},
            "data_summary": {"assumption_ratio": assumption_ratio},
            "issues": [],
        }

    def result(self):
        return {"execution": {"network_used": False, "model_calls": 0}}

    def high_stakes_ticket(self):
        return {
            "task_id": "quality-high-stakes-001",
            "operation": "descriptive_statistics",
            "quality_profile": {
                "decision_class": "high_stakes",
                "benchmark_ids": ["golden-descriptive-001"],
                "independent_cross_check_passed": True,
                "cross_check_method": "independent-reference-implementation-v1",
                "user_approved_for_high_stakes": True,
            },
            "evidence": [
                {
                    "source": "independent cross-check artifact",
                    "sha256": "a" * 64,
                }
            ],
        }

    def test_well_calibrated_formal_result_released(self):
        probabilities = [0.1] * 10 + [0.3] * 10 + [0.7] * 10 + [0.9] * 10
        outcomes = [0] * 9 + [1] + [0] * 7 + [1] * 3 + [0] * 3 + [1] * 7 + [0] + [1] * 9
        report = build_quality_report(
            {
                "task_id": "quality-test-001",
                "operation": "descriptive_statistics",
                "quality_profile": {
                    "decision_class": "formal",
                    "probabilistic_claim": True,
                    "benchmark_ids": ["golden-descriptive-001"],
                },
                "calibration_feedback": {
                    "predicted_probabilities": probabilities,
                    "observed_outcomes": outcomes,
                },
            },
            self.result(),
            self.preflight(),
        )
        self.assertEqual(report["release_status"], "DECISION_RELEASED")
        self.assertTrue(report["decision_grade"])

    def test_high_stakes_requires_independent_cross_check(self):
        report = build_quality_report(
            {
                "task_id": "quality-test-002",
                "operation": "descriptive_statistics",
                "quality_profile": {
                    "decision_class": "high_stakes",
                    "benchmark_ids": ["golden-descriptive-001"],
                    "user_approved_for_high_stakes": True,
                },
            },
            self.result(),
            self.preflight(),
        )
        self.assertEqual(report["release_status"], "DECISION_BLOCKED")
        self.assertFalse(report["constraints"]["formal_decision_use_allowed"])

    def test_high_stakes_requires_explicit_user_approval(self):
        ticket = self.high_stakes_ticket()
        ticket["quality_profile"]["user_approved_for_high_stakes"] = False
        report = build_quality_report(ticket, self.result(), self.preflight())
        self.assertEqual(report["release_status"], "DECISION_BLOCKED")
        codes = {row["code"] for row in report["checks"] if row["status"] == "FAIL"}
        self.assertIn("HIGH_STAKES_USER_APPROVAL", codes)

    def test_self_declared_cross_check_without_hash_is_rejected(self):
        ticket = self.high_stakes_ticket()
        ticket["evidence"] = []
        report = build_quality_report(ticket, self.result(), self.preflight())
        self.assertEqual(report["release_status"], "DECISION_BLOCKED")
        codes = {row["code"] for row in report["checks"] if row["status"] == "FAIL"}
        self.assertIn("INDEPENDENT_CROSS_CHECK", codes)

    def test_unknown_benchmark_id_is_rejected(self):
        ticket = self.high_stakes_ticket()
        ticket["quality_profile"]["benchmark_ids"] = ["invented-benchmark"]
        report = build_quality_report(ticket, self.result(), self.preflight())
        self.assertEqual(report["release_status"], "DECISION_BLOCKED")
        codes = {row["code"] for row in report["checks"] if row["status"] == "FAIL"}
        self.assertIn("BENCHMARK_EVIDENCE", codes)

    def test_fully_evidenced_high_stakes_result_is_released(self):
        report = build_quality_report(
            self.high_stakes_ticket(), self.result(), self.preflight()
        )
        self.assertEqual(report["release_status"], "DECISION_RELEASED")
        self.assertTrue(report["constraints"]["formal_decision_use_allowed"])

    def test_high_stakes_feedback_requires_snapshot_hash(self):
        ticket = self.high_stakes_ticket()
        ticket["calibration_feedback"] = {
            "reference_values": list(range(50)),
            "recent_values": list(range(50)),
        }
        report = build_quality_report(ticket, self.result(), self.preflight())
        self.assertEqual(report["release_status"], "DECISION_BLOCKED")
        codes = {row["code"] for row in report["checks"] if row["status"] == "FAIL"}
        self.assertIn("FEEDBACK_PROVENANCE", codes)

    def test_severe_drift_blocks_reuse(self):
        report = build_quality_report(
            {
                "task_id": "quality-test-003",
                "operation": "descriptive_statistics",
                "quality_profile": {
                    "decision_class": "formal",
                    "benchmark_ids": ["golden-descriptive-001"],
                },
                "calibration_feedback": {
                    "reference_values": list(range(100)),
                    "recent_values": list(range(1000, 1100)),
                },
            },
            self.result(),
            self.preflight(),
        )
        self.assertEqual(report["release_status"], "DECISION_BLOCKED")
        self.assertTrue(report["constraints"]["must_recalibrate_before_reuse"])

    def test_feedback_metrics_are_deterministic(self):
        ticket = {
            "calibration_feedback": {
                "predicted_probabilities": [0.1, 0.2, 0.8, 0.9],
                "observed_outcomes": [0, 0, 1, 1],
                "prediction_intervals": [
                    {"lower": 0, "upper": 2, "actual": 1},
                    {"lower": 3, "upper": 5, "actual": 7},
                ],
            }
        }
        first = evaluate_feedback(ticket)
        second = evaluate_feedback(ticket)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["probability_calibration"]["brier_score"], 0.025)
        self.assertAlmostEqual(first["interval_calibration"]["empirical_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
