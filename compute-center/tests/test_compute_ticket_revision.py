from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compute_ticket
import library_runtime


class ComputeTicketRevisionTests(unittest.TestCase):
    def test_exploratory_unknown_method_is_removed_with_warning(self) -> None:
        packet = {
            "task_id": "exploratory-method-test",
            "operation": "descriptive_statistics",
            "inputs": {"data": [1, 2, 3]},
            "quality_profile": {
                "decision_class": "exploratory",
                "method_ids": ["not-registered"],
            },
        }
        normalized, warnings, errors = compute_ticket._normalize_packet(packet, 21)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["quality_profile"]["method_ids"], [])
        self.assertTrue(any("not-registered" in item for item in warnings))

    def test_formal_unknown_method_fails_closed(self) -> None:
        packet = {
            "task_id": "formal-method-test-0001",
            "operation": "descriptive_statistics",
            "inputs": {"data": [1, 2, 3]},
            "quality_profile": {
                "decision_class": "formal",
                "method_ids": ["not-registered"],
            },
        }
        _, _, errors = compute_ticket._normalize_packet(packet, 22)
        self.assertTrue(any("unknown method IDs" in item for item in errors))
        self.assertTrue(any("upstream_refs" in item for item in errors))

    def test_retry_lineage_generates_immutable_revision_id(self) -> None:
        packet = {
            "task_id": "same-task-identifier",
            "retry_of": {"issue_number": 19},
            "operation": "descriptive_statistics",
            "inputs": {"data": [1, 2, 3]},
        }
        normalized, warnings, errors = compute_ticket._normalize_packet(packet, 23)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["task_id"], "same-task-identifier-r23")
        self.assertNotIn("retry_of", normalized)
        self.assertTrue(any("Issue #19" in item for item in warnings))

    def test_uncertain_inputs_generate_three_stage_plan(self) -> None:
        packet = {
            "assumptions": [
                {"name": "rent", "confidence": "low", "source_type": "proxy"},
                {"name": "living_budget", "confidence": "medium", "source_type": "proxy"},
            ]
        }
        plan = compute_ticket._analysis_chain_plan(packet)
        self.assertIsNotNone(plan)
        self.assertEqual(
            [row["operation"] for row in plan["sequence"]],
            ["scenario_compare", "sensitivity_analysis", "monte_carlo"],
        )
        self.assertFalse(plan["automatic_parallel_execution"])

    def test_exploratory_library_selection_warns_instead_of_failing(self) -> None:
        ticket = {
            "quality_profile": {
                "decision_class": "exploratory",
                "method_ids": ["not-registered"],
            }
        }
        result = library_runtime.resolve_library_selection(ticket)
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["methods"], [])
        self.assertTrue(any("not-registered" in item for item in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
