
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operation_validation import validate_operation_inputs  # noqa: E402


class OperationValidationTests(unittest.TestCase):
    def test_monte_carlo_rejects_object_variables_before_execution(self):
        ticket = {
            "operation": "monte_carlo",
            "inputs": {"seed": 1, "variables": {"demand": 1}, "model": {}},
        }
        with self.assertRaisesRegex(ValueError, "inputs.variables"):
            validate_operation_inputs(ticket)

    def test_finance_mode_is_allowlisted(self):
        validate_operation_inputs(
            {"operation": "finance_decision_analysis", "inputs": {"mode": "performance_metrics"}}
        )
        with self.assertRaises(ValueError):
            validate_operation_inputs(
                {"operation": "finance_decision_analysis", "inputs": {"mode": "guaranteed_profit"}}
            )


if __name__ == "__main__":
    unittest.main()
