import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import sagemath_ticket_runtime as runtime


class SageMathTicketRuntimeTests(unittest.TestCase):
    def ticket(self):
        return {
            "task_id": "sagemath-ticket-test-20260804",
            "provider": "compute-sagemath",
            "operation": "symbolic_mathematics",
            "objective": "Validate a bounded symbolic simplification.",
            "inputs": {
                "mode": "simplify",
                "variables": ["x"],
                "expression": "(x^2-1)/(x-1)"
            },
            "data_policy": {
                "classification": "public-or-user-supplied-nonpersonal",
                "contains_personal_data": False
            },
            "quality_profile": {
                "decision_class": "exploratory"
            },
            "acceptance": {
                "timeout_seconds": 90
            }
        }

    def test_prepare_and_execute_fixture(self):
        os.environ["SAGEMATH_FIXTURE_MODE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                event = root / "event.json"
                output = root / "output"
                event.write_text(json.dumps({"issue": {"body": json.dumps(self.ticket())}}), encoding="utf-8")
                self.assertEqual(runtime.prepare(event, output), 0)
                admission = json.loads((output / "admission.json").read_text(encoding="utf-8"))
                self.assertTrue(admission["accepted"])
                self.assertEqual(runtime.execute(output / "ticket.json", output), 0)
                result = json.loads((output / "compute-result.json").read_text(encoding="utf-8"))
                diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "COMPUTE_SAGEMATH_COMPLETED")
                self.assertEqual(diagnostics["status"], "COMPUTE_SAGEMATH_COMPLETED")
                self.assertFalse(result["runtime_network_used"])
                self.assertEqual(result["model_calls"], 0)
        finally:
            os.environ.pop("SAGEMATH_FIXTURE_MODE", None)

    def test_rejects_arbitrary_code(self):
        ticket = self.ticket()
        ticket["inputs"]["expression"] = "__import__(x)"
        with self.assertRaises(Exception):
            runtime.validate_ticket(ticket)

    def test_rejects_personal_data_and_unknown_fields(self):
        ticket = self.ticket()
        ticket["data_policy"]["contains_personal_data"] = True
        with self.assertRaises(Exception):
            runtime.validate_ticket(ticket)
        ticket = self.ticket()
        ticket["unexpected"] = True
        with self.assertRaises(Exception):
            runtime.validate_ticket(ticket)


if __name__ == "__main__":
    unittest.main()
