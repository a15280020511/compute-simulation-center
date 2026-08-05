from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("certified_math_task", HERE / "certified_math_task.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CertifiedMathTests(unittest.TestCase):
    def test_schema_is_valid(self) -> None:
        schema = json.loads((HERE / "certified-math-ticket.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    def test_smt_sat_and_unsat(self) -> None:
        sat = MODULE.smt_integer_feasibility({
            "variables": ["x", "y"],
            "constraints": [
                {"coefficients": {"x": 1, "y": 1}, "operator": "==", "rhs": 5},
                {"coefficients": {"x": 1}, "operator": ">=", "rhs": 0},
                {"coefficients": {"y": 1}, "operator": ">=", "rhs": 0}
            ]
        })
        self.assertEqual(sat["status"], "SAT")
        self.assertEqual(set(sat["model"]), {"x", "y"})

        unsat = MODULE.smt_integer_feasibility({
            "variables": ["x"],
            "constraints": [
                {"coefficients": {"x": 1}, "operator": ">=", "rhs": 1},
                {"coefficients": {"x": 1}, "operator": "<=", "rhs": 0}
            ]
        })
        self.assertEqual(unsat["status"], "UNSAT")
        self.assertIsNone(unsat["model"])

    def test_exact_polynomial(self) -> None:
        result = MODULE.exact_integer_polynomial({"coefficients": [2, -3, 5], "x": 10})
        self.assertEqual(result["degree"], 2)
        self.assertEqual(result["exact_value"], "175")

    def test_certified_interval_contains_rational(self) -> None:
        result = MODULE.certified_rational_interval({"numerator": 1, "denominator": 3, "precision_digits": 60})
        self.assertEqual(result["exact_rational"], "1/3")
        self.assertTrue(result["contains_exact_rational"])
        self.assertIn("0.333", result["interval"])

    def test_ticket_validation_blocks_unknown_variables(self) -> None:
        ticket = {
            "task_id": "bad-variable",
            "operation": "smt-integer-feasibility",
            "inputs": {
                "variables": ["x"],
                "constraints": [{"coefficients": {"y": 1}, "operator": "==", "rhs": 0}]
            }
        }
        with self.assertRaises(ValueError):
            MODULE.validate_ticket(ticket)

    def test_end_to_end_receipt_and_manifest(self) -> None:
        ticket = {
            "task_id": "poly-e2e",
            "operation": "exact-integer-polynomial",
            "inputs": {"coefficients": [1, 0, -1], "x": 12}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_path = root / "ticket.json"
            output_dir = root / "out"
            ticket_path.write_text(json.dumps(ticket), encoding="utf-8")
            self.assertEqual(MODULE.execute(ticket_path, output_dir), 0)
            receipt = json.loads((output_dir / "certified-math-receipt.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "certified-math-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "COMPUTE_CERTIFIED_MATH_COMPLETED")
            self.assertEqual(receipt["network_calls"], 0)
            self.assertEqual(receipt["result"]["exact_value"], "143")
            self.assertGreaterEqual(len(manifest["files"]), 2)


if __name__ == "__main__":
    unittest.main()
