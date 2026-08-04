import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import sagemath_operations as sage


class SageMathTests(unittest.TestCase):
    def setUp(self):
        os.environ["SAGEMATH_FIXTURE_MODE"] = "1"

    def tearDown(self):
        os.environ.pop("SAGEMATH_FIXTURE_MODE", None)

    def test_all_modes_are_bounded(self):
        samples = [
            {"mode": "simplify", "variables": ["x"], "expression": "(x^2-1)/(x-1)"},
            {"mode": "solve", "variables": ["x"], "variable": "x", "expression": "x^2-4"},
            {"mode": "differentiate", "variables": ["x"], "variable": "x", "expression": "sin(x)*exp(x)", "order": 2},
            {"mode": "integrate", "variables": ["x"], "variable": "x", "expression": "x^2", "lower": 0, "upper": 1},
            {"mode": "matrix_analysis", "matrix": [[1, 2], [3, 4]]},
            {"mode": "number_theory", "action": "factor", "values": [360]},
        ]
        for sample in samples:
            self.assertEqual(sage.symbolic_mathematics(sample)["mode"], sample["mode"])

    def test_expression_injection_is_rejected(self):
        for expression in ["__import__(x)", "x;system(x)", "x.__class__", "[x]", "x**2"]:
            with self.assertRaises(Exception):
                sage.symbolic_mathematics({"mode": "simplify", "variables": ["x"], "expression": expression})

    def test_unknown_names_are_rejected(self):
        with self.assertRaises(Exception):
            sage.symbolic_mathematics({"mode": "simplify", "variables": ["x"], "expression": "mystery(x)"})

    def test_matrix_and_integer_limits(self):
        with self.assertRaises(Exception):
            sage.symbolic_mathematics({"mode": "matrix_analysis", "matrix": [[1, 2], [3]]})
        with self.assertRaises(Exception):
            sage.symbolic_mathematics({"mode": "number_theory", "action": "factor", "values": [10**19]})

    def test_exact_digest_and_offline_policy(self):
        runtime = json.loads((HERE / "sagemath-runtime.json").read_text(encoding="utf-8"))
        self.assertRegex(runtime["image"], r"^sagemath/sagemath@sha256:[0-9a-f]{64}$")
        self.assertEqual(runtime["network_policy"], "none")
        self.assertTrue(runtime["read_only_root"])
        self.assertTrue(runtime["no_new_privileges"])


if __name__ == "__main__":
    unittest.main()
