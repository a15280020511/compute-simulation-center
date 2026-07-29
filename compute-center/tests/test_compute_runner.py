from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compute_runner", ROOT / "compute_runner.py")
assert SPEC and SPEC.loader
compute_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute_runner)


class ComputeRunnerTests(unittest.TestCase):
    def run_ticket(self, ticket: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = compute_runner.run_ticket(ticket, output)
            saved = json.loads((output / "compute-result.json").read_text(encoding="utf-8"))
            self.assertEqual(result, saved)
            self.assertTrue((output / "compute-audit.json").is_file())
            self.assertTrue((output / "compute-summary.md").is_file())
            self.assertTrue((output / "artifact-manifest.json").is_file())
            return result

    def test_break_even(self) -> None:
        result = self.run_ticket(
            {
                "task_id": "compute-break-even-001",
                "operation": "break_even_analysis",
                "inputs": {
                    "fixed_cost": 100000,
                    "unit_price": 80,
                    "variable_cost": 50,
                },
                "assumptions": [
                    {
                        "name": "unit_price",
                        "value": 80,
                        "basis": "user supplied",
                        "confidence": "high",
                    }
                ],
            }
        )
        self.assertAlmostEqual(result["results"]["break_even_units"], 100000 / 30)
        self.assertEqual(result["execution"]["model_calls"], 0)
        self.assertFalse(result["execution"]["network_used"])

    def test_monte_carlo_is_reproducible(self) -> None:
        ticket = {
            "task_id": "compute-monte-carlo-001",
            "operation": "monte_carlo",
            "inputs": {
                "iterations": 5000,
                "seed": 20260727,
                "variables": [
                    {
                        "name": "demand",
                        "distribution": "triangular",
                        "minimum": 8000,
                        "mode": 10000,
                        "maximum": 14000,
                    },
                    {
                        "name": "margin",
                        "distribution": "normal",
                        "mean": 12,
                        "standard_deviation": 2,
                        "clip_minimum": 0,
                    },
                ],
                "model": {
                    "intercept": 0,
                    "coefficients": {"demand": 1, "margin": 1},
                },
                "threshold": 9000,
            },
        }
        first = self.run_ticket(ticket)
        second = self.run_ticket(ticket)
        self.assertEqual(first["results"], second["results"])
        self.assertEqual(first["input_sha256"], second["input_sha256"])
        self.assertEqual(first["result_sha256"], second["result_sha256"])

    def test_scenario_compare(self) -> None:
        result = self.run_ticket(
            {
                "task_id": "compute-scenarios-001",
                "operation": "scenario_compare",
                "inputs": {
                    "model": {
                        "intercept": 0,
                        "coefficients": {"benefit": 1, "cost": -1},
                    },
                    "scenarios": [
                        {"name": "A", "values": {"benefit": 100, "cost": 70}},
                        {"name": "B", "values": {"benefit": 90, "cost": 40}},
                    ],
                },
            }
        )
        self.assertEqual(result["results"]["best_scenario"], "B")

    def test_linear_optimization(self) -> None:
        result = self.run_ticket(
            {
                "task_id": "compute-optimization-001",
                "operation": "constrained_optimization",
                "inputs": {
                    "objective": [3, 2],
                    "maximize": True,
                    "variable_names": ["x", "y"],
                    "A_ub": [[1, 1], [1, 0], [0, 1]],
                    "b_ub": [4, 2, 3],
                    "bounds": [[0, None], [0, None]],
                },
            }
        )
        self.assertAlmostEqual(result["results"]["objective_value"], 10.0, places=6)

    def test_rejects_arbitrary_operation(self) -> None:
        with self.assertRaises(compute_runner.ComputeError):
            compute_runner.validate_ticket(
                {
                    "task_id": "compute-invalid-001",
                    "operation": "run_python",
                    "inputs": {},
                }
            )

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaises(compute_runner.ComputeError):
            compute_runner.run_ticket(
                {
                    "task_id": "compute-invalid-number-001",
                    "operation": "descriptive_statistics",
                    "inputs": {"data": [1, float("inf")]},
                },
                Path(tempfile.mkdtemp()),
            )


if __name__ == "__main__":
    unittest.main()
