from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("compute_runner_full", ROOT / "compute_runner.py")
maintenance = load_module("maintenance_audit", ROOT / "maintenance_audit.py")


class FullComputeAcceptanceTests(unittest.TestCase):
    def execute(self, ticket: dict) -> tuple[dict, Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name)
        result = runner.run_ticket(ticket, output)
        return result, output

    def test_all_six_operations_are_registered(self) -> None:
        self.assertEqual(
            set(runner.OPERATIONS),
            {
                "monte_carlo",
                "sensitivity_analysis",
                "scenario_compare",
                "constrained_optimization",
                "break_even_analysis",
                "descriptive_statistics",
            },
        )

    def test_sensitivity_analysis(self) -> None:
        result, _ = self.execute(
            {
                "task_id": "full-sensitivity-001",
                "operation": "sensitivity_analysis",
                "inputs": {
                    "variables": [
                        {"name": "a", "low": 1, "base": 2, "high": 5},
                        {"name": "b", "low": 4, "base": 5, "high": 6},
                    ],
                    "model": {"intercept": 1, "coefficients": {"a": 3, "b": -1}},
                },
            }
        )
        self.assertEqual(result["results"]["baseline_score"], 2.0)
        self.assertEqual(result["results"]["ranking"][0]["variable"], "a")

    def test_descriptive_statistics(self) -> None:
        result, _ = self.execute(
            {
                "task_id": "full-statistics-001",
                "operation": "descriptive_statistics",
                "inputs": {"data": [1, 2, 3, 4, 5]},
            }
        )
        self.assertEqual(result["results"]["count"], 5)
        self.assertEqual(result["results"]["mean"], 3.0)
        self.assertEqual(result["results"]["median"], 3.0)

    def test_monte_carlo_all_distributions_and_max_iterations(self) -> None:
        ticket = {
            "task_id": "full-monte-carlo-100k",
            "operation": "monte_carlo",
            "inputs": {
                "iterations": 100000,
                "seed": 20260727,
                "variables": [
                    {"name": "constant", "distribution": "constant", "value": 2},
                    {"name": "uniform", "distribution": "uniform", "minimum": 0, "maximum": 1},
                    {"name": "triangular", "distribution": "triangular", "minimum": 0, "mode": 1, "maximum": 2},
                    {"name": "normal", "distribution": "normal", "mean": 3, "standard_deviation": 1, "clip_minimum": 0, "clip_maximum": 6},
                ],
                "model": {"intercept": 0, "coefficients": {"constant": 1, "uniform": 1, "triangular": 1, "normal": 1}},
                "threshold": 4,
            },
        }
        first, _ = self.execute(ticket)
        second, _ = self.execute(ticket)
        self.assertEqual(first["results"], second["results"])
        self.assertTrue(first["execution"]["reproducible"])
        self.assertEqual(first["results"]["iterations"], 100000)
        self.assertEqual(len(first["results"]["sensitivity"]), 4)

    def test_scenario_ranking(self) -> None:
        result, _ = self.execute(
            {
                "task_id": "full-scenario-001",
                "operation": "scenario_compare",
                "inputs": {
                    "model": {"intercept": 0, "coefficients": {"benefit": 1, "cost": -1}},
                    "scenarios": [
                        {"name": "A", "values": {"benefit": 10, "cost": 8}},
                        {"name": "B", "values": {"benefit": 9, "cost": 3}},
                    ],
                },
            }
        )
        self.assertEqual(result["results"]["best_scenario"], "B")
        self.assertEqual(result["results"]["ranking"][0]["rank"], 1)

    def test_constrained_optimization_and_infeasible_error(self) -> None:
        result, _ = self.execute(
            {
                "task_id": "full-optimal-001",
                "operation": "constrained_optimization",
                "inputs": {
                    "objective": [3, 2],
                    "maximize": True,
                    "variable_names": ["x", "y"],
                    "A_ub": [[1, 1]],
                    "b_ub": [4],
                    "bounds": [[0, 2], [0, 3]],
                },
            }
        )
        self.assertEqual(result["results"]["solver_status"], 0)
        with self.assertRaises(runner.ComputeError):
            runner.run_ticket(
                {
                    "task_id": "full-infeasible-001",
                    "operation": "constrained_optimization",
                    "inputs": {
                        "objective": [1],
                        "maximize": True,
                        "A_ub": [[1], [-1]],
                        "b_ub": [0, -1],
                        "bounds": [[None, None]],
                    },
                },
                Path(tempfile.mkdtemp()),
            )

    def test_break_even_and_invalid_margin(self) -> None:
        result, _ = self.execute(
            {
                "task_id": "full-break-even-001",
                "operation": "break_even_analysis",
                "inputs": {"fixed_cost": 1000, "unit_price": 20, "variable_cost": 12, "target_profit": 200},
            }
        )
        self.assertEqual(result["results"]["minimum_whole_units"], 150)
        with self.assertRaises(runner.ComputeError):
            runner.break_even_analysis({"fixed_cost": 1, "unit_price": 10, "variable_cost": 10})

    def test_security_and_hard_limits(self) -> None:
        with self.assertRaises(runner.ComputeError):
            runner.validate_ticket({"task_id": "full-invalid-op-001", "operation": "run_python", "inputs": {}})
        with self.assertRaises(runner.ComputeError):
            runner.monte_carlo(
                {
                    "iterations": 100001,
                    "seed": 1,
                    "variables": [{"name": "x", "distribution": "constant", "value": 1}],
                    "model": {"coefficients": {"x": 1}},
                }
            )
        with self.assertRaises(runner.ComputeError):
            runner.scenario_compare(
                {
                    "model": {"coefficients": {"x": 1}},
                    "scenarios": [{"name": str(index), "values": {"x": index}} for index in range(51)],
                }
            )

    def test_artifact_hashes_and_no_secrets(self) -> None:
        result, output = self.execute(
            {
                "task_id": "full-artifact-001",
                "operation": "descriptive_statistics",
                "inputs": {"data": [2, 4, 6]},
            }
        )
        audit = json.loads((output / "compute-audit.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["result_sha256"], result["result_sha256"])
        self.assertFalse(audit["secret_values_included"])
        self.assertTrue(any(row["path"] == "compute-result.json" for row in manifest["files"]))


class MaintenanceAcceptanceTests(unittest.TestCase):
    def test_configuration_is_complete(self) -> None:
        report = maintenance.validate_configuration(REPO)
        self.assertEqual(report["status"], "PASS", report)

    def test_update_classifier(self) -> None:
        self.assertEqual(maintenance.classify_update("1.2.3", "1.2.4"), "patch")
        self.assertEqual(maintenance.classify_update("1.2.3", "1.3.0"), "minor")
        self.assertEqual(maintenance.classify_update("1.2.3", "2.0.0"), "major")
        self.assertEqual(maintenance.classify_update("bad", "2.0.0"), "unknown")

    def test_cleanup_plan_is_safe_and_scoped(self) -> None:
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)
        artifacts = [
            {"id": 1, "name": "compute-ticket-new", "created_at": "2026-07-10T00:00:00Z", "expired": False},
            {"id": 2, "name": "compute-ticket-old", "created_at": "2026-06-01T00:00:00Z", "expired": False},
            {"id": 3, "name": "compute-center-validation-old", "created_at": "2026-07-01T00:00:00Z", "expired": False},
            {"id": 4, "name": "unrelated-old", "created_at": "2026-01-01T00:00:00Z", "expired": False},
            {"id": 5, "name": "compute-ticket-expired", "created_at": "2026-07-26T00:00:00Z", "expired": True},
        ]
        planned = maintenance.plan_artifact_cleanup(artifacts, now)
        self.assertEqual({row["id"] for row in planned}, {2, 3, 5})
        caches = [
            {"id": 10, "key": "new", "last_accessed_at": "2026-07-20T00:00:00Z"},
            {"id": 11, "key": "old", "last_accessed_at": "2026-06-20T00:00:00Z"},
        ]
        self.assertEqual([row["id"] for row in maintenance.plan_cache_cleanup(caches, now)], [11])


if __name__ == "__main__":
    unittest.main()
