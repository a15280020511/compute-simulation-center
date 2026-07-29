from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostics = _load("compute_diagnostics", ROOT / "compute_diagnostics.py")


class ComputeDiagnosticsTests(unittest.TestCase):
    def test_failure_contains_traceback_stage_and_safe_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_RUN_ID": "123",
                    "GITHUB_SHA": "abc",
                    "OPENROUTER_API_KEY": "must-not-leak",
                },
                clear=False,
            ):
                try:
                    raise ValueError("synthetic failure")
                except ValueError as exc:
                    result = diagnostics.write_failure(
                        root,
                        exc=exc,
                        stage="execute_operation",
                        started_at="2026-07-28T00:00:00+00:00",
                        elapsed_seconds=1.25,
                        ticket_path=Path("ticket.json"),
                        ticket={
                            "task_id": "compute-diagnostic-001",
                            "operation": "descriptive_statistics",
                        },
                    )

            error = json.loads((root / "compute-error.json").read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "compute-diagnostics-v2")
            self.assertEqual(error["schema_version"], "compute-error-v2")
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(error["stage"], "execute_operation")
            self.assertIn("ValueError: synthetic failure", error["traceback"])
            self.assertEqual(error["run_identity"]["github_run_id"], "123")
            self.assertEqual(
                result["stage_status"]["write_manifest"],
                "DEFERRED_TO_DELIVERY_STAGE",
            )
            self.assertEqual(result["manifest_contract"]["owner"], "delivery-stage")
            serialized = json.dumps(error, ensure_ascii=False)
            self.assertNotIn("must-not-leak", serialized)
            self.assertFalse(error["security"]["secret_values_included"])

    def test_success_writes_complete_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = diagnostics.write_success(
                root,
                ticket={
                    "task_id": "compute-diagnostic-002",
                    "operation": "break_even_analysis",
                    "inputs": {},
                },
                result={
                    "operation": "break_even_analysis",
                    "result_sha256": "result-sha",
                },
                elapsed_seconds=0.5,
            )
            self.assertEqual(result["schema_version"], "compute-diagnostics-v2")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["error_code"], "NONE")
            self.assertEqual(result["result_sha256"], "result-sha")
            self.assertIsNotNone(result["ticket_sha256"])
            self.assertEqual(
                result["stage_status"]["write_manifest"],
                "DEFERRED_TO_DELIVERY_STAGE",
            )
            self.assertTrue(result["manifest_contract"]["diagnostics_self_hash_avoided"])
            self.assertFalse(result["security"]["ticket_content_embedded"])


if __name__ == "__main__":
    unittest.main()
