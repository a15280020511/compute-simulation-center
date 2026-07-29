from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compute_dispatch_transfer_test", ROOT / "compute_dispatch.py"
)
assert SPEC and SPEC.loader
compute_dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute_dispatch)


class ComputeTransferPreflightTests(unittest.TestCase):
    def test_preflight_and_quality_are_added_to_transfer_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = {
                "schema_version": "compute-result-v1",
                "task_id": "transfer-test-0001",
                "status": "success",
                "operation": "descriptive_statistics",
                "result_sha256": "a" * 64,
                "results": {"count": 3},
            }
            preflight = {
                "status": "DATA_DEGRADED",
                "execution_allowed": True,
                "requires_user_approval": False,
                "policy": {"enforcement": "strict"},
                "issues": [
                    {
                        "severity": "warning",
                        "code": "DATA_CONTEXT_NOT_DECLARED",
                        "message": "metadata absent",
                        "blocking": False,
                        "variable": None,
                        "remediation": "add metadata",
                    }
                ],
                "recommended_operations": ["sensitivity_analysis"],
                "data_summary": {"declared_variable_count": 0},
            }
            quality_report = {
                "decision_class": "formal",
                "release_status": "DECISION_CONDITIONAL",
                "decision_grade": False,
                "constraints": {
                    "formal_decision_use_allowed": False,
                    "must_collect_more_feedback": False,
                    "must_recalibrate_before_reuse": False,
                },
            }
            (output_dir / "compute-result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            (output_dir / "compute-audit.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            (output_dir / "compute-summary.md").write_text(
                "# COMPUTE_COMPLETED\n", encoding="utf-8"
            )

            compute_dispatch._attach_governance_to_transfer(
                output_dir, result, preflight, quality_report
            )

            transfer = json.loads(
                (output_dir / "compute-result.json").read_text(encoding="utf-8")
            )
            audit = json.loads(
                (output_dir / "compute-audit.json").read_text(encoding="utf-8")
            )
            summary = (output_dir / "compute-summary.md").read_text(encoding="utf-8")
            self.assertEqual(transfer["preflight"]["status"], "DATA_DEGRADED")
            self.assertEqual(transfer["preflight"]["blocking_issue_count"], 0)
            self.assertRegex(transfer["preflight"]["preflight_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(transfer["quality"]["release_status"], "DECISION_CONDITIONAL")
            self.assertFalse(transfer["quality"]["formal_decision_use_allowed"])
            self.assertRegex(transfer["package_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(audit["preflight_status"], "DATA_DEGRADED")
            self.assertEqual(audit["quality_release_status"], "DECISION_CONDITIONAL")
            self.assertEqual(audit["package_sha256"], transfer["package_sha256"])
            self.assertIn("Preflight status", summary)
            self.assertIn("Decision release", summary)


if __name__ == "__main__":
    unittest.main()
