from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance_catalog import validate_catalogs


class GovernanceCatalogTests(unittest.TestCase):
    def test_all_governance_catalogs_are_consistent(self) -> None:
        report = validate_catalogs()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["operation_count"], 29)
        self.assertEqual(report["managed_mode_count"], 124)
        self.assertEqual(report["installed_method_pack_count"], 19)
        self.assertEqual(report["benchmark_category_count"], 5)
        self.assertEqual(report["institutional_library_count"], 16)
        self.assertGreaterEqual(report["strategy_count"], 8)
        self.assertEqual(report["sample_entry_count"], 0)
        self.assertEqual(report["assumption_library_entry_count"], 0)
        self.assertGreaterEqual(report["distribution_count"], 9)
        self.assertGreaterEqual(report["scenario_type_count"], 9)
        self.assertGreaterEqual(report["experiment_design_count"], 8)
        self.assertEqual(report["credibility_factor_count"], 12)
        self.assertFalse(report["live_external_database_access"])
        self.assertFalse(report["database_server_required"])
        self.assertFalse(report["conditional_backends_installed"])
        self.assertFalse(report["unverified_domain_truth_prepopulation_allowed"])
        self.assertFalse(report["single_weighted_credibility_score_allowed"])

    def test_performance_ledger_accepts_append_only_record(self) -> None:
        schema = json.loads((ROOT / "model-performance-ledger.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        record = {
            "record_id": "shadow-result-20260729-001",
            "record_type": "shadow_result",
            "model_id": "crisis_early_warning-registered-v1",
            "model_version": "1.0.0",
            "operation": "crisis_early_warning",
            "mode": "warning_performance_evaluation",
            "task_id": "shadow-task-20260729-001",
            "created_at": "2026-07-29T00:00:00Z",
            "source_snapshot_sha256": "0" * 64,
            "prediction": {"probability": 0.7},
            "outcome": None,
            "metrics": {},
            "decision_status": "NOT_APPLICABLE",
            "append_only_previous_record_sha256": None,
            "record_sha256": "1" * 64,
        }
        self.assertEqual(list(validator.iter_errors(record)), [])

    def test_shadow_manifest_does_not_fake_real_feedback(self) -> None:
        manifest = json.loads((ROOT / "benchmarks" / "shadow" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["registered_shadow_programs"], [])
        self.assertFalse(manifest["policy"]["affects_live_decision"])


if __name__ == "__main__":
    unittest.main()
