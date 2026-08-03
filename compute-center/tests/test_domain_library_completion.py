from __future__ import annotations

import json
import unittest
from pathlib import Path

from domain_library_runtime import (
    DomainLibraryError,
    compute_registered_baseline,
    compute_registered_factor,
    resolve_domain_library_selection,
    validate_material_envelope,
)
from library_runtime import resolve_library_selection

HERE = Path(__file__).resolve().parents[1]


class DomainLibraryCompletionTests(unittest.TestCase):
    def test_factor_and_baseline_registries_are_populated(self) -> None:
        factors = json.loads((HERE / "domain-factor-registry.json").read_text(encoding="utf-8"))
        baselines = json.loads((HERE / "baseline-registry.json").read_text(encoding="utf-8"))
        self.assertEqual(len(factors["factors"]), 20)
        self.assertGreaterEqual(len(baselines["baselines"]), 10)
        self.assertEqual(len({row["factor_id"] for row in factors["factors"]}), 20)

    def test_external_material_registries_are_explicitly_data_pending(self) -> None:
        expectations = {
            "domain-rule-snapshot-registry.json": "snapshots",
            "ontology-crosswalk-registry.json": "crosswalks",
            "regime-event-registry.json": "events",
            "outcome-feedback-registry.json": "records",
        }
        for name, field in expectations.items():
            document = json.loads((HERE / name).read_text(encoding="utf-8"))
            self.assertEqual(document[field], [])
            self.assertEqual(document["status"], "structure-complete-data-pending")

    def test_fixed_factor_implementations(self) -> None:
        ratio = compute_registered_factor(
            "commercial-conversion-rate", {"transactions": 25, "footfall": 100}
        )
        self.assertAlmostEqual(ratio["value"], 0.25)
        growth = compute_registered_factor(
            "commercial-footfall-growth", {"previous_footfall": 100, "current_footfall": 120}
        )
        self.assertAlmostEqual(growth["value"], 0.2)
        concentration = compute_registered_factor(
            "commercial-tenant-concentration", {"tenant_shares": [0.5, 0.3, 0.2]}
        )
        self.assertAlmostEqual(concentration["value"], 0.38)
        self.assertFalse(ratio["runtime_network_used"])
        self.assertFalse(ratio["arbitrary_code_used"])

    def test_fixed_baseline_implementations(self) -> None:
        mean = compute_registered_baseline("historical-mean", {"history": [1, 2, 3]})
        self.assertEqual(mean["value"], 2)
        seasonal = compute_registered_baseline(
            "seasonal-naive", {"history": [10, 11, 12, 13], "season_length": 2}
        )
        self.assertEqual(seasonal["value"], 12)
        weights = compute_registered_baseline("equal-weight", {"item_count": 4})
        self.assertEqual(weights["value"], [0.25, 0.25, 0.25, 0.25])

    def test_unknown_ids_fail_closed(self) -> None:
        with self.assertRaises(DomainLibraryError):
            compute_registered_factor("unknown-factor", {"x": 1})
        with self.assertRaises(DomainLibraryError):
            resolve_domain_library_selection({"domain_rule_snapshot_ids": ["missing"]})

    def test_material_envelope_validation(self) -> None:
        digest = "a" * 64
        envelope = {
            "transfer_id": "transfer-001",
            "material_type": "sample_snapshot",
            "source_center": "intelligence-center",
            "source_repository_reference": "private-evidence-reference",
            "version": "1.0.0",
            "created_at": "2026-08-04T00:00:00Z",
            "valid_from": "2026-08-04",
            "valid_to": None,
            "files": [{"path": "sample/data.parquet", "sha256": digest, "bytes": 10, "media_type": "application/vnd.apache.parquet"}],
            "manifest_sha256": digest,
            "license": "reviewed",
            "geographic_scope": ["CN"],
            "time_range": {"start": "2025-01-01", "end": "2025-12-31"},
            "contains_personal_data": False,
            "gpts_validation": {"status": "PASS"},
        }
        result = validate_material_envelope(envelope)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["runtime_network_used"])
        bad = dict(envelope)
        bad["source_center"] = "compute-center"
        with self.assertRaises(DomainLibraryError):
            validate_material_envelope(bad)

    def test_unified_library_runtime_resolves_extended_selection(self) -> None:
        ticket = {
            "quality_profile": {
                "decision_class": "formal",
                "strategy_id": "minimax-regret",
                "factor_ids": ["finance-book-to-market"],
                "baseline_ids": ["equal-weight"],
            }
        }
        report = resolve_library_selection(ticket)
        self.assertEqual(report["schema_version"], "compute-library-selection-v2")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["domain_libraries"]["factors"][0]["factor_id"], "finance-book-to-market")
        self.assertFalse(report["runtime_network_used"])

    def test_core_and_extension_library_registries_are_complete(self) -> None:
        core = json.loads((HERE / "institutional-library-registry.json").read_text(encoding="utf-8"))
        extension = json.loads((HERE / "governed-domain-library-registry.json").read_text(encoding="utf-8"))
        core_ids = {row["id"] for row in core["libraries"]}
        extension_ids = {row["id"] for row in extension["libraries"]}
        required_extensions = {
            "domain-factor-library",
            "baseline-library",
            "metric-threshold-library",
            "domain-rule-snapshot-library",
            "ontology-crosswalk-library",
            "regime-event-library",
            "outcome-feedback-library",
            "external-domain-material-contract",
        }
        self.assertEqual(len(core_ids), 16)
        self.assertEqual(extension_ids, required_extensions)
        self.assertFalse(core_ids & extension_ids)
        self.assertEqual(len(core_ids | extension_ids), 24)
        self.assertEqual(extension["policy"]["core_institutional_library_count"], 16)
        self.assertFalse(extension["policy"]["direct_center_connection_allowed"])
        for row in extension["libraries"]:
            self.assertTrue((HERE / row["authority"]).is_file())


if __name__ == "__main__":
    unittest.main()
