from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "material_package_validation.py"
SPEC = importlib.util.spec_from_file_location("material_package_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class MaterialPackageValidationTests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        package = root / "package"
        payload = package / "payload/source-001/data.csv"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text("period,value\n2026-01,1\n", encoding="utf-8")
        file_row = {
            "path": "payload/source-001/data.csv",
            "sha256": file_sha(payload),
            "bytes": payload.stat().st_size,
            "media_type": "text/csv",
        }
        manifest = {
            "schema_version": "compute-material-package-manifest-v1",
            "package_id": "package-0001",
            "task_id": "task-000001",
            "material_type": "sample_snapshot",
            "version": "1.0.0",
            "created_at": "2026-08-04T00:00:00Z",
            "source_center": "intelligence-center",
            "source_repository_reference": "hf-dataset:James147258/cloudflare-intelligence-archive",
            "decision_use": "test",
            "source_records": [{
                "record_id": "source-001",
                "version": "1.0.0",
                "record_path": "external-reality/v1/records/source-001.json",
                "record_sha256": "a" * 64,
                "source_root": "external-reality/v1",
                "review_due_at": "2026-12-31",
            }],
            "files": [{
                **file_row,
                "source_path": "external-reality/v1/data/data.csv",
                "source_record_id": "source-001",
            }],
            "total_bytes": payload.stat().st_size,
            "contains_personal_data": False,
            "runtime_code_included": False,
            "selection_sha256": "b" * 64,
        }
        manifest_path = package / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha = file_sha(manifest_path)
        envelope = {
            "schema_version": "compute-external-domain-material-envelope-v2",
            "transfer_id": "package-0001",
            "task_id": "task-000001",
            "material_type": "sample_snapshot",
            "source_center": "intelligence-center",
            "source_repository_reference": "hf-dataset:James147258/cloudflare-intelligence-archive",
            "version": "1.0.0",
            "created_at": "2026-08-04T00:00:00Z",
            "valid_from": "2026-01-01",
            "valid_to": "2026-12-31",
            "files": [file_row],
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha,
            "license": {
                "name": "mixed-source-reviewed",
                "reviewed": True,
                "use_scope": ["compute-analysis"],
                "source_record_ids": ["source-001"],
            },
            "geographic_scope": ["CN"],
            "time_range": {"start": "2026-01-01", "end": "2026-01-31"},
            "contains_personal_data": False,
            "gpts_validation": {
                "status": "PASS",
                "validator": "gpts-usage-center",
                "validated_at": "2026-08-04T00:01:00Z",
                "task_id": "task-000001",
                "selection_sha256": "b" * 64,
            },
        }
        envelope_path = package / "envelope.json"
        envelope_path.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        content_sha = canonical_sha({
            "manifest_sha256": manifest_sha,
            "envelope_sha256": file_sha(envelope_path),
            "files": [file_row],
        })
        relay = {
            "schema_version": "gpts-compute-material-relay-attestation-v1",
            "status": "PASS",
            "validator": "gpts-usage-center",
            "validated_at": "2026-08-04T00:02:00Z",
            "task_id": "task-000001",
            "transfer_id": "package-0001",
            "manifest_sha256": manifest_sha,
            "envelope_sha256": file_sha(envelope_path),
            "content_sha256": content_sha,
        }
        (package / "gpts-relay-attestation.json").write_text(
            json.dumps(relay, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return package

    def refresh_relay(self, package: Path) -> None:
        envelope_path = package / "envelope.json"
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        relay = json.loads((package / "gpts-relay-attestation.json").read_text(encoding="utf-8"))
        relay["manifest_sha256"] = envelope["manifest_sha256"]
        relay["envelope_sha256"] = file_sha(envelope_path)
        relay["content_sha256"] = canonical_sha({
            "manifest_sha256": envelope["manifest_sha256"],
            "envelope_sha256": relay["envelope_sha256"],
            "files": envelope["files"],
        })
        (package / "gpts-relay-attestation.json").write_text(
            json.dumps(relay, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_complete_package_passes_with_actual_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            result = MODULE.validate_material_package(package, as_of=date(2026, 8, 4))
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["actual_file_hashes_verified"])
            self.assertTrue(result["actual_file_sizes_verified"])
            self.assertEqual(result["gpts_selection_validation"], "PASS")
            self.assertEqual(result["gpts_relay_attestation"], "PASS")
            self.assertFalse(result["runtime_network_used"])

    def test_missing_relay_attestation_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            (package / "gpts-relay-attestation.json").unlink()
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "relay-attestation"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))

    def test_payload_tampering_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            (package / "payload/source-001/data.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.MaterialPackageValidationError, "size mismatch|SHA256 mismatch"
            ):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))

    def test_manifest_tampering_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            manifest["decision_use"] = "changed"
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "manifest SHA256"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))

    def test_expired_package_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            envelope = json.loads((package / "envelope.json").read_text(encoding="utf-8"))
            envelope["valid_to"] = "2026-08-03"
            (package / "envelope.json").write_text(json.dumps(envelope), encoding="utf-8")
            self.refresh_relay(package)
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "expired"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))

    def test_selection_and_relay_mismatches_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            envelope = json.loads((package / "envelope.json").read_text(encoding="utf-8"))
            envelope["gpts_validation"]["task_id"] = "other-task"
            (package / "envelope.json").write_text(json.dumps(envelope), encoding="utf-8")
            self.refresh_relay(package)
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "selection task_id"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            relay = json.loads((package / "gpts-relay-attestation.json").read_text(encoding="utf-8"))
            relay["content_sha256"] = "0" * 64
            (package / "gpts-relay-attestation.json").write_text(json.dumps(relay), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "content_sha256"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))

    def test_license_source_binding_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            envelope = json.loads((package / "envelope.json").read_text(encoding="utf-8"))
            envelope["license"]["source_record_ids"] = ["other-record"]
            (package / "envelope.json").write_text(json.dumps(envelope), encoding="utf-8")
            self.refresh_relay(package)
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "license/source"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))

    def test_undeclared_file_and_secret_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            (package / "hidden.txt").write_text("hidden", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "undeclared"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))
        with tempfile.TemporaryDirectory() as tmp:
            package = self.fixture(Path(tmp))
            payload = package / "payload/source-001/data.csv"
            fake_secret = "hf_" + "abcdefghijklmnopqrstuvwxyz123456"
            payload.write_text(f"token,{fake_secret}\n", encoding="utf-8")
            envelope = json.loads((package / "envelope.json").read_text(encoding="utf-8"))
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            size = payload.stat().st_size
            digest = file_sha(payload)
            envelope["files"][0]["bytes"] = size
            envelope["files"][0]["sha256"] = digest
            manifest["files"][0]["bytes"] = size
            manifest["files"][0]["sha256"] = digest
            manifest["total_bytes"] = size
            (package / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            envelope["manifest_sha256"] = file_sha(package / "manifest.json")
            (package / "envelope.json").write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.refresh_relay(package)
            with self.assertRaisesRegex(MODULE.MaterialPackageValidationError, "credential-like"):
                MODULE.validate_material_package(package, as_of=date(2026, 8, 4))


if __name__ == "__main__":
    unittest.main()
