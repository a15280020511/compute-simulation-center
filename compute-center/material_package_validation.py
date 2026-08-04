#!/usr/bin/env python3
"""Deeply validate complete Intelligence-to-Compute material packages offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "complete-material-package-contract.json"
ALLOWED_CONTROL_FILES = {"envelope.json", "manifest.json", "receipt.json"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
FORBIDDEN_SUFFIXES = {
    ".py", ".pyc", ".pyo", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".js", ".mjs", ".cjs", ".exe", ".dll", ".so", ".dylib", ".jar", ".war",
    ".whl", ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z",
}


class MaterialPackageValidationError(ValueError):
    """Raised when an immutable material package is unsafe or inconsistent."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise MaterialPackageValidationError(f"unsafe {label}: {text!r}")
    if any(part in {"", ".git"} for part in path.parts):
        raise MaterialPackageValidationError(f"unsafe {label}: {text!r}")
    return path


def _parse_date(value: Any, label: str, *, nullable: bool = False) -> date | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise MaterialPackageValidationError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MaterialPackageValidationError(f"invalid {label}: {value}") from exc


def _parse_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise MaterialPackageValidationError(f"{label} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaterialPackageValidationError(f"invalid {label}: {value}") from exc
    if parsed.tzinfo is None:
        raise MaterialPackageValidationError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterialPackageValidationError(f"{label} must be an object")
    return value


def _scan_secrets(path: Path, *, limit: int = 5_000_000) -> None:
    if path.stat().st_size > limit:
        return
    data = path.read_bytes()
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise MaterialPackageValidationError(f"credential-like content detected: {path.name}")


def validate_material_package(package_root: Path, *, as_of: date | None = None) -> dict[str, Any]:
    root = package_root.resolve()
    if not root.is_dir():
        raise MaterialPackageValidationError("package root is missing")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise MaterialPackageValidationError("symbolic links are forbidden")
    envelope_path = root / "envelope.json"
    manifest_path = root / "manifest.json"
    if not envelope_path.is_file() or not manifest_path.is_file():
        raise MaterialPackageValidationError("envelope.json and manifest.json are required")
    contract = _require_object(_load_json(CONTRACT_PATH), "contract")
    envelope = _require_object(_load_json(envelope_path), "envelope")
    manifest = _require_object(_load_json(manifest_path), "manifest")
    today = as_of or datetime.now(timezone.utc).date()

    required_envelope = set(contract["required_envelope_fields"])
    missing = sorted(required_envelope - set(envelope))
    if missing:
        raise MaterialPackageValidationError(f"missing envelope fields: {', '.join(missing)}")
    if envelope.get("schema_version") != "compute-external-domain-material-envelope-v2":
        raise MaterialPackageValidationError("unsupported envelope schema")
    if manifest.get("schema_version") != "compute-material-package-manifest-v1":
        raise MaterialPackageValidationError("unsupported manifest schema")
    if envelope.get("source_center") != "intelligence-center":
        raise MaterialPackageValidationError("source_center must be intelligence-center")
    reference = str(envelope.get("source_repository_reference") or "")
    if not reference.startswith("hf-dataset:") or "://" in reference:
        raise MaterialPackageValidationError("source repository reference must be an opaque HF dataset reference")
    if envelope.get("contains_personal_data") is not False:
        raise MaterialPackageValidationError("personal data is forbidden")
    material_type = str(envelope.get("material_type") or "")
    if material_type not in contract["accepted_material_types"]:
        raise MaterialPackageValidationError("unsupported material_type")
    if envelope.get("manifest_path") != "manifest.json":
        raise MaterialPackageValidationError("manifest_path must be manifest.json")
    if _file_sha(manifest_path) != envelope.get("manifest_sha256"):
        raise MaterialPackageValidationError("manifest SHA256 mismatch")

    for field in (
        "package_id", "task_id", "material_type", "version", "created_at",
        "source_center", "source_repository_reference",
    ):
        manifest_field = "package_id" if field == "package_id" else field
        envelope_field = "transfer_id" if field == "package_id" else field
        if manifest.get(manifest_field) != envelope.get(envelope_field):
            raise MaterialPackageValidationError(f"manifest/envelope mismatch: {field}")
    created_at = _parse_datetime(envelope["created_at"], "created_at")
    valid_from = _parse_date(envelope.get("valid_from"), "valid_from", nullable=True)
    valid_to = _parse_date(envelope.get("valid_to"), "valid_to", nullable=True)
    if valid_from and valid_to and valid_to < valid_from:
        raise MaterialPackageValidationError("valid_to precedes valid_from")
    if valid_to and valid_to < today:
        raise MaterialPackageValidationError("material package is expired")
    if created_at.date() > today:
        raise MaterialPackageValidationError("created_at is in the future")

    license_block = _require_object(envelope.get("license"), "license")
    for field in contract["required_license_fields"]:
        if field not in license_block:
            raise MaterialPackageValidationError(f"license field missing: {field}")
    if license_block.get("reviewed") is not True:
        raise MaterialPackageValidationError("license must be reviewed")
    use_scope = license_block.get("use_scope")
    if not isinstance(use_scope, list) or "compute-analysis" not in use_scope:
        raise MaterialPackageValidationError("license excludes compute-analysis")

    approval = _require_object(envelope.get("gpts_validation"), "gpts_validation")
    for field in contract["required_gpts_validation_fields"]:
        if field not in approval:
            raise MaterialPackageValidationError(f"GPTs validation field missing: {field}")
    if approval.get("status") != "PASS" or approval.get("validator") != "gpts-usage-center":
        raise MaterialPackageValidationError("GPTs approval is not valid")
    if approval.get("task_id") != envelope.get("task_id"):
        raise MaterialPackageValidationError("GPTs task_id mismatch")
    if approval.get("approved_manifest_sha256") != envelope.get("manifest_sha256"):
        raise MaterialPackageValidationError("GPTs approved manifest SHA mismatch")
    validated_at = _parse_datetime(approval.get("validated_at"), "gpts_validation.validated_at")
    if validated_at < created_at:
        raise MaterialPackageValidationError("GPTs validation predates package creation")
    if validated_at.date() > today:
        raise MaterialPackageValidationError("GPTs validation is in the future")

    envelope_files = envelope.get("files")
    manifest_files = manifest.get("files")
    if not isinstance(envelope_files, list) or not envelope_files:
        raise MaterialPackageValidationError("envelope files must be non-empty")
    if not isinstance(manifest_files, list) or not manifest_files:
        raise MaterialPackageValidationError("manifest files must be non-empty")
    manifest_core = [
        {key: row.get(key) for key in contract["file_fields"]}
        for row in manifest_files if isinstance(row, Mapping)
    ]
    if manifest_core != envelope_files:
        raise MaterialPackageValidationError("manifest and envelope file tables differ")

    allowed_media = contract["allowed_media_types"]
    declared_paths: set[str] = set()
    total_bytes = 0
    for row in envelope_files:
        if not isinstance(row, Mapping):
            raise MaterialPackageValidationError("file entry must be an object")
        if set(row) != set(contract["file_fields"]):
            raise MaterialPackageValidationError("file entry fields are incomplete or excessive")
        relative = _safe_relative(row["path"], "payload path")
        if relative.parts[0] != "payload":
            raise MaterialPackageValidationError("payload files must be under payload/")
        text = str(relative)
        if text in declared_paths:
            raise MaterialPackageValidationError("duplicate payload path")
        declared_paths.add(text)
        path = root / Path(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise MaterialPackageValidationError(f"payload file missing or unsafe: {text}")
        suffix = relative.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            raise MaterialPackageValidationError(f"executable or archive payload forbidden: {text}")
        media_type = str(row.get("media_type") or "")
        extensions = allowed_media.get(media_type)
        if not isinstance(extensions, list) or suffix not in extensions:
            raise MaterialPackageValidationError(f"media type/extension mismatch: {text}")
        actual_size = path.stat().st_size
        actual_sha = _file_sha(path)
        if actual_size != row.get("bytes"):
            raise MaterialPackageValidationError(f"payload size mismatch: {text}")
        if actual_sha != row.get("sha256"):
            raise MaterialPackageValidationError(f"payload SHA256 mismatch: {text}")
        if actual_size > int(contract["limits"]["max_file_bytes"]):
            raise MaterialPackageValidationError(f"payload file exceeds size limit: {text}")
        total_bytes += actual_size
        if total_bytes > int(contract["limits"]["max_total_bytes"]):
            raise MaterialPackageValidationError("package exceeds total size limit")
        if media_type in {
            "application/json", "application/x-ndjson", "text/csv", "application/geo+json"
        }:
            _scan_secrets(path)

    actual_relative_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_relative_files - declared_paths - ALLOWED_CONTROL_FILES)
    if unexpected:
        raise MaterialPackageValidationError(f"undeclared package files: {unexpected[:10]}")
    for control in (envelope_path, manifest_path):
        _scan_secrets(control)

    if manifest.get("total_bytes") != total_bytes:
        raise MaterialPackageValidationError("manifest total_bytes mismatch")
    if manifest.get("contains_personal_data") is not False:
        raise MaterialPackageValidationError("manifest personal-data flag is invalid")
    if manifest.get("runtime_code_included") is not False:
        raise MaterialPackageValidationError("runtime code is forbidden")
    source_records = manifest.get("source_records")
    if not isinstance(source_records, list) or not source_records:
        raise MaterialPackageValidationError("source_records are required")
    for row in source_records:
        if not isinstance(row, Mapping):
            raise MaterialPackageValidationError("source record entry must be an object")
        review_due = _parse_date(row.get("review_due_at"), "source review_due_at")
        if review_due < today:
            raise MaterialPackageValidationError("source record review is overdue")
        digest = str(row.get("record_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MaterialPackageValidationError("source record SHA256 is invalid")

    package_sha = _canonical_sha({
        "manifest_sha256": envelope["manifest_sha256"],
        "envelope_sha256": _file_sha(envelope_path),
        "files": envelope_files,
    })
    result = {
        "schema_version": "compute-material-package-validation-receipt-v1",
        "status": "PASS",
        "package_id": envelope["transfer_id"],
        "task_id": envelope["task_id"],
        "material_type": material_type,
        "file_count": len(envelope_files),
        "total_bytes": total_bytes,
        "manifest_sha256": envelope["manifest_sha256"],
        "envelope_sha256": _file_sha(envelope_path),
        "package_sha256": package_sha,
        "license_reviewed": True,
        "gpts_validation": "PASS",
        "actual_file_hashes_verified": True,
        "actual_file_sizes_verified": True,
        "validity_verified_as_of": today.isoformat(),
        "credential_scan_passed": True,
        "runtime_network_used": False,
        "database_credentials_used": False,
        "direct_center_connection": False,
        "model_calls": 0,
    }
    result["validation_sha256"] = _canonical_sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    receipt = validate_material_package(Path(args.package_root), as_of=as_of)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
