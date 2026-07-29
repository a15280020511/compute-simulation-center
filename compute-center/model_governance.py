#!/usr/bin/env python3
"""Machine-readable model, assumption, calibration, constraint and validation governance."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
MODEL_REGISTRY_PATH = HERE / "model-registry.json"
SCHEMA_PATHS = {
    "assumption_register": HERE / "assumption-register.schema.json",
    "mechanism_register": HERE / "mechanism-register.schema.json",
    "experiment_profile": HERE / "experiment-profile.schema.json",
    "credibility_profile": HERE / "credibility-profile.schema.json",
    "calibration_profile": HERE / "calibration-profile.schema.json",
    "constraint_profile": HERE / "constraint-profile.schema.json",
    "validation_profile": HERE / "validation-profile.schema.json",
}
ALLOWED_MATURITY = {"experimental", "controlled-preview", "production", "decision-grade", "retired"}


class GovernanceError(ValueError):
    pass


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(path: Path) -> Draft202012Validator:
    schema = _load_json(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_document(kind: str, value: Any) -> None:
    try:
        validator = _validator(SCHEMA_PATHS[kind])
    except KeyError as exc:
        raise GovernanceError(f"unknown governance document kind: {kind}") from exc
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise GovernanceError(f"{kind} {location}: {first.message}")


MODEL_REQUIRED_FIELDS = {
    "model_id", "operation", "mode", "version", "maturity",
    "engineering_maturity", "evidence_maturity", "risk_tier",
    "intended_use", "prohibited_use", "theoretical_basis",
    "required_variables", "parameter_definitions", "calibration_supported",
    "allowed_backends", "calibration_datasets", "validation_datasets",
    "benchmark_ids", "known_failure_conditions", "assurance_owner",
    "last_calibrated_at", "revalidation_trigger", "sunset_date",
}


def _merged_model_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("models")
    defaults = value.get("defaults")
    if not isinstance(rows, list) or not rows or not isinstance(defaults, Mapping):
        raise GovernanceError("model registry must contain defaults and models")
    merged_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise GovernanceError("model registry row must be an object")
        merged = dict(defaults)
        merged.update(raw)
        merged_rows.append(merged)
    return merged_rows


def _validate_model_row(row: Mapping[str, Any], index: int, seen: set[str]) -> None:
    missing = sorted(MODEL_REQUIRED_FIELDS - set(row))
    if missing:
        raise GovernanceError(f"model registry row {index} missing: {', '.join(missing)}")
    model_id = str(row["model_id"])
    if not model_id or model_id in seen:
        raise GovernanceError(f"duplicate or empty model_id: {model_id}")
    seen.add(model_id)
    for field in ("maturity", "engineering_maturity", "evidence_maturity"):
        if str(row[field]) not in ALLOWED_MATURITY:
            raise GovernanceError(f"invalid {field} for {model_id}: {row[field]}")
    if row["maturity"] != row["engineering_maturity"]:
        raise GovernanceError(
            f"legacy maturity must equal engineering_maturity for {model_id}"
        )
    calibration_supported = row["calibration_supported"]
    allowed_backends = row["allowed_backends"]
    if not isinstance(calibration_supported, bool) or not isinstance(allowed_backends, list):
        raise GovernanceError(f"invalid calibration metadata for {model_id}")
    if not calibration_supported and allowed_backends:
        raise GovernanceError(f"non-calibratable model exposes backends: {model_id}")
    if row["evidence_maturity"] in {"production", "decision-grade"} and (
        not row["benchmark_ids"] or not row["validation_datasets"]
    ):
        raise GovernanceError(
            f"evidence maturity lacks benchmarks or validation data: {model_id}"
        )
    if (
        row["evidence_maturity"] == "decision-grade"
        and calibration_supported
        and not row["last_calibrated_at"]
    ):
        raise GovernanceError(
            f"decision-grade calibratable model has no calibration date: {model_id}"
        )


def load_model_registry() -> dict[str, Any]:
    value = _load_json(MODEL_REGISTRY_PATH)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "compute-model-registry-v2"
    ):
        raise GovernanceError("invalid model registry schema_version")
    rows = _merged_model_rows(value)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _validate_model_row(row, index, seen)
    value["models"] = rows
    return value


def registered_model(operation: str, mode: str | None = None) -> dict[str, Any]:
    candidates = []
    for row in load_model_registry()["models"]:
        if row["operation"] != operation:
            continue
        if row["mode"] == mode:
            return dict(row)
        if row["mode"] == "*":
            candidates.append(row)
    if candidates:
        return dict(candidates[0])
    raise GovernanceError(f"no registered model for operation={operation!r}, mode={mode!r}")


def validate_ticket_governance(ticket: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(ticket.get("operation") or "")
    inputs = ticket.get("inputs") if isinstance(ticket.get("inputs"), Mapping) else {}
    mode = str(inputs.get("mode") or "") or None
    model = registered_model(operation, mode)
    if model["engineering_maturity"] == "retired":
        raise GovernanceError(f"registered model is retired: {model['model_id']}")
    document_pairs = (
        ("assumption_register", "assumption_register"),
        ("mechanism_register", "mechanism_register"),
        ("experiment_profile", "experiment_profile"),
        ("credibility_profile", "credibility_profile"),
        ("calibration_profile", "calibration_profile"),
        ("constraint_profile", "constraint_profile"),
        ("validation_profile", "validation_profile"),
    )
    for field, kind in document_pairs:
        if field in ticket:
            validate_document(kind, ticket[field])
    if "calibration_profile" in ticket and not model["calibration_supported"]:
        raise GovernanceError(f"model does not support calibration: {model['model_id']}")
    if "calibration_profile" in ticket:
        backend = str(ticket["calibration_profile"]["backend"])
        if backend not in model["allowed_backends"]:
            raise GovernanceError(f"backend {backend} is not allowed for {model['model_id']}")
    document_names = tuple(field for field, _ in document_pairs)
    return {
        "schema_version": "compute-governance-validation-v3",
        "model_id": model["model_id"],
        "operation": operation,
        "mode": mode,
        "maturity": model["engineering_maturity"],
        "engineering_maturity": model["engineering_maturity"],
        "registered_evidence_maturity": model["evidence_maturity"],
        "risk_tier": model["risk_tier"],
        "calibration_supported": model["calibration_supported"],
        "documents": {key: key in ticket for key in document_names},
    }


def lifecycle_status(model: Mapping[str, Any], *, today: date | None = None, triggered_events: set[str] | None = None) -> dict[str, Any]:
    today = today or datetime.now(timezone.utc).date()
    triggered_events = triggered_events or set()
    engineering = str(model.get("engineering_maturity") or model.get("maturity") or "")
    sunset_raw = model.get("sunset_date")
    expired = bool(sunset_raw and date.fromisoformat(str(sunset_raw)) < today)
    configured = set(str(item) for item in model.get("revalidation_trigger") or [])
    triggered = sorted(configured & triggered_events)
    status = "MODEL_RETIRED" if engineering == "retired" else "MODEL_SUSPENDED" if expired else "MODEL_RECALIBRATION_REQUIRED" if triggered else "MODEL_ACTIVE"
    return {"model_id": str(model.get("model_id") or ""), "status": status, "expired": expired, "triggered_revalidation_events": triggered}


def validate_registry_operation_coverage(operations: set[str]) -> dict[str, Any]:
    registry = load_model_registry()
    covered = {str(row["operation"]) for row in registry["models"] if row["engineering_maturity"] != "retired"}
    missing = sorted(operations - covered)
    unknown = sorted(covered - operations)
    return {"status": "PASS" if not missing else "FAIL", "missing_operations": missing, "unknown_operations": unknown, "covered_operation_count": len(covered & operations)}
