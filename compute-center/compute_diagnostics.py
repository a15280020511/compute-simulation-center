#!/usr/bin/env python3
"""Structured, redacted diagnostics for the independent compute center."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SAFE_ENV_KEYS = (
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_SHA",
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
    "ISSUE_NUMBER",
)
MAX_TRACEBACK_CHARS = 40_000
MANIFEST_STAGE = "DEFERRED_TO_DELIVERY_STAGE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _canonical_sha(value: Any) -> str | None:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(raw).hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {
            "artifact-manifest.json",
            "compute-diagnostics.json",
        }:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    return rows


def _run_identity() -> dict[str, str | None]:
    return {key.lower(): os.getenv(key) for key in SAFE_ENV_KEYS}


def _manifest_contract() -> dict[str, Any]:
    return {
        "owner": "delivery-stage",
        "file": "artifact-manifest.json",
        "diagnostics_self_hash_avoided": True,
        "verification_source": "workflow step outcome plus final manifest contents",
    }


def _error_code(exc: BaseException, stage: str) -> str:
    name = type(exc).__name__.upper()
    message = str(exc).upper()
    if message.startswith("PREFLIGHT_BLOCKED:"):
        return "COMPUTE_PREFLIGHT_BLOCKED"
    if name == "JSONDECODEERROR":
        return "COMPUTE_TICKET_JSON_INVALID"
    if "SCHEMA" in message or "IS A REQUIRED PROPERTY" in message:
        return "COMPUTE_TICKET_SCHEMA_INVALID"
    if name == "COMPUTEERROR":
        return "COMPUTE_REQUEST_INVALID"
    if isinstance(exc, FileNotFoundError):
        return "COMPUTE_INPUT_FILE_MISSING"
    if isinstance(exc, PermissionError):
        return "COMPUTE_FILESYSTEM_PERMISSION_DENIED"
    if isinstance(exc, TimeoutError):
        return "COMPUTE_TIMEOUT"
    if isinstance(exc, OSError):
        return "COMPUTE_IO_ERROR"
    return f"COMPUTE_{stage.upper()}_{name}"


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, (OSError, TimeoutError)) and not isinstance(
        exc, (FileNotFoundError, PermissionError)
    )


def write_success(
    output_dir: Path,
    *,
    ticket: Mapping[str, Any],
    result: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    operation = str(ticket.get("operation") or result.get("operation") or "")
    payload = {
        "schema_version": "compute-diagnostics-v2",
        "created_at": _utc_now(),
        "status": "PASS",
        "stage": "complete",
        "error_code": "NONE",
        "task_id": str(ticket.get("task_id") or ""),
        "operation": operation,
        "ticket_sha256": _canonical_sha(ticket),
        "result_sha256": result.get("result_sha256"),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "retryable": False,
        "run_identity": _run_identity(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "stage_status": {
            "load_ticket": "PASS",
            "validate_ticket": "PASS",
            "data_preflight": "PASS",
            "execute_operation": "PASS",
            "write_result": "PASS",
            "write_manifest": MANIFEST_STAGE,
        },
        "manifest_contract": _manifest_contract(),
        "security": {
            "secret_values_included": False,
            "environment_allowlist": [key.lower() for key in SAFE_ENV_KEYS],
            "ticket_content_embedded": False,
        },
        "artifact_inventory_before_delivery": _inventory(output_dir),
    }
    _write_json(output_dir / "compute-diagnostics.json", payload)
    return payload


def write_failure(
    output_dir: Path,
    *,
    exc: BaseException,
    stage: str,
    started_at: str,
    elapsed_seconds: float,
    ticket_path: Path | None = None,
    ticket: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    trace = trace[-MAX_TRACEBACK_CHARS:]
    ticket_mapping = ticket if isinstance(ticket, Mapping) else {}
    operation = str(ticket_mapping.get("operation") or "")
    task_id = str(ticket_mapping.get("task_id") or "")
    error_code = _error_code(exc, stage)
    error = {
        "schema_version": "compute-error-v2",
        "status": "error",
        "created_at": _utc_now(),
        "started_at": started_at,
        "stage": stage,
        "error_code": error_code,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": trace,
        "task_id": task_id,
        "operation": operation,
        "ticket_path": str(ticket_path) if ticket_path else None,
        "ticket_sha256": _canonical_sha(ticket_mapping) if ticket_mapping else None,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "model_calls": 0,
        "network_used": False,
        "retryable": _retryable(exc),
        "run_identity": _run_identity(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "security": {
            "secret_values_included": False,
            "environment_allowlist": [key.lower() for key in SAFE_ENV_KEYS],
            "ticket_content_embedded": False,
        },
    }
    _write_json(output_dir / "compute-error.json", error)

    if stage == "load_ticket":
        validate_state = preflight_state = execute_state = "NOT_REACHED"
    elif stage == "validate_ticket":
        validate_state, preflight_state, execute_state = "FAIL", "NOT_REACHED", "NOT_REACHED"
    elif stage == "data_preflight":
        validate_state, preflight_state, execute_state = "PASS", "FAIL", "NOT_REACHED"
    elif stage == "execute_operation":
        validate_state, preflight_state, execute_state = "PASS", "PASS", "FAIL"
    else:
        validate_state, preflight_state, execute_state = "PASS", "PASS", "PASS"

    diagnostics = {
        "schema_version": "compute-diagnostics-v2",
        "created_at": _utc_now(),
        "status": "FAIL",
        "primary_failure": {
            "code": error_code,
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": error["retryable"],
        },
        "stage_status": {
            "load_ticket": "FAIL" if stage == "load_ticket" else "PASS",
            "validate_ticket": validate_state,
            "data_preflight": preflight_state,
            "execute_operation": execute_state,
            "write_result": "NOT_REACHED" if stage != "write_manifest" else "PASS",
            "write_manifest": MANIFEST_STAGE,
        },
        "manifest_contract": _manifest_contract(),
        "task_id": task_id,
        "operation": operation,
        "ticket_sha256": error["ticket_sha256"],
        "elapsed_seconds": error["elapsed_seconds"],
        "run_identity": error["run_identity"],
        "runtime": error["runtime"],
        "traceback_file": "compute-error.json",
        "remediation_hints": _remediation(error_code, stage),
        "security": error["security"],
        "artifact_inventory_before_delivery": _inventory(output_dir),
    }
    _write_json(output_dir / "compute-diagnostics.json", diagnostics)
    return diagnostics


def _remediation(error_code: str, stage: str) -> list[str]:
    if error_code == "COMPUTE_PREFLIGHT_BLOCKED":
        return [
            "Read compute-preflight.json and resolve every blocking data issue.",
            "GPTs may obtain data through the API catalog, request user data, or create a new ticket with explicit approved assumptions.",
            "Do not retry the unchanged ticket.",
        ]
    if error_code == "COMPUTE_TICKET_JSON_INVALID":
        return ["Correct the Issue JSON body; do not retry unchanged input."]
    if error_code in {"COMPUTE_TICKET_SCHEMA_INVALID", "COMPUTE_REQUEST_INVALID"}:
        return [
            "Compare the ticket with compute-ticket.schema.json and the selected operation contract.",
            "Do not create a duplicate task; correct the input and reference the failed Issue.",
        ]
    if stage == "execute_operation":
        return [
            "Inspect compute-error.json traceback and compute-console.log.",
            "Check operation-specific bounds, dimensions, finite values, and dependency versions.",
        ]
    return [
        "Inspect the first failing stage, traceback, run identity, and artifact hashes."
    ]
