#!/usr/bin/env python3
"""Issue-ticket runtime for the dedicated offline SageMath compute entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sagemath_operations import symbolic_mathematics  # noqa: E402

SCHEMA_PATH = HERE / "sagemath-ticket.schema.json"
MAX_EVENT_BYTES = 300_000
MAX_TICKET_BYTES = 100_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_issue_body(body: str) -> dict[str, Any]:
    text = str(body or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("issue body must contain one JSON object")
    return value


def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_ticket(ticket: dict[str, Any]) -> None:
    errors = sorted(validator().iter_errors(ticket), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(f"schema validation failed at {location}: {error.message}")
    symbolic_mathematics_validate_only(ticket["inputs"])


def symbolic_mathematics_validate_only(inputs: dict[str, Any]) -> None:
    import sagemath_operations as sage

    sage.validate_inputs(inputs)


def prepare(event_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    accepted = False
    failure: dict[str, str] | None = None
    ticket: dict[str, Any] | None = None
    try:
        if event_path.stat().st_size > MAX_EVENT_BYTES:
            raise ValueError("event payload too large")
        event = json.loads(event_path.read_text(encoding="utf-8"))
        issue = event.get("issue")
        if not isinstance(issue, dict):
            raise ValueError("event does not contain an issue")
        ticket = parse_issue_body(str(issue.get("body") or ""))
        raw = canonical_bytes(ticket)
        if len(raw) > MAX_TICKET_BYTES:
            raise ValueError("ticket payload too large")
        validate_ticket(ticket)
        write_json(output_dir / "ticket.json", ticket)
        accepted = True
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    receipt = {
        "schema_version": "compute-sagemath-admission-v1",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "accepted": accepted,
        "task_id": ticket.get("task_id") if isinstance(ticket, dict) else None,
        "ticket_sha256": sha256(ticket) if isinstance(ticket, dict) else None,
        "started_at": started,
        "completed_at": utc_now(),
        "failure": failure,
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_allowed": False,
    }
    write_json(output_dir / "admission.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if accepted else 2


def execute(ticket_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    ticket: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    status = "COMPUTE_SAGEMATH_FAILED"
    try:
        if ticket_path.stat().st_size > MAX_TICKET_BYTES:
            raise ValueError("ticket payload too large")
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        if not isinstance(ticket, dict):
            raise ValueError("ticket root must be an object")
        validate_ticket(ticket)
        operation_result = symbolic_mathematics(ticket["inputs"])
        result = {
            "schema_version": "compute-sagemath-result-v1",
            "status": "COMPUTE_SAGEMATH_COMPLETED",
            "task_id": ticket["task_id"],
            "provider": ticket["provider"],
            "operation": ticket["operation"],
            "objective": ticket["objective"],
            "quality_profile": ticket["quality_profile"],
            "result": operation_result,
            "runtime_network_used": False,
            "external_data_fetches": 0,
            "model_calls": 0,
            "arbitrary_code_allowed": False,
            "ticket_sha256": sha256(ticket),
        }
        result["result_sha256"] = sha256(result)
        write_json(output_dir / "compute-result.json", result)
        status = "COMPUTE_SAGEMATH_COMPLETED"
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    diagnostics = {
        "schema_version": "compute-sagemath-diagnostics-v1",
        "status": status,
        "task_id": ticket.get("task_id") if isinstance(ticket, dict) else None,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "runtime_network_used": False,
        "external_data_fetches": 0,
        "model_calls": 0,
        "arbitrary_code_allowed": False,
        "failure": failure,
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    manifest = {
        "schema_version": "compute-sagemath-manifest-v1",
        "status": status,
        "files": files,
        "secret_values_exposed": False,
        "runtime_network_used": False,
        "model_calls": 0,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(diagnostics, ensure_ascii=False))
    return 0 if status == "COMPUTE_SAGEMATH_COMPLETED" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--event-path", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--ticket", required=True)
    execute_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(Path(args.event_path), Path(args.output_dir))
    return execute(Path(args.ticket), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
