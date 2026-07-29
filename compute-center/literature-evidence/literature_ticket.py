#!/usr/bin/env python3
"""Control plane for fixed OpenAlex + Crossref literature-evidence tickets.

The ticket can only supply a research query, a bounded result count, and descriptive
comparability context. It cannot supply URLs, Python code, HTTP headers, or numeric
model parameters. Execution always uses the allowlisted adapter in
``literature_evidence.py`` and freezes the result before any later calibration task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from literature_evidence import LiteratureEvidenceError, build


class LiteratureTicketError(ValueError):
    """Raised when a ticket or frozen package violates the fixed contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _extract_json(body: str) -> dict[str, Any]:
    text = body.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LiteratureTicketError(f"issue body must be one JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LiteratureTicketError("issue body must be one JSON object")
    return value


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "literature-evidence-ticket.schema.json"


def _validate(ticket: dict[str, Any]) -> None:
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(ticket), key=lambda item: list(item.path))
    if errors:
        parts = []
        for error in errors[:10]:
            location = ".".join(str(item) for item in error.path) or "$"
            parts.append(f"{location}: {error.message}")
        raise LiteratureTicketError("; ".join(parts))
    query = str(ticket["query"]).strip()
    if "\x00" in query:
        raise LiteratureTicketError("query contains a null byte")
    lowered = query.lower()
    forbidden_fragments = ("javascript:", "file://", "ftp://", "169.254.169.254")
    if any(fragment in lowered for fragment in forbidden_fragments):
        raise LiteratureTicketError("query contains a forbidden URL or metadata target")


def _fingerprint(ticket: dict[str, Any]) -> str:
    normalized = {
        "query": " ".join(str(ticket["query"]).split()).casefold(),
        "research_context": ticket.get("research_context") or {},
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def prepare(event_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = False
    status: dict[str, Any]
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
        issue = event.get("issue") or {}
        body = issue.get("body") or ""
        ticket = _extract_json(body)
        _validate(ticket)
        ticket = {
            **ticket,
            "query": " ".join(str(ticket["query"]).split()),
            "per_page": int(ticket.get("per_page", 10)),
        }
        fingerprint = _fingerprint(ticket)
        frozen_ticket = {
            **ticket,
            "schema_version": "literature-evidence-ticket-v1",
            "issue_number": issue.get("number"),
            "issue_url": issue.get("html_url"),
            "prepared_at": _utc_now(),
            "semantic_fingerprint": fingerprint,
            "network_policy": "allowlisted-literature-only",
            "allowed_hosts": ["api.openalex.org", "api.crossref.org"],
            "numeric_dispatch_allowed": False,
            "automatic_parameter_promotion_allowed": False,
        }
        _write_json(output_dir / "ticket.json", frozen_ticket)
        _write_json(
            output_dir / "request-plan.json",
            {
                "schema_version": "literature-request-plan-v1",
                "task_id": ticket["task_id"],
                "semantic_fingerprint": fingerprint,
                "providers": ["OpenAlex", "Crossref"],
                "maximum_openalex_records": ticket["per_page"],
                "maximum_crossref_verifications": ticket["per_page"],
                "allowed_hosts": ["api.openalex.org", "api.crossref.org"],
                "arbitrary_url_allowed": False,
                "numeric_dispatch_allowed": False,
            },
        )
        accepted = True
        status = {
            "schema_version": "literature-ticket-status-v1",
            "status": "LITERATURE_TICKET_ACCEPTED",
            "task_id": ticket["task_id"],
            "semantic_fingerprint": fingerprint,
            "accepted": True,
            "prepared_at": _utc_now(),
        }
    except (OSError, json.JSONDecodeError, LiteratureTicketError) as exc:
        status = {
            "schema_version": "literature-ticket-status-v1",
            "status": "LITERATURE_TICKET_REJECTED",
            "accepted": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:4000],
            "prepared_at": _utc_now(),
        }
    _write_json(output_dir / "ticket-status.json", status)
    _set_output("accepted", "true" if accepted else "false")
    _set_output("task_id", str(status.get("task_id") or ""))
    _set_output("semantic_fingerprint", str(status.get("semantic_fingerprint") or ""))
    return 0 if accepted else 2


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        data = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "schema_version": "literature-evidence-artifact-manifest-v1",
        "created_at": _utc_now(),
        "files": rows,
    }


def execute(ticket_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        if ticket.get("network_policy") != "allowlisted-literature-only":
            raise LiteratureTicketError("ticket network policy is not allowlisted-literature-only")
        if ticket.get("numeric_dispatch_allowed") is not False:
            raise LiteratureTicketError("numeric dispatch must remain disabled")
        package = build(str(ticket["query"]), int(ticket.get("per_page", 10)))
        package.update(
            {
                "task_id": ticket["task_id"],
                "semantic_fingerprint": ticket["semantic_fingerprint"],
                "research_context": ticket.get("research_context") or {},
                "evidence_state": "frozen-candidate-evidence",
                "numeric_dispatch_allowed": False,
                "automatic_parameter_promotion_allowed": False,
                "required_next_checks": [
                    "study-design-screening",
                    "sample-geography-time-policy-comparability",
                    "effect-direction-and-unit-normalization",
                    "heterogeneity-check",
                    "prior-predictive-check",
                    "local-observation-calibration",
                    "posterior-predictive-check",
                ],
            }
        )
        stable_payload = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        package["frozen_package_sha256"] = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()
        _write_json(output_dir / "literature-evidence-package.json", package)
        audit = {
            "schema_version": "literature-evidence-audit-v1",
            "status": "LITERATURE_EVIDENCE_COMPLETED",
            "task_id": ticket["task_id"],
            "record_count": len(package.get("records") or []),
            "providers": ["OpenAlex", "Crossref"],
            "allowed_hosts": ["api.openalex.org", "api.crossref.org"],
            "arbitrary_url_used": False,
            "numeric_dispatch_used": False,
            "model_calls": 0,
            "completed_at": _utc_now(),
            "frozen_package_sha256": package["frozen_package_sha256"],
        }
        _write_json(output_dir / "literature-evidence-audit.json", audit)
        summary = "\n".join(
            [
                "# Literature Evidence Package",
                "",
                f"- Task ID: `{ticket['task_id']}`",
                f"- Records: `{audit['record_count']}`",
                "- Providers: `OpenAlex + Crossref`",
                "- Evidence state: `frozen-candidate-evidence`",
                "- Numeric dispatch: `disabled`",
                "- Automatic parameter promotion: `disabled`",
                f"- Package SHA-256: `{package['frozen_package_sha256']}`",
            ]
        )
        (output_dir / "literature-evidence-summary.md").write_text(summary + "\n", encoding="utf-8")
        _write_json(output_dir / "artifact-manifest.json", _artifact_manifest(output_dir))
        return 0
    except (OSError, KeyError, TypeError, ValueError, LiteratureEvidenceError, LiteratureTicketError) as exc:
        _write_json(
            output_dir / "literature-evidence-error.json",
            {
                "schema_version": "literature-evidence-error-v1",
                "status": "LITERATURE_EVIDENCE_FAILED",
                "error_type": type(exc).__name__,
                "message": str(exc)[:4000],
                "retryable": isinstance(exc, (OSError, LiteratureEvidenceError)),
                "failed_at": _utc_now(),
            },
        )
        _write_json(output_dir / "artifact-manifest.json", _artifact_manifest(output_dir))
        return 1


def render(phase: str, output_dir: Path, run_url: str, artifact_url: str | None) -> str:
    status_path = output_dir / "ticket-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if phase == "accepted":
        return "\n".join(
            [
                "## LITERATURE_TICKET_ACCEPTED",
                "",
                f"- Task ID: `{status.get('task_id')}`",
                f"- Semantic fingerprint: `{status.get('semantic_fingerprint')}`",
                "- Network: `OpenAlex + Crossref allowlist only`",
                "- Numeric dispatcher: `disabled`",
                f"- Run: {run_url}",
            ]
        )
    if phase == "rejected":
        return "\n".join(
            [
                "## LITERATURE_TICKET_REJECTED",
                "",
                f"- Error type: `{status.get('error_type') or 'validation'}`",
                f"- Message: `{status.get('message') or 'invalid ticket'}`",
                "- External requests: `0`",
                f"- Run: {run_url}",
            ]
        )
    if phase == "completed":
        audit = json.loads((output_dir / "literature-evidence-audit.json").read_text(encoding="utf-8"))
        return "\n".join(
            [
                "## LITERATURE_EVIDENCE_COMPLETED",
                "",
                f"- Task ID: `{audit['task_id']}`",
                f"- Records: `{audit['record_count']}`",
                "- Evidence state: `frozen-candidate-evidence`",
                "- Numeric dispatcher: `disabled`",
                f"- Package SHA-256: `{audit['frozen_package_sha256']}`",
                f"- Artifact: {artifact_url or 'unavailable'}",
                f"- Run: {run_url}",
            ]
        )
    error_path = output_dir / "literature-evidence-error.json"
    error = json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else {}
    return "\n".join(
        [
            "## LITERATURE_EVIDENCE_FAILED",
            "",
            f"- Error type: `{error.get('error_type') or 'unknown'}`",
            f"- Message: `{error.get('message') or 'execution failed'}`",
            f"- Retryable: `{str(bool(error.get('retryable'))).lower()}`",
            f"- Artifact: {artifact_url or 'unavailable'}",
            f"- Run: {run_url}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--event-path", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--ticket", type=Path, required=True)
    execute_parser.add_argument("--output-dir", type=Path, required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--phase", choices=["accepted", "rejected", "completed", "failed"], required=True)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--run-url", required=True)
    render_parser.add_argument("--artifact-url")

    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args.event_path, args.output_dir)
    if args.command == "execute":
        return execute(args.ticket, args.output_dir)
    print(render(args.phase, args.output_dir, args.run_url, args.artifact_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
