#!/usr/bin/env python3
"""Authorize and validate independent [compute] Issue tickets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from operation_validation import validate_operation_inputs

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "compute-ticket.schema.json"
MAX_BODY_CHARS = 100_000
TRUSTED_STATE_PREFIXES = (
    "## COMPUTE_ACCEPTED",
    "## COMPUTE_COMPLETED",
    "## COMPUTE_FAILED",
    "## COMPUTE_REJECTED",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"), parse_constant=_reject_constant)
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)


def _write_output(name: str, value: Any) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    text = str(value).replace("\n", " ").replace("\r", " ")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={text}\n")


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _api_json(url: str) -> Any:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "independent-compute-center"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _trusted_comments(repo: str, issue_number: int) -> Iterable[str]:
    if not repo or not os.getenv("GITHUB_TOKEN"):
        return []
    rows = _api_json(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments?per_page=100"
    )
    if not isinstance(rows, list):
        return []
    comments = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        user = row.get("user") if isinstance(row.get("user"), Mapping) else {}
        if str(user.get("login") or "") != "github-actions[bot]":
            continue
        body = str(row.get("body") or "").strip()
        if body.startswith(TRUSTED_STATE_PREFIXES):
            comments.append(body)
    return comments


def _duplicate_reason(repo: str, current_issue: int, packet: Mapping[str, Any], fingerprint: str) -> str:
    if not repo or not os.getenv("GITHUB_TOKEN"):
        return ""
    task_id = str(packet.get("task_id") or "")
    for page in range(1, 6):
        rows = _api_json(
            f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100&page={page}"
        )
        if not isinstance(rows, list):
            break
        for row in rows:
            if not isinstance(row, Mapping) or row.get("pull_request"):
                continue
            number = int(row.get("number") or 0)
            if number == current_issue:
                continue
            if not str(row.get("title") or "").startswith("[compute]"):
                continue
            body = str(row.get("body") or "")
            try:
                prior = json.loads(body, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(prior, Mapping):
                continue
            same_id = str(prior.get("task_id") or "") == task_id
            same_fingerprint = _canonical_sha(prior) == fingerprint
            if same_id or same_fingerprint:
                reason = "task_id" if same_id else "ticket fingerprint"
                return f"duplicate {reason}; previously submitted in Issue #{number}"
        if len(rows) < 100:
            break
    return ""


def _status(
    *,
    accepted: bool,
    reason: str,
    packet: Mapping[str, Any] | None,
    issue_number: int,
    fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "accepted": accepted,
        "reason": reason,
        "issue_number": issue_number,
        "task_id": str((packet or {}).get("task_id") or ""),
        "operation": str((packet or {}).get("operation") or ""),
        "ticket_sha256": fingerprint,
        "analysis_owner": "web-gpt",
        "execution_owner": "github-compute-center",
        "model_calls": 0,
        "network_fetches": 0,
    }


def prepare(args: argparse.Namespace) -> int:
    event = json.loads(Path(args.event_path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    issue = event.get("issue") if isinstance(event.get("issue"), Mapping) else {}
    actor = str((event.get("sender") or issue.get("user") or {}).get("login") or "")
    owner = str(os.getenv("REPOSITORY_OWNER") or "")
    title = str(issue.get("title") or "")
    body = str(issue.get("body") or "")
    issue_number = int(issue.get("number") or 0)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    packet: Mapping[str, Any] | None = None
    fingerprint: str | None = None
    errors: list[str] = []
    if not title.startswith("[compute]"):
        errors.append("Issue title must start with [compute]")
    if not owner or actor != owner:
        errors.append("only the repository owner may submit compute tickets")
    if len(body) > MAX_BODY_CHARS:
        errors.append(f"Issue body exceeds {MAX_BODY_CHARS} characters")
    try:
        parsed = json.loads(body, parse_constant=_reject_constant)
        if isinstance(parsed, Mapping):
            packet = parsed
        else:
            errors.append("Issue body JSON root must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid JSON: {exc}")

    if packet is not None:
        validation_errors = sorted(
            VALIDATOR.iter_errors(packet),
            key=lambda item: list(item.absolute_path),
        )
        for error in validation_errors[:20]:
            path = ".".join(str(item) for item in error.absolute_path) or "$"
            errors.append(f"{path}: {error.message}")
        if not validation_errors:
            try:
                validate_operation_inputs(packet)
            except ValueError as exc:
                errors.append(str(exc))
        if not validation_errors and not errors:
            fingerprint = _canonical_sha(packet)
            current_comments = list(_trusted_comments(os.getenv("GITHUB_REPOSITORY", ""), issue_number))
            if any(body.startswith("## COMPUTE_COMPLETED") for body in current_comments):
                errors.append("this compute Issue already completed")
            elif any(body.startswith("## COMPUTE_ACCEPTED") for body in current_comments) and not any(
                body.startswith("## COMPUTE_FAILED") for body in current_comments
            ):
                errors.append("this compute Issue is already accepted or running")
            duplicate = _duplicate_reason(
                os.getenv("GITHUB_REPOSITORY", ""),
                issue_number,
                packet,
                fingerprint,
            )
            if duplicate:
                errors.append(duplicate)

    accepted = not errors and packet is not None and fingerprint is not None
    reason = "validated independent compute ticket" if accepted else "; ".join(errors)
    status = _status(
        accepted=accepted,
        reason=reason,
        packet=packet,
        issue_number=issue_number,
        fingerprint=fingerprint,
    )
    (root / "ticket-status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if packet is not None:
        (root / "ticket.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    for name in ("accepted", "reason", "task_id", "operation", "ticket_sha256"):
        value = status.get(name, "")
        _write_output(name, str(value).lower() if isinstance(value, bool) else value)
    return 0 if accepted else 2


def render(args: argparse.Namespace) -> int:
    root = Path(args.output_dir)
    status = json.loads((root / "ticket-status.json").read_text(encoding="utf-8"))
    phase = args.phase
    heading = {
        "accepted": "COMPUTE_ACCEPTED",
        "rejected": "COMPUTE_REJECTED",
    }[phase]
    print(f"## {heading}")
    print()
    print(f"- Task ID: `{status.get('task_id') or 'unknown'}`")
    print(f"- Operation: `{status.get('operation') or 'unknown'}`")
    print(f"- Accepted: `{str(bool(status.get('accepted'))).lower()}`")
    print(f"- Ticket SHA256: `{status.get('ticket_sha256') or 'none'}`")
    print("- Model calls: `0`")
    print("- External data fetches: `0`")
    print(f"- Run: `{args.run_url}`")
    if phase == "rejected":
        print(f"- Reason: `{status.get('reason') or 'unknown'}`")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--event-path", required=True)
    prepare_parser.add_argument("--output-dir", default="compute-artifacts")
    prepare_parser.set_defaults(func=prepare)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--phase", choices=["accepted", "rejected"], required=True)
    render_parser.add_argument("--output-dir", default="compute-artifacts")
    render_parser.add_argument("--run-url", default="")
    render_parser.set_defaults(func=render)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
