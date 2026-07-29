#!/usr/bin/env python3
"""Authorize and validate independent ``[compute]`` Issue tickets."""
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
MAX_DUPLICATE_PAGES = 5
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
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "independent-compute-center",
    }
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
    comments: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        user = row.get("user") if isinstance(row.get("user"), Mapping) else {}
        body = str(row.get("body") or "").strip()
        trusted = str(user.get("login") or "") == "github-actions[bot]"
        if trusted and body.startswith(TRUSTED_STATE_PREFIXES):
            comments.append(body)
    return comments


def _prior_ticket(row: Mapping[str, Any], current_issue: int) -> tuple[int, Mapping[str, Any]] | None:
    if row.get("pull_request"):
        return None
    number = int(row.get("number") or 0)
    if number == current_issue or not str(row.get("title") or "").startswith("[compute]"):
        return None
    try:
        parsed = json.loads(str(row.get("body") or ""), parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        return None
    return (number, parsed) if isinstance(parsed, Mapping) else None


def _duplicate_in_rows(
    rows: list[Any],
    *,
    current_issue: int,
    task_id: str,
    fingerprint: str,
) -> str:
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        prior = _prior_ticket(raw, current_issue)
        if prior is None:
            continue
        number, packet = prior
        same_id = str(packet.get("task_id") or "") == task_id
        same_fingerprint = _canonical_sha(packet) == fingerprint
        if same_id or same_fingerprint:
            reason = "task_id" if same_id else "ticket fingerprint"
            return f"duplicate {reason}; previously submitted in Issue #{number}"
    return ""


def _duplicate_reason(
    repo: str,
    current_issue: int,
    packet: Mapping[str, Any],
    fingerprint: str,
) -> str:
    if not repo or not os.getenv("GITHUB_TOKEN"):
        return ""
    task_id = str(packet.get("task_id") or "")
    for page in range(1, MAX_DUPLICATE_PAGES + 1):
        rows = _api_json(
            f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100&page={page}"
        )
        if not isinstance(rows, list):
            return ""
        duplicate = _duplicate_in_rows(
            rows,
            current_issue=current_issue,
            task_id=task_id,
            fingerprint=fingerprint,
        )
        if duplicate:
            return duplicate
        if len(rows) < 100:
            return ""
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


def _event_fields(event: Mapping[str, Any]) -> tuple[str, str, str, int]:
    issue = event.get("issue") if isinstance(event.get("issue"), Mapping) else {}
    sender = event.get("sender")
    actor_source = sender if isinstance(sender, Mapping) else issue.get("user")
    actor = str(actor_source.get("login") or "") if isinstance(actor_source, Mapping) else ""
    return (
        actor,
        str(issue.get("title") or ""),
        str(issue.get("body") or ""),
        int(issue.get("number") or 0),
    )


def _parse_packet(body: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    try:
        parsed = json.loads(body, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, [f"invalid JSON: {exc}"]
    if not isinstance(parsed, Mapping):
        return None, ["Issue body JSON root must be an object"]
    return parsed, []


def _ticket_validation_errors(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    validation_errors = sorted(
        VALIDATOR.iter_errors(packet),
        key=lambda item: list(item.absolute_path),
    )
    for error in validation_errors[:20]:
        path = ".".join(str(item) for item in error.absolute_path) or "$"
        errors.append(f"{path}: {error.message}")
    if validation_errors:
        return errors
    try:
        validate_operation_inputs(packet)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def _current_issue_errors(repo: str, issue_number: int) -> list[str]:
    comments = list(_trusted_comments(repo, issue_number))
    if any(body.startswith("## COMPUTE_COMPLETED") for body in comments):
        return ["this compute Issue already completed"]
    accepted = any(body.startswith("## COMPUTE_ACCEPTED") for body in comments)
    failed = any(body.startswith("## COMPUTE_FAILED") for body in comments)
    return ["this compute Issue is already accepted or running"] if accepted and not failed else []


def _authorization_errors(actor: str, owner: str, title: str, body: str) -> list[str]:
    errors: list[str] = []
    if not title.startswith("[compute]"):
        errors.append("Issue title must start with [compute]")
    if not owner or actor != owner:
        errors.append("only the repository owner may submit compute tickets")
    if len(body) > MAX_BODY_CHARS:
        errors.append(f"Issue body exceeds {MAX_BODY_CHARS} characters")
    return errors


def _persist_status(
    root: Path,
    status: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
) -> None:
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


def prepare(args: argparse.Namespace) -> int:
    event = json.loads(
        Path(args.event_path).read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(event, Mapping):
        raise ValueError("GitHub event root must be an object")
    actor, title, body, issue_number = _event_fields(event)
    owner = str(os.getenv("REPOSITORY_OWNER") or "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    errors = _authorization_errors(actor, owner, title, body)
    packet, parse_errors = _parse_packet(body)
    errors.extend(parse_errors)
    fingerprint: str | None = None
    if packet is not None:
        errors.extend(_ticket_validation_errors(packet))
    if packet is not None and not errors:
        fingerprint = _canonical_sha(packet)
        errors.extend(_current_issue_errors(repo, issue_number))
        duplicate = _duplicate_reason(repo, issue_number, packet, fingerprint)
        if duplicate:
            errors.append(duplicate)

    accepted = not errors and packet is not None and fingerprint is not None
    status = _status(
        accepted=accepted,
        reason="validated independent compute ticket" if accepted else "; ".join(errors),
        packet=packet,
        issue_number=issue_number,
        fingerprint=fingerprint,
    )
    _persist_status(root, status, packet)
    return 0 if accepted else 2


def render(args: argparse.Namespace) -> int:
    root = Path(args.output_dir)
    status = json.loads((root / "ticket-status.json").read_text(encoding="utf-8"))
    heading = {"accepted": "COMPUTE_ACCEPTED", "rejected": "COMPUTE_REJECTED"}[
        args.phase
    ]
    print(f"## {heading}")
    print()
    print(f"- Task ID: `{status.get('task_id') or 'unknown'}`")
    print(f"- Operation: `{status.get('operation') or 'unknown'}`")
    print(f"- Accepted: `{str(bool(status.get('accepted'))).lower()}`")
    print(f"- Ticket SHA256: `{status.get('ticket_sha256') or 'none'}`")
    print("- Model calls: `0`")
    print("- External data fetches: `0`")
    print(f"- Run: `{args.run_url}`")
    if args.phase == "rejected":
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
    render_parser.add_argument(
        "--phase", choices=["accepted", "rejected"], required=True
    )
    render_parser.add_argument("--output-dir", default="compute-artifacts")
    render_parser.add_argument("--run-url", default="")
    render_parser.set_defaults(func=render)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
