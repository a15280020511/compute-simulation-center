#!/usr/bin/env python3
"""Authorize, normalize, validate and de-duplicate independent ``[compute]`` tickets.

The admission layer remains deterministic. It supports explicit retry lineage,
softly removes unknown method IDs only for exploratory tickets, requires hashed
upstream evidence for formal/high-stakes tickets, and preserves the global
single-task execution slot.
"""
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
METHOD_REGISTRY_PATH = HERE / "method-registry.json"
MAX_BODY_CHARS = 100_000
MAX_DUPLICATE_PAGES = 5
MAX_ACTIVE_TASK_PAGES = 5
TRUSTED_STATE_PREFIXES = (
    "## COMPUTE_ACCEPTED",
    "## COMPUTE_COMPLETED",
    "## COMPUTE_FAILED",
    "## COMPUTE_REJECTED",
)
TERMINAL_STATE_PREFIXES = (
    "## COMPUTE_COMPLETED",
    "## COMPUTE_FAILED",
    "## COMPUTE_REJECTED",
)
ACTIVE_RUN_STATUSES = ("queued", "in_progress")


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


def _prior_ticket(
    row: Mapping[str, Any], current_issue: int
) -> tuple[int, Mapping[str, Any]] | None:
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


def _retry_issue(packet: Mapping[str, Any]) -> int | None:
    retry = packet.get("retry_of")
    if retry is None:
        return None
    if isinstance(retry, int) and not isinstance(retry, bool) and retry > 0:
        return retry
    if isinstance(retry, Mapping):
        value = retry.get("issue_number")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return -1


def _duplicate_in_rows(
    rows: list[Any],
    *,
    current_issue: int,
    task_id: str,
    fingerprint: str,
    retry_issue: int | None = None,
) -> str:
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        prior = _prior_ticket(raw, current_issue)
        if prior is None:
            continue
        number, packet = prior
        comments = list(
            _trusted_comments(os.getenv("GITHUB_REPOSITORY", ""), number)
        )
        abandoned = str(raw.get("state") or "") == "closed" and not comments
        if abandoned:
            continue
        same_id = str(packet.get("task_id") or "") == task_id
        same_fingerprint = _canonical_sha(packet) == fingerprint
        if same_id or same_fingerprint:
            if retry_issue == number:
                terminal_failure = any(
                    body.startswith(("## COMPUTE_FAILED", "## COMPUTE_REJECTED"))
                    for body in comments
                )
                if terminal_failure:
                    continue
                return f"retry_of Issue #{number} is not a failed or rejected terminal task"
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
    retry_issue = _retry_issue(packet)
    if retry_issue == -1:
        return "retry_of must be a positive Issue number or an object containing issue_number"
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
            retry_issue=retry_issue,
        )
        if duplicate:
            return duplicate
        if len(rows) < 100:
            return ""
    return ""


def _active_issue_number(
    rows: list[Any],
    *,
    current_issue: int,
    repo: str,
) -> int | None:
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("pull_request"):
            continue
        number = int(raw.get("number") or 0)
        title = str(raw.get("title") or "")
        if number <= 0 or number == current_issue or not title.startswith("[compute]"):
            continue
        comments = list(_trusted_comments(repo, number))
        accepted = any(body.startswith("## COMPUTE_ACCEPTED") for body in comments)
        terminal = any(body.startswith(TERMINAL_STATE_PREFIXES) for body in comments)
        if accepted and not terminal:
            return number
    return None


def _active_task_reason(repo: str, current_issue: int) -> str:
    if not repo or not os.getenv("GITHUB_TOKEN"):
        return ""
    for page in range(1, MAX_ACTIVE_TASK_PAGES + 1):
        rows = _api_json(
            f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100&page={page}"
        )
        if not isinstance(rows, list):
            return ""
        number = _active_issue_number(rows, current_issue=current_issue, repo=repo)
        if number is not None:
            return f"another compute task is already accepted and active in Issue #{number}"
        if len(rows) < 100:
            return ""
    return ""


def _workflow_run_ids(payload: Any) -> set[int]:
    if not isinstance(payload, Mapping):
        return set()
    rows = payload.get("workflow_runs")
    if not isinstance(rows, list):
        return set()
    run_ids: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        event = str(row.get("event") or "issues")
        status = str(row.get("status") or "")
        run_id = int(row.get("id") or 0)
        if event == "issues" and status in ACTIVE_RUN_STATUSES and run_id > 0:
            run_ids.add(run_id)
    return run_ids


def _active_workflow_run_reason(repo: str) -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    current_raw = os.getenv("GITHUB_RUN_ID")
    if not repo or not token or not current_raw:
        return ""
    try:
        current_run = int(current_raw)
    except ValueError:
        return "unable to verify global compute slot: invalid GITHUB_RUN_ID"
    active = {current_run}
    try:
        for status in ACTIVE_RUN_STATUSES:
            payload = _api_json(
                f"https://api.github.com/repos/{repo}/actions/workflows/compute-ticket.yml/runs"
                f"?status={status}&per_page=100"
            )
            active.update(_workflow_run_ids(payload))
    except Exception as exc:
        return f"unable to verify global compute slot: {type(exc).__name__}"
    winner = min(active)
    if winner != current_run:
        return f"another compute workflow run #{winner} owns the global execution slot"
    return ""


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


def _registered_method_ids() -> set[str]:
    try:
        data = json.loads(METHOD_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    rows = data.get("installed_method_packs") if isinstance(data, Mapping) else []
    return {
        str(row.get("id"))
        for row in rows
        if isinstance(row, Mapping) and row.get("id")
    }


def _normalize_packet(
    packet: Mapping[str, Any], issue_number: int
) -> tuple[dict[str, Any], list[str], list[str]]:
    normalized = json.loads(json.dumps(packet, ensure_ascii=False, allow_nan=False))
    warnings: list[str] = []
    errors: list[str] = []

    retry_issue = _retry_issue(normalized)
    if retry_issue == -1:
        errors.append(
            "retry_of must be a positive Issue number or an object containing issue_number"
        )
    elif retry_issue is not None:
        base_id = str(normalized.get("task_id") or "")
        suffix = f"-r{issue_number}"
        normalized["task_id"] = (
            base_id[: 128 - len(suffix)] + suffix
            if base_id
            else f"compute-retry-{issue_number}"
        )
        warnings.append(
            f"retry lineage declared from Issue #{retry_issue}; task_id revised for immutable audit"
        )
        normalized.pop("retry_of", None)

    quality = (
        normalized.get("quality_profile")
        if isinstance(normalized.get("quality_profile"), Mapping)
        else {}
    )
    decision_class = str(quality.get("decision_class") or "exploratory")
    method_ids = (
        quality.get("method_ids") if isinstance(quality.get("method_ids"), list) else []
    )
    if method_ids:
        registered = _registered_method_ids()
        unknown = sorted({str(item) for item in method_ids} - registered)
        if unknown and decision_class == "exploratory":
            kept = [str(item) for item in method_ids if str(item) in registered]
            normalized.setdefault("quality_profile", {})["method_ids"] = kept
            warnings.append(
                "unknown exploratory method IDs removed: " + ", ".join(unknown)
            )
        elif unknown:
            errors.append(
                "unknown method IDs for formal/high-stakes ticket: "
                + ", ".join(unknown)
            )

    upstream: list[Any] = []
    pipeline = normalized.get("pipeline")
    if isinstance(pipeline, Mapping) and isinstance(pipeline.get("upstream_refs"), list):
        upstream = pipeline["upstream_refs"]
    if decision_class in {"formal", "high_stakes"} and not upstream:
        errors.append(
            "formal/high-stakes tickets require pipeline.upstream_refs with hashed evidence"
        )
    elif decision_class == "exploratory" and not upstream:
        warnings.append(
            "no hashed upstream evidence supplied; release must remain exploratory/blocked"
        )
    return normalized, warnings, errors


def _analysis_chain_plan(packet: Mapping[str, Any]) -> dict[str, Any] | None:
    assumptions = (
        packet.get("assumptions") if isinstance(packet.get("assumptions"), list) else []
    )
    variables: list[Any] = []
    data_context = packet.get("data_context")
    if isinstance(data_context, Mapping) and isinstance(data_context.get("variables"), list):
        variables = data_context["variables"]
    uncertain: list[str] = []
    for row in [*assumptions, *variables]:
        if not isinstance(row, Mapping):
            continue
        confidence = str(row.get("confidence") or "")
        source_type = str(row.get("source_type") or "")
        if confidence in {"low", "medium"} or source_type in {
            "proxy",
            "gpts_assumption",
            "expert_hypothesis",
        }:
            uncertain.append(str(row.get("name") or "unknown"))
    if len(set(uncertain)) < 2:
        return None
    return {
        "schema_version": "compute-analysis-chain-plan-v1",
        "status": "REQUIRED_BEFORE_UNCONDITIONAL_RELEASE",
        "trigger_variables": sorted(set(uncertain)),
        "sequence": [
            {
                "operation": "scenario_compare",
                "purpose": "compare explicit structural cases",
            },
            {
                "operation": "sensitivity_analysis",
                "purpose": "rank range-driven conclusion instability",
            },
            {
                "operation": "monte_carlo",
                "purpose": "estimate outcome distribution and threshold risk",
            },
        ],
        "execution_policy": "sequential-single-task-tickets",
        "automatic_parallel_execution": False,
        "model_calls": 0,
        "network_fetches": 0,
    }


def _ticket_validation_errors(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    validation_errors = sorted(
        VALIDATOR.iter_errors(packet), key=lambda item: list(item.absolute_path)
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


def _authorization_errors(
    actor: str, owner: str, title: str, body: str
) -> list[str]:
    errors: list[str] = []
    if not title.startswith("[compute]"):
        errors.append("Issue title must start with [compute]")
    if not owner or actor != owner:
        errors.append("only the repository owner may submit compute tickets")
    if len(body) > MAX_BODY_CHARS:
        errors.append(f"Issue body exceeds {MAX_BODY_CHARS} characters")
    return errors


def _status(
    *,
    accepted: bool,
    reason: str,
    packet: Mapping[str, Any] | None,
    issue_number: int,
    fingerprint: str | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": 2,
        "accepted": accepted,
        "reason": reason,
        "warnings": list(warnings or []),
        "issue_number": issue_number,
        "task_id": str((packet or {}).get("task_id") or ""),
        "operation": str((packet or {}).get("operation") or ""),
        "ticket_sha256": fingerprint,
        "analysis_owner": "web-gpt",
        "execution_owner": "github-compute-center",
        "model_calls": 0,
        "network_fetches": 0,
    }


def _persist_status(
    root: Path,
    status: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
) -> None:
    (root / "ticket-status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
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
    warnings: list[str] = []
    original_packet = packet
    if packet is not None:
        packet, warnings, normalization_errors = _normalize_packet(packet, issue_number)
        errors.extend(normalization_errors)
        errors.extend(_ticket_validation_errors(packet))
    fingerprint: str | None = None
    if packet is not None and not errors:
        fingerprint = _canonical_sha(packet)
        errors.extend(_current_issue_errors(repo, issue_number))
        workflow_slot = _active_workflow_run_reason(repo)
        if workflow_slot:
            errors.append(workflow_slot)
        active = _active_task_reason(repo, issue_number)
        if active:
            errors.append(active)
        duplicate_packet = (
            original_packet if isinstance(original_packet, Mapping) else packet
        )
        duplicate = _duplicate_reason(
            repo,
            issue_number,
            duplicate_packet,
            _canonical_sha(duplicate_packet),
        )
        if duplicate:
            errors.append(duplicate)

    accepted = not errors and packet is not None and fingerprint is not None
    status = _status(
        accepted=accepted,
        reason="validated independent compute ticket" if accepted else "; ".join(errors),
        packet=packet,
        issue_number=issue_number,
        fingerprint=fingerprint,
        warnings=warnings,
    )
    _persist_status(root, status, packet)
    if warnings:
        (root / "ticket-normalization.json").write_text(
            json.dumps(
                {
                    "schema_version": "compute-ticket-normalization-v1",
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if packet is not None:
        plan = _analysis_chain_plan(packet)
        if plan is not None:
            (root / "analysis-chain-plan.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return 0 if accepted else 2


def render(args: argparse.Namespace) -> int:
    root = Path(args.output_dir)
    status = json.loads(
        (root / "ticket-status.json").read_text(encoding="utf-8")
    )
    heading = {
        "accepted": "COMPUTE_ACCEPTED",
        "rejected": "COMPUTE_REJECTED",
    }[args.phase]
    print(f"## {heading}")
    print()
    print(f"- Task ID: `{status.get('task_id') or 'unknown'}`")
    print(f"- Operation: `{status.get('operation') or 'unknown'}`")
    print(f"- Accepted: `{str(bool(status.get('accepted'))).lower()}`")
    print(f"- Ticket SHA256: `{status.get('ticket_sha256') or 'none'}`")
    print("- Model calls: `0`")
    print("- External data fetches: `0`")
    print(f"- Run: `{args.run_url}`")
    for warning in status.get("warnings") or []:
        print(f"- Warning: `{warning}`")
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


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
