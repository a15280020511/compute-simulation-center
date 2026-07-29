#!/usr/bin/env python3
"""Institutional assumption-library resolution, dependency checks and risk treatment."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
LIBRARY_PATH = HERE / "assumption-library.json"
LIBRARY_SCHEMA_PATH = HERE / "assumption-library.schema.json"
REGISTER_SCHEMA_PATH = HERE / "assumption-register.schema.json"
REQUIRED_LIBRARY_FIELDS = {
    "assumption_id",
    "version",
    "status",
    "type",
    "statement",
    "source_type",
    "basis",
    "confidence",
    "calibration_status",
    "uncertainty_type",
    "criticality",
    "evidence_strength",
    "owner",
    "intended_use",
    "content_sha256",
}
WEAK_SOURCE_TYPES = {
    "proxy",
    "gpts_assumption",
    "expert_elicitation",
    "user_assumption",
}


class AssumptionGovernanceError(ValueError):
    pass


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate(schema_path: Path, value: Any, label: str) -> None:
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: list(item.absolute_path),
    )
    if not errors:
        return
    first = errors[0]
    where = ".".join(str(item) for item in first.absolute_path) or "$"
    raise AssumptionGovernanceError(f"{label} {where}: {first.message}")


def _validate_library_row(row: Mapping[str, Any], seen: set[str]) -> None:
    missing = REQUIRED_LIBRARY_FIELDS - set(row)
    if missing:
        raise AssumptionGovernanceError(
            f"library assumption missing fields: {sorted(missing)}"
        )
    key = f"{row['assumption_id']}@{row['version']}"
    if key in seen:
        raise AssumptionGovernanceError(f"duplicate library assumption: {key}")
    seen.add(key)
    expected = _sha({name: value for name, value in row.items() if name != "content_sha256"})
    if row["content_sha256"] != expected:
        raise AssumptionGovernanceError(f"library assumption hash mismatch: {key}")


def load_library() -> dict[str, Any]:
    library = _load(LIBRARY_PATH)
    _validate(LIBRARY_SCHEMA_PATH, library, "assumption_library")
    rows = library.get("assumptions") or []
    _validate(REGISTER_SCHEMA_PATH, rows, "assumption_library.assumptions")
    seen: set[str] = set()
    for row in rows:
        _validate_library_row(row, seen)
    return library


def _risk_score(row: Mapping[str, Any]) -> int:
    confidence = {"high": 1, "medium": 2, "low": 3}.get(str(row.get("confidence")), 3)
    criticality = {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(
        str(row.get("criticality")), 2
    )
    evidence = {"strong": 1, "moderate": 2, "weak": 3, "none": 4}.get(
        str(row.get("evidence_strength")), 3
    )
    rank = row.get("sensitivity_rank")
    if rank == 1:
        sensitivity = 4
    elif isinstance(rank, int) and rank <= 3:
        sensitivity = 3
    elif isinstance(rank, int) and rank <= 10:
        sensitivity = 2
    else:
        sensitivity = 1
    return round(100 * (confidence + criticality + evidence + sensitivity) / 15)


def _has_cycle(rows: list[Mapping[str, Any]]) -> bool:
    graph = {
        str(row["assumption_id"]): [str(item) for item in row.get("dependencies", [])]
        for row in rows
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        cycle = any(child in graph and walk(child) for child in graph.get(node, []))
        visiting.remove(node)
        visited.add(node)
        return cycle

    return any(walk(node) for node in graph)


def _issue(
    code: str,
    status: str,
    blocking: bool,
    message: str,
    assumption_id: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "status": status,
        "blocking": blocking,
        "message": message,
    }
    if assumption_id is not None:
        result["assumption_id"] = assumption_id
    return result


def _resolve_assumptions(
    refs: list[Any],
    inline: list[Any],
    index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for ref in refs:
        assumption_id = str(ref.get("assumption_id") or "") if isinstance(ref, Mapping) else ""
        row = index.get(assumption_id)
        if row is None:
            issues.append(
                _issue(
                    "UNKNOWN_ASSUMPTION_REF",
                    "FAIL",
                    True,
                    "Referenced institutional assumption does not exist.",
                    assumption_id,
                )
            )
            continue
        requested_version = ref.get("version") if isinstance(ref, Mapping) else None
        if requested_version and str(requested_version) != str(row.get("version")):
            issues.append(
                _issue(
                    "ASSUMPTION_VERSION_MISMATCH",
                    "FAIL",
                    True,
                    "Referenced assumption version does not match the approved library version.",
                    assumption_id,
                )
            )
            continue
        resolved.append(dict(row))
    resolved.extend(dict(row) for row in inline if isinstance(row, Mapping))
    return resolved, issues


def _assessment_date(ticket: Mapping[str, Any]) -> str | None:
    profile = ticket.get("credibility_profile")
    if not isinstance(profile, Mapping):
        return None
    value = profile.get("assessment_date")
    return str(value) if value else None


def _review_overdue(row: Mapping[str, Any], assessment_date: str | None) -> bool:
    review_due = row.get("review_due_at")
    if not review_due or not assessment_date:
        return False
    return date.fromisoformat(str(review_due)[:10]) < date.fromisoformat(assessment_date[:10])


def _row_issues(
    row: Mapping[str, Any],
    known_ids: set[str],
    assessment_date: str | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    assumption_id = str(row.get("assumption_id") or "")
    status = str(row.get("status") or "approved")
    criticality = str(row.get("criticality") or "medium")
    uncertainty_type = str(row.get("uncertainty_type") or "epistemic")
    source_type = str(row.get("source_type") or "")

    if status in {"invalidated", "retired"} or row.get("calibration_status") == "invalidated":
        issues.append(
            _issue(
                "INACTIVE_ASSUMPTION",
                "FAIL",
                True,
                "Invalidated or retired assumptions cannot be used.",
                assumption_id,
            )
        )
    if uncertainty_type in {"epistemic", "mixed"} and not (
        row.get("falsification_test") and row.get("invalid_when")
    ):
        issues.append(
            _issue(
                "EPISTEMIC_TEST_GAP",
                "WARN",
                criticality == "critical",
                "Epistemic uncertainty requires falsification and invalidation conditions.",
                assumption_id,
            )
        )
    if criticality in {"high", "critical"} and not (
        row.get("sensitivity_required") or isinstance(row.get("sensitivity_rank"), int)
    ):
        issues.append(
            _issue(
                "SENSITIVITY_TREATMENT_GAP",
                "WARN",
                criticality == "critical",
                "High-impact assumptions require sensitivity or stress treatment.",
                assumption_id,
            )
        )
    if source_type in WEAK_SOURCE_TYPES and not row.get("evidence_sha256"):
        issues.append(
            _issue(
                "ASSUMPTION_EVIDENCE_GAP",
                "WARN",
                criticality == "critical",
                "Weak-source assumptions require immutable supporting evidence or explicit approval.",
                assumption_id,
            )
        )
    unknown = sorted(set(str(item) for item in row.get("dependencies", [])) - known_ids)
    if unknown:
        issues.append(
            _issue(
                "UNKNOWN_ASSUMPTION_DEPENDENCY",
                "FAIL",
                True,
                f"Unknown dependencies: {', '.join(unknown)}",
                assumption_id,
            )
        )
    if _review_overdue(row, assessment_date):
        issues.append(
            _issue(
                "ASSUMPTION_REVIEW_OVERDUE",
                "FAIL",
                criticality in {"high", "critical"},
                "Assumption review date has expired for the declared assessment date.",
                assumption_id,
            )
        )
    return issues


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = profile.get("decision_class") if isinstance(profile, Mapping) else None
    return str(value or "formal")


def _risk_ranking(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranking = [
        {
            "assumption_id": str(row.get("assumption_id") or ""),
            "risk_score": _risk_score(row),
            "criticality": str(row.get("criticality") or "medium"),
            "confidence": str(row.get("confidence") or "low"),
            "uncertainty_type": str(row.get("uncertainty_type") or "epistemic"),
        }
        for row in rows
    ]
    return sorted(ranking, key=lambda item: (-item["risk_score"], item["assumption_id"]))


def assess_assumptions(ticket: Mapping[str, Any]) -> dict[str, Any]:
    library = load_library()
    index = {
        str(row["assumption_id"]): row for row in library.get("assumptions", [])
    }
    refs = ticket.get("assumption_refs")
    inline = ticket.get("assumption_register")
    refs = refs if isinstance(refs, list) else []
    inline = inline if isinstance(inline, list) else []
    _validate(REGISTER_SCHEMA_PATH, inline, "assumption_register")

    resolved, issues = _resolve_assumptions(refs, inline, index)
    ids = [str(row.get("assumption_id") or "") for row in resolved]
    if len(ids) != len(set(ids)):
        issues.append(
            _issue(
                "DUPLICATE_ASSUMPTION",
                "FAIL",
                True,
                "Resolved assumption IDs must be unique.",
            )
        )
    known_ids = set(ids)
    assessment_date = _assessment_date(ticket)
    for row in resolved:
        issues.extend(_row_issues(row, known_ids, assessment_date))
    if _has_cycle(resolved):
        issues.append(
            _issue(
                "ASSUMPTION_DEPENDENCY_CYCLE",
                "FAIL",
                True,
                "Assumption dependencies must be acyclic.",
            )
        )

    decision_class = _decision_class(ticket)
    if not resolved:
        issues.append(
            _issue(
                "NO_EXPLICIT_ASSUMPTIONS",
                "WARN",
                decision_class == "high_stakes",
                "No explicit structural, boundary, data, numerical or parameter assumptions were declared.",
            )
        )
    blocking = [row for row in issues if row.get("blocking") and row.get("status") == "FAIL"]
    warnings = [row for row in issues if row.get("status") == "WARN"]
    status = "BLOCKED" if blocking else "WARN" if warnings else "PASS"
    return {
        "schema_version": "compute-assumption-assurance-v1",
        "status": status,
        "decision_class": decision_class,
        "library_reference_count": len(refs),
        "inline_assumption_count": len(inline),
        "resolved_assumption_count": len(resolved),
        "resolved_snapshot_sha256": _sha(resolved),
        "risk_ranking": _risk_ranking(resolved),
        "issues": issues,
        "blocking_issue_count": len(blocking),
        "warning_count": len(warnings),
        "resolved_assumptions": resolved,
    }
