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


class AssumptionGovernanceError(ValueError):
    pass


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate(schema_path: Path, value: Any, label: str) -> None:
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        where = ".".join(str(item) for item in first.absolute_path) or "$"
        raise AssumptionGovernanceError(f"{label} {where}: {first.message}")


def load_library() -> dict[str, Any]:
    library = _load(LIBRARY_PATH)
    _validate(LIBRARY_SCHEMA_PATH, library, "assumption_library")
    rows = library.get("assumptions") or []
    _validate(REGISTER_SCHEMA_PATH, rows, "assumption_library.assumptions")
    seen: set[str] = set()
    required = {"assumption_id", "version", "status", "type", "statement", "source_type", "basis", "confidence", "calibration_status", "uncertainty_type", "criticality", "evidence_strength", "owner", "intended_use", "content_sha256"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise AssumptionGovernanceError(f"library assumption missing fields: {sorted(missing)}")
        key = f"{row['assumption_id']}@{row['version']}"
        if key in seen:
            raise AssumptionGovernanceError(f"duplicate library assumption: {key}")
        seen.add(key)
        expected = _sha({key: value for key, value in row.items() if key != "content_sha256"})
        if row["content_sha256"] != expected:
            raise AssumptionGovernanceError(f"library assumption hash mismatch: {key}")
    return library


def _risk_score(row: Mapping[str, Any]) -> int:
    confidence = {"high": 1, "medium": 2, "low": 3}.get(str(row.get("confidence")), 3)
    criticality = {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(str(row.get("criticality")), 2)
    evidence = {"strong": 1, "moderate": 2, "weak": 3, "none": 4}.get(str(row.get("evidence_strength")), 3)
    rank = row.get("sensitivity_rank")
    sensitivity = 4 if rank == 1 else 3 if isinstance(rank, int) and rank <= 3 else 2 if isinstance(rank, int) and rank <= 10 else 1
    return round(100 * (confidence + criticality + evidence + sensitivity) / 15)


def _has_cycle(rows: list[Mapping[str, Any]]) -> bool:
    graph = {str(row["assumption_id"]): [str(item) for item in row.get("dependencies", [])] for row in rows}
    visiting: set[str] = set(); visited: set[str] = set()
    def walk(node: str) -> bool:
        if node in visiting: return True
        if node in visited: return False
        visiting.add(node)
        for child in graph.get(node, []):
            if child in graph and walk(child): return True
        visiting.remove(node); visited.add(node); return False
    return any(walk(node) for node in graph)


def assess_assumptions(ticket: Mapping[str, Any]) -> dict[str, Any]:
    library = load_library()
    index = {str(row["assumption_id"]): row for row in library.get("assumptions", [])}
    refs = ticket.get("assumption_refs") if isinstance(ticket.get("assumption_refs"), list) else []
    inline = ticket.get("assumption_register") if isinstance(ticket.get("assumption_register"), list) else []
    _validate(REGISTER_SCHEMA_PATH, inline, "assumption_register")
    issues: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for ref in refs:
        assumption_id = str(ref.get("assumption_id") or "") if isinstance(ref, Mapping) else ""
        row = index.get(assumption_id)
        if row is None:
            issues.append({"code":"UNKNOWN_ASSUMPTION_REF","status":"FAIL","blocking":True,"assumption_id":assumption_id,"message":"Referenced institutional assumption does not exist."})
            continue
        if isinstance(ref, Mapping) and ref.get("version") and str(ref["version"]) != str(row.get("version")):
            issues.append({"code":"ASSUMPTION_VERSION_MISMATCH","status":"FAIL","blocking":True,"assumption_id":assumption_id,"message":"Referenced assumption version does not match the approved library version."})
            continue
        resolved.append(dict(row))
    resolved.extend(dict(row) for row in inline)
    ids = [str(row.get("assumption_id") or "") for row in resolved]
    if len(ids) != len(set(ids)):
        issues.append({"code":"DUPLICATE_ASSUMPTION","status":"FAIL","blocking":True,"message":"Resolved assumption IDs must be unique."})
    known = set(ids)
    for row in resolved:
        assumption_id = str(row.get("assumption_id") or "")
        status = str(row.get("status") or "approved")
        criticality = str(row.get("criticality") or "medium")
        uncertainty_type = str(row.get("uncertainty_type") or "epistemic")
        source_type = str(row.get("source_type") or "")
        if status in {"invalidated", "retired"} or row.get("calibration_status") == "invalidated":
            issues.append({"code":"INACTIVE_ASSUMPTION","status":"FAIL","blocking":True,"assumption_id":assumption_id,"message":"Invalidated or retired assumptions cannot be used."})
        if uncertainty_type in {"epistemic", "mixed"} and not (row.get("falsification_test") and row.get("invalid_when")):
            issues.append({"code":"EPISTEMIC_TEST_GAP","status":"WARN","blocking":criticality == "critical","assumption_id":assumption_id,"message":"Epistemic uncertainty requires falsification and invalidation conditions."})
        if criticality in {"high", "critical"} and not (row.get("sensitivity_required") or isinstance(row.get("sensitivity_rank"), int)):
            issues.append({"code":"SENSITIVITY_TREATMENT_GAP","status":"WARN","blocking":criticality == "critical","assumption_id":assumption_id,"message":"High-impact assumptions require sensitivity or stress treatment."})
        if source_type in {"proxy", "gpts_assumption", "expert_elicitation", "user_assumption"} and not row.get("evidence_sha256"):
            issues.append({"code":"ASSUMPTION_EVIDENCE_GAP","status":"WARN","blocking":criticality == "critical","assumption_id":assumption_id,"message":"Weak-source assumptions require immutable supporting evidence or explicit approval."})
        unknown_dependencies = sorted(set(str(item) for item in row.get("dependencies", [])) - known)
        if unknown_dependencies:
            issues.append({"code":"UNKNOWN_ASSUMPTION_DEPENDENCY","status":"FAIL","blocking":True,"assumption_id":assumption_id,"message":f"Unknown dependencies: {', '.join(unknown_dependencies)}"})
        review_due = row.get("review_due_at")
        assessment_date = ((ticket.get("credibility_profile") or {}).get("assessment_date") if isinstance(ticket.get("credibility_profile"), Mapping) else None)
        if review_due and assessment_date and date.fromisoformat(str(review_due)[:10]) < date.fromisoformat(str(assessment_date)[:10]):
            issues.append({"code":"ASSUMPTION_REVIEW_OVERDUE","status":"FAIL","blocking":criticality in {"high","critical"},"assumption_id":assumption_id,"message":"Assumption review date has expired for the declared assessment date."})
    if _has_cycle(resolved):
        issues.append({"code":"ASSUMPTION_DEPENDENCY_CYCLE","status":"FAIL","blocking":True,"message":"Assumption dependencies must be acyclic."})
    decision_class = str(((ticket.get("quality_profile") or {}).get("decision_class") if isinstance(ticket.get("quality_profile"), Mapping) else None) or "formal")
    if not resolved:
        issues.append({"code":"NO_EXPLICIT_ASSUMPTIONS","status":"WARN","blocking":decision_class == "high_stakes","message":"No explicit structural, boundary, data, numerical or parameter assumptions were declared."})
    blocking = [row for row in issues if row.get("blocking") and row.get("status") == "FAIL"]
    warnings = [row for row in issues if row.get("status") == "WARN"]
    status = "BLOCKED" if blocking else "WARN" if warnings else "PASS"
    ranked = sorted([{"assumption_id":str(row.get("assumption_id") or ""),"risk_score":_risk_score(row),"criticality":str(row.get("criticality") or "medium"),"confidence":str(row.get("confidence") or "low"),"uncertainty_type":str(row.get("uncertainty_type") or "epistemic")} for row in resolved], key=lambda row: (-row["risk_score"], row["assumption_id"]))
    return {"schema_version":"compute-assumption-assurance-v1","status":status,"decision_class":decision_class,"library_reference_count":len(refs),"inline_assumption_count":len(inline),"resolved_assumption_count":len(resolved),"resolved_snapshot_sha256":_sha(resolved),"risk_ranking":ranked,"issues":issues,"blocking_issue_count":len(blocking),"warning_count":len(warnings),"resolved_assumptions":resolved}
