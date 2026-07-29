#!/usr/bin/env python3
"""Build a fail-closed assumption plan for missing or weak input data.

The runtime never invents an observed value. It classifies gaps, proposes an ordered
source strategy, and permits an assumption only when its range/distribution, evidence,
invalidating condition and approval are explicit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


_SOURCE_ORDER = [
    "api_observation",
    "user_provided_record",
    "hashed_public_snapshot",
    "same-definition_historical_sample",
    "approved_benchmark_sample",
    "explicit_proxy",
    "approved_range_or_distribution",
]
_RULES = [
    "Never silently fill zero, mean, median or a model-generated point value.",
    "Prefer observed data; then user records, hashed public evidence, comparable historical data, benchmarks, proxies, and only then an approved range or distribution.",
    "Low-confidence values require a range or distribution and sensitivity, scenario or Monte Carlo treatment.",
    "A result that changes materially across the approved range cannot be released as one unconditional recommendation.",
    "Every assumption must include a basis, confidence, invalidation condition and approval owner.",
]
_APPROVED_STATES = {"user", "gpts_policy", "approved", "calibrated", "validated"}
_ASSUMED_SOURCE_TYPES = {"proxy", "gpts_assumption", "expert_hypothesis"}


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _assumption_index(ticket: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("assumptions", "assumption_register"):
        value = ticket.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, Mapping))
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = str(row.get("name") or row.get("linked_parameter") or row.get("assumption_id") or "")
        if name:
            index[name] = row
    return index


def _declared_range(variable: Mapping[str, Any]) -> dict[str, float] | None:
    expected = variable.get("expected_range")
    if not isinstance(expected, Mapping):
        return None
    minimum, maximum = expected.get("minimum"), expected.get("maximum")
    if minimum is None or maximum is None:
        return None
    lower, upper = float(minimum), float(maximum)
    return {"minimum": lower, "maximum": upper} if lower <= upper else None


def _assumption_checks(existing: Mapping[str, Any]) -> dict[str, bool]:
    approved = str(existing.get("approved_by") or existing.get("status") or "") in _APPROVED_STATES
    has_bounds = (
        isinstance(existing.get("sensitivity_range"), Mapping)
        or (existing.get("minimum") is not None and existing.get("maximum") is not None)
        or existing.get("distribution") == "constant"
    )
    return {
        "approved": approved,
        "range_or_distribution_present": has_bounds,
        "basis_present": bool(existing.get("basis")),
        "invalidation_condition_present": bool(existing.get("invalid_when") or existing.get("falsification_test")),
    }


def _candidate(variable: Mapping[str, Any], existing: Mapping[str, Any] | None) -> dict[str, Any]:
    name = str(variable.get("name") or "")
    candidate: dict[str, Any] = {
        "variable": name,
        "point_estimate_allowed": False,
        "preferred_source_order": list(_SOURCE_ORDER),
        "current_source_type": str(variable.get("source_type") or ""),
        "current_confidence": str(variable.get("confidence") or "low"),
        "candidate_distribution": None,
        "candidate_range": None,
        "approval_required": True,
        "execution_use_allowed": False,
        "reason": "No defensible observation is available.",
    }
    declared_range = _declared_range(variable)
    if declared_range is not None:
        candidate.update(
            candidate_range=declared_range,
            candidate_distribution="uniform",
            reason="A declared admissible range exists; uncertainty must be propagated rather than collapsed to one value.",
        )
    if existing is None:
        candidate["existing_assumption_found"] = False
        return candidate

    checks = _assumption_checks(existing)
    allowed = all(checks.values())
    candidate.update(
        approval_required=not checks["approved"],
        execution_use_allowed=allowed,
        existing_assumption_found=True,
        assumption_checks=checks,
    )
    if allowed:
        candidate["reason"] = "An explicit approved assumption with uncertainty treatment and invalidation rule is available."
    return candidate


def _needs_assumption(variable: Mapping[str, Any]) -> bool:
    return any(
        (
            bool(variable.get("missing")),
            str(variable.get("replacement_strategy") or "none") != "none",
            str(variable.get("confidence") or "low") == "low",
            str(variable.get("source_type") or "") in _ASSUMED_SOURCE_TYPES,
        )
    )


def _candidate_rows(ticket: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = ticket.get("data_context")
    variables = context.get("variables") if isinstance(context, Mapping) else []
    if not isinstance(variables, list):
        return []
    assumptions = _assumption_index(ticket)
    rows: list[dict[str, Any]] = []
    for variable in variables:
        if isinstance(variable, Mapping) and _needs_assumption(variable):
            name = str(variable.get("name") or "")
            rows.append(_candidate(variable, assumptions.get(name)))
    return rows


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = profile.get("decision_class") if isinstance(profile, Mapping) else None
    return str(value or "formal")


def _plan_status(decision_class: str, unresolved_count: int) -> str:
    if not unresolved_count:
        return "PASS"
    return "BLOCKED" if decision_class in {"formal", "high_stakes"} else "CONDITIONAL"


def build_assumption_plan(ticket: Mapping[str, Any]) -> dict[str, Any]:
    rows = _candidate_rows(ticket)
    approved_count = sum(bool(row["execution_use_allowed"]) for row in rows)
    unresolved_count = len(rows) - approved_count
    decision_class = _decision_class(ticket)
    plan: dict[str, Any] = {
        "schema_version": "compute-assumption-plan-v1",
        "task_id": str(ticket.get("task_id") or ""),
        "operation": str(ticket.get("operation") or ""),
        "decision_class": decision_class,
        "status": _plan_status(decision_class, unresolved_count),
        "gap_count": len(rows),
        "approved_assumption_count": approved_count,
        "unresolved_count": unresolved_count,
        "assumption_candidates": rows,
        "rules": list(_RULES),
        "new_ticket_required_after_resolution": bool(unresolved_count),
    }
    plan["plan_sha256"] = _sha(plan)
    return plan
