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


def _candidate(variable: Mapping[str, Any], existing: Mapping[str, Any] | None) -> dict[str, Any]:
    name = str(variable.get("name") or "")
    expected = variable.get("expected_range") if isinstance(variable.get("expected_range"), Mapping) else {}
    minimum = expected.get("minimum")
    maximum = expected.get("maximum")
    source_type = str(variable.get("source_type") or "")
    confidence = str(variable.get("confidence") or "low")
    candidate: dict[str, Any] = {
        "variable": name,
        "point_estimate_allowed": False,
        "preferred_source_order": [
            "api_observation",
            "user_provided_record",
            "hashed_public_snapshot",
            "same-definition_historical_sample",
            "approved_benchmark_sample",
            "explicit_proxy",
            "approved_range_or_distribution",
        ],
        "current_source_type": source_type,
        "current_confidence": confidence,
        "candidate_distribution": None,
        "candidate_range": None,
        "approval_required": True,
        "execution_use_allowed": False,
        "reason": "No defensible observation is available.",
    }
    if minimum is not None and maximum is not None and float(minimum) <= float(maximum):
        candidate["candidate_range"] = {"minimum": float(minimum), "maximum": float(maximum)}
        candidate["candidate_distribution"] = "uniform"
        candidate["reason"] = "A declared admissible range exists; uncertainty must be propagated rather than collapsed to one value."
    if existing is not None:
        approved = str(existing.get("approved_by") or existing.get("status") or "") in {
            "user", "gpts_policy", "approved", "calibrated", "validated"
        }
        has_bounds = (
            isinstance(existing.get("sensitivity_range"), Mapping)
            or (existing.get("minimum") is not None and existing.get("maximum") is not None)
            or existing.get("distribution") == "constant"
        )
        has_basis = bool(existing.get("basis"))
        has_invalidation = bool(existing.get("invalid_when") or existing.get("falsification_test"))
        candidate["approval_required"] = not approved
        candidate["execution_use_allowed"] = bool(approved and has_bounds and has_basis and has_invalidation)
        candidate["existing_assumption_found"] = True
        candidate["assumption_checks"] = {
            "approved": approved,
            "range_or_distribution_present": has_bounds,
            "basis_present": has_basis,
            "invalidation_condition_present": has_invalidation,
        }
        if candidate["execution_use_allowed"]:
            candidate["reason"] = "An explicit approved assumption with uncertainty treatment and invalidation rule is available."
    else:
        candidate["existing_assumption_found"] = False
    return candidate


def build_assumption_plan(ticket: Mapping[str, Any]) -> dict[str, Any]:
    context = ticket.get("data_context") if isinstance(ticket.get("data_context"), Mapping) else {}
    variables = context.get("variables") if isinstance(context.get("variables"), list) else []
    assumptions = _assumption_index(ticket)
    rows: list[dict[str, Any]] = []
    for variable in variables:
        if not isinstance(variable, Mapping):
            continue
        missing = bool(variable.get("missing"))
        replacement = str(variable.get("replacement_strategy") or "none")
        weak = str(variable.get("confidence") or "low") == "low"
        assumed_source = str(variable.get("source_type") or "") in {"proxy", "gpts_assumption", "expert_hypothesis"}
        if missing or replacement != "none" or weak or assumed_source:
            name = str(variable.get("name") or "")
            rows.append(_candidate(variable, assumptions.get(name)))

    unresolved = [row for row in rows if not row["execution_use_allowed"]]
    permitted = [row for row in rows if row["execution_use_allowed"]]
    decision_class = str(((ticket.get("quality_profile") or {}).get("decision_class") if isinstance(ticket.get("quality_profile"), Mapping) else None) or "formal")
    status = "BLOCKED" if unresolved and decision_class in {"formal", "high_stakes"} else "CONDITIONAL" if unresolved else "PASS"
    plan: dict[str, Any] = {
        "schema_version": "compute-assumption-plan-v1",
        "task_id": str(ticket.get("task_id") or ""),
        "operation": str(ticket.get("operation") or ""),
        "decision_class": decision_class,
        "status": status,
        "gap_count": len(rows),
        "approved_assumption_count": len(permitted),
        "unresolved_count": len(unresolved),
        "assumption_candidates": rows,
        "rules": [
            "Never silently fill zero, mean, median or a model-generated point value.",
            "Prefer observed data; then user records, hashed public evidence, comparable historical data, benchmarks, proxies, and only then an approved range or distribution.",
            "Low-confidence values require a range or distribution and sensitivity, scenario or Monte Carlo treatment.",
            "A result that changes materially across the approved range cannot be released as one unconditional recommendation.",
            "Every assumption must include a basis, confidence, invalidation condition and approval owner.",
        ],
        "new_ticket_required_after_resolution": bool(unresolved),
    }
    plan["plan_sha256"] = _sha(plan)
    return plan
