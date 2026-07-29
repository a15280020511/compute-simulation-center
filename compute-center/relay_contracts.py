#!/usr/bin/env python3
"""Build deterministic relay documents without contacting another center.

The compute center may describe what GPTs should do next, but it never calls the API
center or expert center. GPTs remains the sole controller and evidence courier.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

RELAY_OWNER = "gpts-usage-center"


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fallback_chain(variable: str | None) -> list[dict[str, Any]]:
    subject = variable or "unresolved-data-gap"
    return [
        {
            "priority": 1,
            "actor": RELAY_OWNER,
            "action": "create_separate_api_ticket",
            "subject": subject,
            "condition": "An enabled allowlisted API connector can supply an observed value or defensible proxy.",
            "direct_center_call": False,
        },
        {
            "priority": 2,
            "actor": RELAY_OWNER,
            "action": "request_user_record",
            "subject": subject,
            "condition": "The user can provide a real record with unit, time and scope metadata.",
            "direct_center_call": False,
        },
        {
            "priority": 3,
            "actor": RELAY_OWNER,
            "action": "capture_verifiable_public_evidence",
            "subject": subject,
            "condition": "A public source can be snapshotted and hashed outside the compute runtime.",
            "direct_center_call": False,
        },
        {
            "priority": 4,
            "actor": RELAY_OWNER,
            "action": "use_historical_or_benchmark_data",
            "subject": subject,
            "condition": "Comparable source, region, period and business definitions are documented.",
            "direct_center_call": False,
        },
        {
            "priority": 5,
            "actor": RELAY_OWNER,
            "action": "propose_explicit_proxy",
            "subject": subject,
            "condition": "Proxy relationship, confidence and invalidation rule are recorded.",
            "direct_center_call": False,
        },
        {
            "priority": 6,
            "actor": RELAY_OWNER,
            "action": "propose_assumption_range_for_user_approval",
            "subject": subject,
            "condition": "No better source exists; use a range or distribution and run sensitivity or scenario analysis.",
            "direct_center_call": False,
        },
    ]


def build_data_gap_plan(ticket: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for issue in preflight.get("issues") or []:
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or "UNKNOWN")
        variable = str(issue.get("variable") or "") or None
        blocking = bool(issue.get("blocking"))
        request: dict[str, Any] = {
            "issue_code": code,
            "variable": variable,
            "blocking": blocking,
            "message": str(issue.get("message") or ""),
            "remediation": str(issue.get("remediation") or ""),
            "relay_owner": RELAY_OWNER,
            "center_direct_contact_allowed": False,
        }
        if code in {
            "REQUIRED_DATA_MISSING",
            "DATA_REPLACEMENT_REQUIRED",
            "ASSUMPTION_RATIO_HIGH",
            "PROXY_DATA_FORBIDDEN",
            "ASSUMPTIONS_FORBIDDEN",
        }:
            request["ordered_fallback_chain"] = _fallback_chain(variable)
        elif code in {
            "LOW_CONFIDENCE_ASSUMPTION_NOT_APPROVED",
            "EXPERT_HYPOTHESIS_NOT_APPROVED",
        }:
            request["required_action"] = "gpts_present_assumption_to_user_and_create_new_approved_compute_ticket"
        elif code in {
            "LOW_CONFIDENCE_ASSUMPTION_WITHOUT_RANGE",
            "MEDIUM_CONFIDENCE_ASSUMPTION_WITHOUT_RANGE",
        }:
            request["required_action"] = "gpts_add_range_or_distribution_and_schedule_sensitivity_or_scenario_analysis"
        else:
            request["required_action"] = "gpts_correct_or_restructure_compute_ticket"
        requests.append(request)

    plan: dict[str, Any] = {
        "schema_version": "compute-data-gap-plan-v1",
        "task_id": str(ticket.get("task_id") or ""),
        "operation": str(ticket.get("operation") or ""),
        "preflight_status": str(preflight.get("status") or ""),
        "execution_allowed": bool(preflight.get("execution_allowed")),
        "relay_owner": RELAY_OWNER,
        "compute_center_network_policy": "deny",
        "compute_center_external_fetches": 0,
        "center_direct_contact_allowed": False,
        "gpts_relay_requests": requests,
        "recommended_operations": list(preflight.get("recommended_operations") or []),
        "stop_rule": "If no defensible observed, user, public, historical, benchmark, proxy or approved range exists, keep DATA_INSUFFICIENT and do not manufacture a point estimate.",
        "new_ticket_required_after_change": bool(requests),
    }
    plan["plan_sha256"] = _sha(plan)
    return plan


def build_expert_review_request(
    ticket: Mapping[str, Any],
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": "compute-expert-review-request-v1",
        "task_id": str(ticket.get("task_id") or ""),
        "operation": str(ticket.get("operation") or ""),
        "source_center": "compute-center",
        "target_center": "expert-center",
        "relay_owner": RELAY_OWNER,
        "delivery_mode": "GPTs creates a separate [execution] ticket after retrieving and verifying the complete compute package.",
        "direct_center_delivery_allowed": False,
        "compute_center_dispatch_performed": False,
        "required_verified_files": [
            "compute-preflight.json",
            "compute-result.json",
            "compute-audit.json",
            "compute-diagnostics.json",
            "artifact-manifest.json",
        ],
        "required_review_tasks": [
            "Interpret the numerical result in the user's decision context without changing the computation.",
            "Red-team the data provenance, assumptions, omitted variables and model-form risk.",
            "Identify which parameters can reverse the recommendation and state decision thresholds.",
            "Compare the result with plausible alternative explanations and adverse scenarios.",
            "State applicability limits, invalidation conditions and additional evidence worth obtaining.",
        ],
        "preflight_status": str(preflight.get("status") or ""),
        "data_summary": dict(preflight.get("data_summary") or {}),
        "recommended_follow_up_operations": list(preflight.get("recommended_operations") or []),
        "compute_package_sha256": str(result.get("package_sha256") or ""),
        "decision_support_only": True,
    }
    request["request_sha256"] = _sha(request)
    return request
