#!/usr/bin/env python3
"""Deterministic data-readiness preflight for compute tickets.

The preflight does not fetch data, call models, execute arbitrary code, or alter the
business inputs. It classifies data gaps, assumptions, representative-value choices,
and numerical risks before the selected fixed operation runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "data-readiness-policy.json"
OUTPUT_SCHEMA_PATH = HERE / "compute-preflight.schema.json"
ASSUMPTION_SOURCES = {"gpts_assumption", "expert_hypothesis"}
OBSERVED_SOURCES = {"api_observation", "user_provided", "public_source", "historical"}
APPROVAL_BLOCK_CODES = {
    "LOW_CONFIDENCE_ASSUMPTION_NOT_APPROVED",
    "EXPERT_HYPOTHESIS_NOT_APPROVED",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    blocking: bool,
    variable: str | None = None,
    remediation: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "blocking": blocking,
        "variable": variable,
        "remediation": remediation,
    }


class _AssessmentState:
    """Mutable state used only during one deterministic preflight assessment."""

    def __init__(self) -> None:
        self.issues: list[dict[str, Any]] = []
        self.representative: list[dict[str, Any]] = []
        self.recommended_operations: set[str] = set()
        self.observed_count = 0
        self.assumption_variable_count = 0
        self.missing_required_count = 0
        self.medium_assumption_count = 0
        self.low_unapproved_count = 0


def _walk_numbers(value: Any, path: str = "inputs") -> Iterable[tuple[str, str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, path.rsplit(".", 1)[-1].casefold(), float(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, f"{path}[{index}]")


def _walk_named_lists(value: Any, path: str = "inputs") -> Iterable[tuple[str, str, list[Any]]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = f"{path}.{key}"
            if isinstance(item, list):
                yield next_path, str(key).casefold(), item
            yield from _walk_named_lists(item, next_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_named_lists(item, f"{path}[{index}]")


def _merge_policy(ticket: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(policy.get("default_policy") or {})
    requested = ticket.get("preflight_policy")
    if isinstance(requested, Mapping):
        merged.update(requested)
    return merged


def _declared_rows(ticket: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    assumptions = ticket.get("assumptions")
    data_context = ticket.get("data_context")
    variables: Any = []
    if isinstance(data_context, Mapping):
        variables = data_context.get("variables")
    return (
        assumptions if isinstance(assumptions, list) else [],
        variables if isinstance(variables, list) else [],
    )


def _representative_method(variable: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[str, str]:
    characteristics = variable.get("characteristics")
    if not isinstance(characteristics, Mapping):
        characteristics = {}
    if characteristics.get("weights_available") is True:
        return "weighted_mean", "样本权重不同"
    groups = characteristics.get("group_dimensions")
    if isinstance(groups, list) and groups:
        return "grouped_summary", "存在时段、地域或类别分组差异"
    if characteristics.get("skewed") is True:
        return "median", "分布偏态"
    outlier_rate = characteristics.get("outlier_rate")
    if _above_threshold(outlier_rate, policy.get("outlier_rate_threshold", 0.05)):
        return "trimmed_mean", "异常值比例超过阈值"
    sample_size = variable.get("sample_size")
    if isinstance(sample_size, int) and sample_size < int(policy.get("small_sample_threshold", 10)):
        return "interval", "样本量较少，不宜输出单点平均值"
    return "arithmetic_mean", "未声明偏态、权重、分组或显著异常值"


def _above_threshold(value: Any, threshold: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > float(threshold)
    )


def _numeric_scalar_issues(inputs: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for path, name, number in _walk_numbers(inputs):
        if not math.isfinite(number):
            issues.append(
                _issue(
                    "high",
                    "NON_FINITE_VALUE",
                    f"{path} is not finite",
                    blocking=True,
                    variable=path,
                    remediation="Replace NaN or Infinity with a finite observed value, interval, or explicit assumption.",
                )
            )
        if any(token in name for token in ("probability", "chance", "likelihood")) and not 0.0 <= number <= 1.0:
            issues.append(
                _issue(
                    "high",
                    "INVALID_PROBABILITY_RANGE",
                    f"{path}={number} is outside [0,1]",
                    blocking=True,
                    variable=path,
                    remediation="Correct the probability or document a different unit such as percent.",
                )
            )
        if name in {"denominator", "divisor"} and number == 0:
            issues.append(
                _issue(
                    "high",
                    "DIVISION_BY_ZERO_RISK",
                    f"{path} is zero",
                    blocking=True,
                    variable=path,
                    remediation="Provide a non-zero denominator or reformulate the fixed operation.",
                )
            )
    return issues


def _numeric_list_issues(inputs: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    probability_names = {"probabilities", "probability_vector", "initial_probabilities"}
    for path, name, values in _walk_named_lists(inputs):
        if name not in probability_names:
            continue
        numeric = [
            float(item)
            for item in values
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if len(numeric) != len(values) or not numeric:
            continue
        total = sum(numeric)
        if any(item < 0 or item > 1 for item in numeric) or abs(total - 1.0) > 1e-6:
            issues.append(
                _issue(
                    "high",
                    "INVALID_PROBABILITY_VECTOR",
                    f"{path} must contain values in [0,1] summing to 1; observed sum={total}",
                    blocking=True,
                    variable=path,
                    remediation="Normalize or correct the probability vector before computation.",
                )
            )
    return issues


def _numeric_issues(inputs: Any) -> list[dict[str, Any]]:
    return _numeric_scalar_issues(inputs) + _numeric_list_issues(inputs)


def _append_missing_context_issue(state: _AssessmentState, variables: list[Any]) -> None:
    if variables:
        return
    state.issues.append(
        _issue(
            "warning",
            "DATA_CONTEXT_NOT_DECLARED",
            "No variable-level source, unit, freshness, sample, or missingness metadata was supplied.",
            blocking=False,
            remediation="GPTs should add data_context.variables for material decisions; legacy tickets remain executable in advisory mode.",
        )
    )


def _append_missing_data_issue(
    state: _AssessmentState,
    *,
    name: str,
    missing: bool,
    required: bool,
    replacement: str,
) -> None:
    if missing and required and replacement == "none":
        state.missing_required_count += 1
        state.issues.append(
            _issue(
                "high",
                "REQUIRED_DATA_MISSING",
                f"Required variable {name} is missing without an approved replacement strategy.",
                blocking=True,
                variable=name,
                remediation="Use GPTs to obtain the value from the API catalog, user input, public evidence, benchmark, proxy, or an explicit approved assumption.",
            )
        )
    elif missing:
        state.issues.append(
            _issue(
                "warning",
                "DATA_REPLACEMENT_REQUIRED",
                f"Variable {name} has missing data and declares replacement_strategy={replacement}.",
                blocking=False,
                variable=name,
                remediation="Apply the declared replacement outside the compute runtime and preserve provenance.",
            )
        )


def _append_source_policy_issues(
    state: _AssessmentState,
    *,
    name: str,
    source_type: str,
    policy: Mapping[str, Any],
) -> None:
    if source_type == "proxy" and not bool(policy.get("allow_proxy", True)):
        state.issues.append(
            _issue(
                "high",
                "PROXY_DATA_FORBIDDEN",
                f"Variable {name} uses proxy data while the ticket policy forbids proxies.",
                blocking=True,
                variable=name,
                remediation="Replace with observed, user-provided, public, historical, or benchmark data.",
            )
        )
    if source_type in ASSUMPTION_SOURCES and not bool(policy.get("allow_assumptions", True)):
        state.issues.append(
            _issue(
                "high",
                "ASSUMPTIONS_FORBIDDEN",
                f"Variable {name} is assumption-based while assumptions are disabled.",
                blocking=True,
                variable=name,
                remediation="Obtain observed data or enable assumptions with explicit approval.",
            )
        )


def _assess_variable(
    state: _AssessmentState,
    variable: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    name = str(variable.get("name") or "unknown")
    source_type = str(variable.get("source_type") or "unknown")
    confidence = str(variable.get("confidence") or "unknown")
    missing = bool(variable.get("missing", False)) or int(variable.get("missing_count") or 0) > 0
    required = bool(variable.get("required", True))
    replacement = str(variable.get("replacement_strategy") or "none")

    state.observed_count += int(source_type in OBSERVED_SOURCES)
    state.assumption_variable_count += int(source_type in ASSUMPTION_SOURCES)
    _append_missing_data_issue(
        state,
        name=name,
        missing=missing,
        required=required,
        replacement=replacement,
    )
    _append_source_policy_issues(state, name=name, source_type=source_type, policy=policy)
    if confidence == "low" and source_type in ASSUMPTION_SOURCES:
        state.recommended_operations.update({"sensitivity_analysis", "scenario_compare", "monte_carlo"})

    method, reason = _representative_method(variable, policy)
    state.representative.append(
        {
            "variable": name,
            "method": method,
            "reason": reason,
            "computed_by_preflight": False,
        }
    )
    characteristics = variable.get("characteristics")
    if isinstance(characteristics, Mapping) and characteristics.get("time_series") is True:
        state.recommended_operations.add("time_series_forecast")


def _assess_variables(
    state: _AssessmentState,
    variables: list[Any],
    policy: Mapping[str, Any],
) -> None:
    _append_missing_context_issue(state, variables)
    for variable in variables:
        if isinstance(variable, Mapping):
            _assess_variable(state, variable, policy)


def _append_low_confidence_issues(
    state: _AssessmentState,
    assumption: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    name = str(assumption.get("name") or "unknown")
    approved_by = str(assumption.get("approved_by") or "not_approved")
    sensitivity_range = assumption.get("sensitivity_range")
    state.recommended_operations.update({"sensitivity_analysis", "scenario_compare", "monte_carlo"})
    if not isinstance(sensitivity_range, Mapping):
        state.issues.append(
            _issue(
                "high",
                "LOW_CONFIDENCE_ASSUMPTION_WITHOUT_RANGE",
                f"Low-confidence assumption {name} has no sensitivity_range.",
                blocking=True,
                variable=name,
                remediation="Provide minimum and maximum values or an explicit probability distribution.",
            )
        )
    if bool(policy.get("require_user_approval_for_low_confidence", True)) and approved_by != "user":
        state.low_unapproved_count += 1
        state.issues.append(
            _issue(
                "high",
                "LOW_CONFIDENCE_ASSUMPTION_NOT_APPROVED",
                f"Low-confidence assumption {name} is not approved by the user.",
                blocking=True,
                variable=name,
                remediation="GPTs must present the assumption and range to the user, then create a new ticket with approved_by=user.",
            )
        )


def _append_medium_confidence_issues(
    state: _AssessmentState,
    assumption: Mapping[str, Any],
) -> None:
    name = str(assumption.get("name") or "unknown")
    state.medium_assumption_count += 1
    state.recommended_operations.update({"sensitivity_analysis", "scenario_compare"})
    if not isinstance(assumption.get("sensitivity_range"), Mapping):
        state.issues.append(
            _issue(
                "warning",
                "MEDIUM_CONFIDENCE_ASSUMPTION_WITHOUT_RANGE",
                f"Medium-confidence assumption {name} has no sensitivity_range.",
                blocking=False,
                variable=name,
                remediation="Add a plausible range and test conclusion stability.",
            )
        )


def _append_assumption_governance_issues(
    state: _AssessmentState,
    assumption: Mapping[str, Any],
) -> None:
    name = str(assumption.get("name") or "unknown")
    approved_by = str(assumption.get("approved_by") or "not_approved")
    if str(assumption.get("source_type") or "gpts_assumption") == "expert_hypothesis" and approved_by != "user":
        state.issues.append(
            _issue(
                "high",
                "EXPERT_HYPOTHESIS_NOT_APPROVED",
                f"Expert hypothesis {name} cannot enter computation without GPTs restructuring and user approval.",
                blocking=True,
                variable=name,
                remediation="GPTs must create a new compute ticket and mark approved_by=user.",
            )
        )
    if not str(assumption.get("invalid_when") or ""):
        state.issues.append(
            _issue(
                "warning",
                "ASSUMPTION_INVALIDATION_RULE_MISSING",
                f"Assumption {name} does not state when it becomes invalid.",
                blocking=False,
                variable=name,
                remediation="Add invalid_when, such as replacement after same-region same-period observed data becomes available.",
            )
        )


def _assess_assumptions(
    state: _AssessmentState,
    assumptions: list[Any],
    policy: Mapping[str, Any],
) -> None:
    for assumption in assumptions:
        if not isinstance(assumption, Mapping):
            continue
        confidence = str(assumption.get("confidence") or "")
        if confidence == "low":
            _append_low_confidence_issues(state, assumption, policy)
        elif confidence == "medium":
            _append_medium_confidence_issues(state, assumption)
        _append_assumption_governance_issues(state, assumption)


def _append_assumption_ratio_issue(
    state: _AssessmentState,
    *,
    variables: list[Any],
    assumptions: list[Any],
    policy: Mapping[str, Any],
) -> float | None:
    total_declared = len(variables)
    assumption_count = state.assumption_variable_count or len(assumptions)
    ratio = assumption_count / total_declared if total_declared else None
    if ratio is None or ratio <= float(policy.get("max_assumption_ratio", 0.5)):
        return ratio
    strict = str(policy.get("enforcement")) == "strict"
    state.issues.append(
        _issue(
            "high" if strict else "warning",
            "ASSUMPTION_RATIO_HIGH",
            f"Assumption-based variables are {ratio:.1%}, above the policy limit.",
            blocking=strict,
            remediation="Use GPTs and the API catalog to replace material assumptions with observed, historical, benchmark, or user-provided data.",
        )
    )
    return ratio


def _status(issues: list[Mapping[str, Any]], assumptions: list[Any]) -> tuple[str, bool, bool]:
    blocking_codes = {str(item["code"]) for item in issues if item.get("blocking")}
    requires_user_approval = bool(APPROVAL_BLOCK_CODES & blocking_codes)
    insufficient = bool(blocking_codes - APPROVAL_BLOCK_CODES)
    warnings = any(item.get("severity") == "warning" for item in issues)
    if insufficient:
        status = "DATA_INSUFFICIENT"
    elif requires_user_approval:
        status = "USER_APPROVAL_REQUIRED"
    elif assumptions:
        status = "DATA_DEGRADED" if warnings else "DATA_READY_WITH_ASSUMPTIONS"
    elif warnings:
        status = "DATA_DEGRADED"
    else:
        status = "DATA_READY"
    return status, status not in {"DATA_INSUFFICIENT", "USER_APPROVAL_REQUIRED"}, requires_user_approval


def _validate_result(result: Mapping[str, Any]) -> None:
    schema = load_json(OUTPUT_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(result),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return
    rendered = "; ".join(
        f"{'.'.join(str(item) for item in error.absolute_path) or '$'}: {error.message}"
        for error in errors[:20]
    )
    raise ValueError(f"invalid preflight output: {rendered}")


def assess(ticket: Mapping[str, Any], policy_document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy_document = policy_document or load_json(POLICY_PATH)
    policy = _merge_policy(ticket, policy_document)
    assumptions, variables = _declared_rows(ticket)
    state = _AssessmentState()

    _assess_variables(state, variables, policy)
    _assess_assumptions(state, assumptions, policy)
    state.issues.extend(_numeric_issues(ticket.get("inputs")))
    assumption_ratio = _append_assumption_ratio_issue(
        state,
        variables=variables,
        assumptions=assumptions,
        policy=policy,
    )
    status, execution_allowed, requires_user_approval = _status(state.issues, assumptions)
    assumption_count = state.assumption_variable_count or len(assumptions)
    result = {
        "schema_version": "compute-preflight-v1",
        "status": status,
        "execution_allowed": execution_allowed,
        "requires_user_approval": requires_user_approval,
        "task_id": str(ticket.get("task_id") or ""),
        "operation": str(ticket.get("operation") or ""),
        "ticket_sha256": canonical_sha(ticket),
        "policy": policy,
        "data_summary": {
            "declared_variable_count": len(variables),
            "observed_variable_count": state.observed_count,
            "assumption_variable_count": assumption_count,
            "assumption_ratio": assumption_ratio,
            "assumption_record_count": len(assumptions),
            "medium_confidence_assumption_count": state.medium_assumption_count,
            "low_confidence_unapproved_count": state.low_unapproved_count,
            "missing_required_variable_count": state.missing_required_count,
            "data_context_declared": bool(variables),
        },
        "issues": state.issues,
        "representative_value_recommendations": state.representative,
        "recommended_operations": sorted(state.recommended_operations),
        "source_priority": list(policy_document.get("source_priority") or []),
        "security": {
            "external_data_fetch_used": False,
            "model_calls": 0,
            "arbitrary_code_used": False,
        },
    }
    _validate_result(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--output", default="compute-preflight.json")
    args = parser.parse_args()
    ticket = load_json(Path(args.ticket))
    if not isinstance(ticket, Mapping):
        raise SystemExit("ticket root must be an object")
    result = assess(ticket)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution_allowed": result["execution_allowed"],
                "issue_count": len(result["issues"]),
                "recommended_operations": result["recommended_operations"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["execution_allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
