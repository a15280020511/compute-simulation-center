#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for bounded observational causal evaluation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from dynamic_family_engine import (
    FamilyDefinition,
    StructuredFamilyError,
    build_plan,
    decision_class,
    load_family_spec,
    run_structured_family,
)
from dynamic_family_router import resolve_dynamic_family

HERE = Path(__file__).resolve().parent
FAMILY = "causal-policy"
DECLARED_OPERATION = "causal_policy_evaluation"
REQUIRED_STAGE_ID = "primary_effect"
PRIMARY_MODES = {"backdoor_adjustment", "propensity_weighting"}
DEFINITION = FamilyDefinition(
    family=FAMILY,
    declared_operation=DECLARED_OPERATION,
    required_stage_id=REQUIRED_STAGE_ID,
    policy_path=HERE / "dynamic-causal-policy.json",
    graph_path=HERE / "dynamic-causal-capability-graph.json",
    policy_schema_version="compute-dynamic-causal-policy-v1",
    graph_schema_version="compute-dynamic-causal-capability-graph-v1",
    maximum_stages=4,
    required_safety={"primary_effect_stage_required": True},
)


class DynamicCausalPolicyError(StructuredFamilyError):
    """Raised when a causal-policy dynamic request is unsafe or invalid."""


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicCausalPolicyError(f"{name} must be an array")
    return value


def _numeric_vector(value: Any, name: str, *, binary: bool = False) -> list[float]:
    raw = _sequence(value, name)
    result: list[float] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DynamicCausalPolicyError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise DynamicCausalPolicyError(f"{name}[{index}] must be finite")
        if binary and number not in {0.0, 1.0}:
            raise DynamicCausalPolicyError(f"{name}[{index}] must be 0 or 1")
        result.append(number)
    return result


def _signals(ticket: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicCausalPolicyError("ticket was not routed to causal-policy family")
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicCausalPolicyError("ticket inputs must be an object")
    mode = str(inputs.get("mode") or "")
    if mode not in PRIMARY_MODES:
        raise DynamicCausalPolicyError(
            "dynamic causal-policy family currently admits only backdoor_adjustment or propensity_weighting as the primary estimator"
        )
    treatment = _numeric_vector(inputs.get("treatment"), "inputs.treatment", binary=True)
    outcome = _numeric_vector(inputs.get("outcome"), "inputs.outcome")
    if len(treatment) != len(outcome):
        raise DynamicCausalPolicyError("treatment and outcome must have equal length")
    minimum_rows = int(policy["selection_policy"]["minimum_dynamic_rows"])
    if len(treatment) < minimum_rows:
        raise DynamicCausalPolicyError(f"causal dynamic family requires at least {minimum_rows} rows")
    treated = sum(1 for item in treatment if item == 1.0)
    control = len(treatment) - treated
    if treated == 0 or control == 0:
        raise DynamicCausalPolicyError("causal dynamic family requires both treatment and control observations")

    confounders = inputs.get("confounders")
    if not isinstance(confounders, Mapping) or not confounders:
        raise DynamicCausalPolicyError("causal dynamic family requires at least one declared confounder")
    for raw_name, raw_values in confounders.items():
        values = _numeric_vector(raw_values, f"inputs.confounders.{raw_name}")
        if len(values) != len(treatment):
            raise DynamicCausalPolicyError("all confounders must match treatment length")

    context = inputs.get("dynamic_context")
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise DynamicCausalPolicyError("inputs.dynamic_context must be an object")
    decision = decision_class(ticket)
    placebo_minimum = int(policy["selection_policy"]["placebo_minimum_rows"])
    placebo_eligible = len(treatment) >= placebo_minimum
    if decision == "high_stakes" and not placebo_eligible:
        raise DynamicCausalPolicyError(
            f"high-stakes causal dynamic family requires at least {placebo_minimum} rows for mandatory placebo refutation"
        )
    signals = {
        "causal_input_valid": True,
        "confounders_present": True,
        "placebo_eligible": placebo_eligible,
        "causal_diagnostics_requested": context.get("causal_diagnostics") is True,
        "estimator_cross_check_requested": context.get("estimator_cross_check") is True,
        "placebo_requested": context.get("placebo_refutation") is True,
        "formal_decision": decision in {"formal", "high_stakes"},
        "high_stakes": decision == "high_stakes",
    }
    features = {
        "observation_count": len(treatment),
        "treated_count": treated,
        "control_count": control,
        "confounder_count": len(confounders),
        "primary_estimator": mode,
        "placebo_eligible": placebo_eligible,
        "decision_class": decision,
        "causal_diagnostics_requested": signals["causal_diagnostics_requested"],
        "estimator_cross_check_requested": signals["estimator_cross_check_requested"],
        "placebo_requested": signals["placebo_requested"],
    }
    return signals, features


def plan_dynamic_causal_policy(ticket: Mapping[str, Any]) -> dict[str, Any]:
    try:
        policy, graph = load_family_spec(DEFINITION)
        signals, features = _signals(ticket, policy)
        return build_plan(
            DEFINITION,
            policy,
            graph,
            signals,
            features,
            family_reason=(
                "causal-policy family was selected from explicit operation=causal_policy_evaluation, "
                "an admitted observational estimator mode, and structured treatment/outcome/confounder inputs"
            ),
        )
    except StructuredFamilyError as exc:
        if isinstance(exc, DynamicCausalPolicyError):
            raise
        raise DynamicCausalPolicyError(str(exc)) from exc


def _effect(result: Mapping[str, Any], name: str) -> float:
    raw = result.get("effect")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DynamicCausalPolicyError(f"{name}.effect must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise DynamicCausalPolicyError(f"{name}.effect must be finite")
    return value


def causal_quality_gate(
    stage_results: Mapping[str, dict[str, Any]],
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    policy, _graph = load_family_spec(DEFINITION)
    primary = stage_results.get(REQUIRED_STAGE_ID)
    if not isinstance(primary, Mapping):
        raise DynamicCausalPolicyError("primary causal result is missing")
    primary_effect = _effect(primary, REQUIRED_STAGE_ID)
    primary_allowed = primary.get("causal_claim_allowed") is True
    reasons: list[str] = []
    warnings: list[str] = []
    causal_claim_allowed = primary_allowed
    if not primary_allowed:
        reasons.append("primary estimator did not pass its mode-specific causal identification/diagnostic gate")

    alternate_payload: dict[str, Any] | None = None
    alternate = stage_results.get("alternate_estimate")
    if isinstance(alternate, Mapping):
        alternate_effect = _effect(alternate, "alternate_estimate")
        same_sign = primary_effect == 0.0 or alternate_effect == 0.0 or (primary_effect > 0) == (alternate_effect > 0)
        relative_difference = abs(alternate_effect - primary_effect) / max(abs(primary_effect), 1e-12)
        threshold = float(policy["quality_gate"]["alternate_relative_difference_warn"])
        alternate_payload = {
            "primary_effect": primary_effect,
            "alternate_effect": alternate_effect,
            "effect_sign_consistent": same_sign,
            "relative_difference": relative_difference,
            "warning_threshold": threshold,
            "alternate_causal_claim_allowed": alternate.get("causal_claim_allowed") is True,
        }
        if not same_sign:
            causal_claim_allowed = False
            reasons.append("primary and alternate estimators disagree on effect sign")
        elif relative_difference > threshold:
            warnings.append("primary and alternate estimators materially differ in effect magnitude")
        if alternate.get("causal_claim_allowed") is not True:
            warnings.append("alternate estimator did not pass its own causal-claim gate")

    placebo_payload: dict[str, Any] | None = None
    placebo = stage_results.get("placebo_refutation")
    if isinstance(placebo, Mapping):
        passed = placebo.get("refutation_passed") is True
        placebo_payload = {
            "refutation_passed": passed,
            "empirical_p_value": placebo.get("empirical_p_value"),
            "repetitions": placebo.get("repetitions"),
        }
        if policy["quality_gate"]["require_placebo_pass_when_selected"] and not passed:
            causal_claim_allowed = False
            reasons.append("selected placebo refutation did not pass")

    status = "PASS"
    if not causal_claim_allowed:
        status = "REJECT_CAUSAL_CLAIM"
    elif warnings:
        status = "WARN"
    return {
        "status": status,
        "causal_claim_allowed": causal_claim_allowed,
        "primary_mode": primary.get("mode"),
        "primary_effect": primary_effect,
        "selected_validations": [
            stage_id for stage_id in plan["stage_order"] if stage_id != REQUIRED_STAGE_ID
        ],
        "alternate_estimator_check": alternate_payload,
        "placebo_refutation": placebo_payload,
        "reasons": reasons,
        "warnings": warnings,
        "interpretation_boundary": (
            "A successful computation is not itself permission for causal language; causal claims require this quality gate to allow them."
        ),
    }


def run_dynamic_causal_policy_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicCausalPolicyError("ticket is not an admitted causal-policy dynamic request")
    plan = plan_dynamic_causal_policy(ticket)
    try:
        return run_structured_family(
            ticket,
            plan,
            output_dir,
            operations,
            quality_gate=causal_quality_gate,
        )
    except StructuredFamilyError as exc:
        if isinstance(exc, DynamicCausalPolicyError):
            raise
        raise DynamicCausalPolicyError(str(exc)) from exc
