#!/usr/bin/env python3
"""Risk-aware experiment-design assurance for fixed compute operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "experiment-design-registry.json"
SCHEMA_PATH = HERE / "experiment-profile.schema.json"
STOCHASTIC = {
    "monte_carlo",
    "discrete_event_simulation",
    "agent_evolution",
    "markov_simulation",
    "agent_based_simulation",
    "information_diffusion_analysis",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision_class(ticket: Mapping[str, Any]) -> str:
    quality = ticket.get("quality_profile")
    value = quality.get("decision_class") if isinstance(quality, Mapping) else None
    return str(value or "formal")


def _issue(code: str, status: str, blocking: bool, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "status": status,
        "blocking": blocking,
        "message": message,
    }


def _validate_profile(profile: Mapping[str, Any]) -> None:
    schema = _load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(profile),
        key=lambda item: list(item.absolute_path),
    )
    if not errors:
        return
    first = errors[0]
    where = ".".join(str(item) for item in first.absolute_path) or "$"
    raise ValueError(f"experiment_profile {where}: {first.message}")


def _missing_profile_result(operation: str, decision_class: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if operation in STOCHASTIC:
        issues.append(
            _issue(
                "EXPERIMENT_PROFILE_MISSING",
                "WARN",
                decision_class == "high_stakes",
                "Formal stochastic simulation should declare seeds, replications, aggregation and stopping rules.",
            )
        )
    status = (
        "BLOCKED"
        if any(row["blocking"] for row in issues)
        else "WARN"
        if issues
        else "NOT_REQUIRED"
    )
    return {
        "schema_version": "compute-experiment-assurance-v1",
        "status": status,
        "operation": operation,
        "design_id": None,
        "issues": issues,
    }


def _design_issues(
    operation: str,
    design: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if design is None:
        return [
            _issue(
                "UNKNOWN_EXPERIMENT_DESIGN",
                "FAIL",
                True,
                "Experiment design is not registered.",
            )
        ]
    if operation not in design.get("applies_to", []):
        return [
            _issue(
                "EXPERIMENT_DESIGN_MISMATCH",
                "FAIL",
                True,
                "Experiment design is not approved for this operation.",
            )
        ]
    return []


def _stochastic_issues(
    ticket: Mapping[str, Any],
    profile: Mapping[str, Any],
    decision_class: str,
) -> list[dict[str, Any]]:
    inputs = ticket.get("inputs")
    seed = inputs.get("seed") if isinstance(inputs, Mapping) else None
    issues: list[dict[str, Any]] = []
    if profile.get("base_seed") is None and not isinstance(seed, int):
        issues.append(
            _issue(
                "SEED_MISSING",
                "FAIL",
                decision_class != "exploratory",
                "Stochastic formal use requires a fixed seed.",
            )
        )
    replications = int(profile.get("replications", 1))
    minimum = 10 if decision_class == "high_stakes" else 3 if decision_class == "formal" else 1
    if replications < minimum:
        issues.append(
            _issue(
                "INSUFFICIENT_REPLICATIONS",
                "WARN",
                decision_class == "high_stakes",
                f"Declared replications {replications} are below the {minimum} run profile minimum.",
            )
        )
    if not profile.get("stopping_rule"):
        issues.append(
            _issue(
                "STOPPING_RULE_MISSING",
                "WARN",
                decision_class == "high_stakes",
                "Stochastic simulation requires an explicit stopping or precision rule.",
            )
        )
    return issues


def assess_experiment(ticket: Mapping[str, Any]) -> dict[str, Any]:
    registry = _load(REGISTRY_PATH)
    designs = {str(row["id"]): row for row in registry["designs"]}
    operation = str(ticket.get("operation") or "")
    decision_class = _decision_class(ticket)
    raw_profile = ticket.get("experiment_profile")
    if not isinstance(raw_profile, Mapping):
        return _missing_profile_result(operation, decision_class)
    profile = dict(raw_profile)
    _validate_profile(profile)
    design_id = str(profile["design_id"])
    design = designs.get(design_id)
    issues = _design_issues(operation, design)
    if operation in STOCHASTIC:
        issues.extend(_stochastic_issues(ticket, profile, decision_class))
    blocking = [
        row
        for row in issues
        if row.get("blocking") and row.get("status") in {"FAIL", "WARN"}
    ]
    return {
        "schema_version": "compute-experiment-assurance-v1",
        "status": "BLOCKED" if blocking else "WARN" if issues else "PASS",
        "operation": operation,
        "design_id": design_id,
        "registered_controls": list(design.get("controls", [])) if design else [],
        "profile": profile,
        "issues": issues,
    }
