#!/usr/bin/env python3
"""Risk-adjusted credibility case derived from runtime facts and typed evidence.

Ticket factor_levels are treated as requested targets only. They never increase the
achieved level by themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "credibility-factor-registry.json"
SCHEMA_PATH = HERE / "credibility-profile.schema.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _derived_assumption_level(assurance: Mapping[str, Any]) -> int:
    if assurance.get("status") == "BLOCKED":
        return 0
    rows = assurance.get("resolved_assumptions") or []
    if not rows:
        return 0
    if assurance.get("status") == "WARN":
        return 1
    if all(row.get("calibration_status") in {"calibrated", "validated"} and row.get("evidence_sha256") for row in rows):
        return 4
    if all(row.get("falsification_test") and row.get("invalid_when") for row in rows):
        return 3
    return 2


def _mechanism_assurance(ticket: Mapping[str, Any]) -> dict[str, Any]:
    rows = ticket.get("mechanism_register") if isinstance(ticket.get("mechanism_register"), list) else []
    rows = [row for row in rows if isinstance(row, Mapping)]
    critical = [row for row in rows if row.get("importance") == "critical"]
    critical_gaps = [
        {
            "mechanism_id": str(row.get("mechanism_id") or ""),
            "inclusion_status": str(row.get("inclusion_status") or ""),
            "rationale": str(row.get("rationale") or ""),
        }
        for row in critical
        if row.get("inclusion_status") in {"excluded", "unknown"}
    ]
    if not rows:
        level = 0
    elif critical_gaps:
        level = 1
    elif not critical:
        level = 2
    elif all(row.get("validation_evidence_sha256") for row in critical):
        level = 4 if all(row.get("falsification_test") for row in critical) else 3
    else:
        level = 2
    return {
        "declared_mechanism_count": len(rows),
        "critical_mechanism_count": len(critical),
        "critical_gap_count": len(critical_gaps),
        "critical_gaps": critical_gaps,
        "derived_conceptual_model_level": level,
    }


def _evidence_levels(profile: Mapping[str, Any], factors: set[str]) -> tuple[dict[str, int], dict[str, int]]:
    rank = {"documentation": 1, "independent_check": 2, "domain_validation": 3, "operational_feedback": 4}
    levels = {factor_id: 0 for factor_id in factors}
    counts = {factor_id: 0 for factor_id in factors}
    for row in profile.get("evidence") or []:
        if not isinstance(row, Mapping) or not row.get("sha256"):
            continue
        evidence_type = str(row.get("evidence_type") or "documentation")
        level = rank.get(evidence_type, 1)
        if evidence_type == "independent_check" and row.get("independent") is not True:
            level = 1
        for factor_id in row.get("factor_ids", []):
            factor_id = str(factor_id)
            if factor_id not in factors:
                raise ValueError(f"credibility evidence references unknown factor: {factor_id}")
            counts[factor_id] += 1
            levels[factor_id] = max(levels[factor_id], level)
    return levels, counts


def build_credibility_case(
    ticket: Mapping[str, Any],
    model: Mapping[str, Any],
    assumption_assurance: Mapping[str, Any],
    experiment_assurance: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _load(REGISTRY_PATH)
    factors = {str(row["id"]): row for row in registry["factors"]}
    quality = ticket.get("quality_profile") if isinstance(ticket.get("quality_profile"), Mapping) else {}
    decision_class = str(quality.get("decision_class") or "formal")
    profile_raw = ticket.get("credibility_profile") if isinstance(ticket.get("credibility_profile"), Mapping) else None
    if profile_raw is not None:
        schema = _load(SCHEMA_PATH)
        Draft202012Validator.check_schema(schema)
        errors = sorted(Draft202012Validator(schema).iter_errors(profile_raw), key=lambda item: list(item.absolute_path))
        if errors:
            first = errors[0]
            where = ".".join(str(item) for item in first.absolute_path) or "$"
            raise ValueError(f"credibility_profile {where}: {first.message}")
    profile = dict(profile_raw or {})
    requested = {str(key): int(value) for key, value in (profile.get("factor_levels") or {}).items()}
    unknown = sorted(set(requested) - set(factors))
    if unknown:
        raise ValueError(f"credibility_profile contains unknown factors: {unknown}")

    evidence_levels, evidence_counts = _evidence_levels(profile, set(factors))
    mechanism = _mechanism_assurance(ticket)
    derived = {
        "assumption_governance": _derived_assumption_level(assumption_assurance),
        "conceptual_model": mechanism["derived_conceptual_model_level"],
        "process_management": 2 if model.get("model_id") and model.get("version") else 1,
    }
    if profile.get("use_statement") and ticket.get("objective"):
        derived["intended_use"] = 1
    if isinstance(ticket.get("data_context"), Mapping) and ticket["data_context"].get("variables"):
        derived["input_pedigree"] = 1
    if model.get("validation_datasets"):
        derived["data_pedigree"] = 2
    if model.get("benchmark_ids"):
        derived["implementation_verification"] = 2
    if experiment_assurance.get("status") == "PASS":
        derived["results_robustness"] = 1

    achieved = {
        factor_id: max(derived.get(factor_id, 0), evidence_levels.get(factor_id, 0))
        for factor_id in factors
    }
    thresholds_cfg = registry["profiles"][decision_class]
    required = {
        factor_id: int(thresholds_cfg["factor_overrides"].get(factor_id, thresholds_cfg["default_required_level"]))
        for factor_id in factors
    }
    if profile.get("model_influence") == "high" and profile.get("decision_consequence") == "high":
        escalation = registry["risk_escalation"]["high_model_influence_and_high_consequence"]
        for factor_id in escalation["factors"]:
            required[factor_id] = min(int(escalation["cap"]), required[factor_id] + int(escalation["increment"]))

    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for factor_id, row in factors.items():
        level = achieved[factor_id]
        threshold = required[factor_id]
        status = "PASS" if level >= threshold else "GAP"
        item = {
            "factor_id": factor_id,
            "domain": row["domain"],
            "requested_level": requested.get(factor_id),
            "achieved_level": level,
            "derived_level": derived.get(factor_id, 0),
            "typed_evidence_level": evidence_levels.get(factor_id, 0),
            "required_level": threshold,
            "evidence_count": evidence_counts[factor_id],
            "status": status,
        }
        rows.append(item)
        if status == "GAP":
            gaps.append(item)

    missing_profile = not profile
    critical_mechanism_block = decision_class == "high_stakes" and mechanism["critical_gap_count"] > 0
    if decision_class == "high_stakes" and (missing_profile or gaps or critical_mechanism_block):
        status = "BLOCKED"
    elif gaps or mechanism["critical_gap_count"]:
        status = "WARN"
    else:
        status = "PASS"
    weakest = min(rows, key=lambda item: (item["achieved_level"] - item["required_level"], item["factor_id"])) if rows else None
    return {
        "schema_version": "compute-credibility-case-v2",
        "status": status,
        "decision_class": decision_class,
        "model_id": str(model.get("model_id") or ""),
        "use_statement": str(profile.get("use_statement") or ticket.get("objective") or ""),
        "model_influence": profile.get("model_influence"),
        "decision_consequence": profile.get("decision_consequence"),
        "single_weighted_score": None,
        "declared_levels_are_targets_only": True,
        "mechanism_assurance": mechanism,
        "factor_assessments": rows,
        "gap_count": len(gaps),
        "gaps": gaps,
        "weakest_factor": weakest,
        "note": "Achieved credibility is derived from runtime facts and typed hashed evidence; ticket declarations cannot self-certify a level.",
    }
