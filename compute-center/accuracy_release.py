#!/usr/bin/env python3
"""Apply accuracy evidence to the institutional decision release report."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_LEVEL = {"experimental": 0, "controlled-preview": 1, "production": 2, "decision-grade": 3}
_REQUIRED = {"exploratory": 0, "formal": 2, "high_stakes": 3}


def apply_accuracy_release_gate(
    report: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(report)
    checks = [dict(row) for row in value.get("checks", [])]
    decision_class = str(value.get("decision_class") or "formal")
    maturity = result.get("maturity_assessment") if isinstance(result.get("maturity_assessment"), Mapping) else {}
    evidence_maturity = str(maturity.get("evidence_maturity") or "experimental")
    required_level = _REQUIRED.get(decision_class, 2)
    maturity_pass = _LEVEL.get(evidence_maturity, -1) >= required_level
    checks.append({
        "code": "EVIDENCE_MATURITY",
        "status": "PASS" if maturity_pass else "FAIL",
        "blocking": decision_class in {"formal", "high_stakes"} and not maturity_pass,
        "message": f"Evidence maturity={evidence_maturity}; required for {decision_class}={next(name for name, level in _LEVEL.items() if level == required_level)}.",
    })

    calibration = result.get("calibration_assurance") if isinstance(result.get("calibration_assurance"), Mapping) else {}
    if calibration.get("requested"):
        executed = calibration.get("execution_status") == "EXECUTED"
        checks.append({
            "code": "CALIBRATION_EXECUTION",
            "status": "PASS" if executed else "FAIL",
            "blocking": not executed,
            "message": "Requested calibration must execute, not merely validate a profile.",
        })

    experiment = result.get("experiment_execution") if isinstance(result.get("experiment_execution"), Mapping) else {}
    if experiment.get("requested"):
        passed = experiment.get("status") == "PASS" and experiment.get("executed_replications") == experiment.get("requested_replications")
        checks.append({
            "code": "REPLICATION_EXECUTION",
            "status": "PASS" if passed else "FAIL",
            "blocking": decision_class == "high_stakes" and not passed,
            "message": f"Requested replications={experiment.get('requested_replications')}; executed={experiment.get('executed_replications')}.",
        })

    validation = result.get("validation_assurance") if isinstance(result.get("validation_assurance"), Mapping) else {}
    if validation.get("requested"):
        passed = validation.get("status") == "PASS"
        checks.append({
            "code": "OUT_OF_SAMPLE_VALIDATION",
            "status": "PASS" if passed else "FAIL",
            "blocking": decision_class in {"formal", "high_stakes"} and not passed,
            "message": f"Executable validation status={validation.get('status')}.",
        })

    blocking = [row for row in checks if row.get("blocking") and row.get("status") == "FAIL"]
    warnings = [row for row in checks if row.get("status") == "WARN"]
    nonblocking_failures = [row for row in checks if not row.get("blocking") and row.get("status") == "FAIL"]
    release_status = "DECISION_BLOCKED" if blocking else "DECISION_CONDITIONAL" if warnings or nonblocking_failures else "DECISION_RELEASED"
    constraints = dict(value.get("constraints") or {})
    constraints["formal_decision_use_allowed"] = release_status == "DECISION_RELEASED"
    constraints["must_complete_accuracy_evidence"] = not maturity_pass
    constraints["required_evidence_maturity"] = next(name for name, level in _LEVEL.items() if level == required_level)
    constraints["observed_evidence_maturity"] = evidence_maturity
    value.update({
        "schema_version": "compute-quality-report-v3",
        "checks": checks,
        "release_status": release_status,
        "decision_grade": release_status == "DECISION_RELEASED",
        "constraints": constraints,
        "accuracy_gate": {
            "engineering_maturity": maturity.get("engineering_maturity"),
            "evidence_maturity": evidence_maturity,
            "required_evidence_maturity": constraints["required_evidence_maturity"],
            "calibration_status": calibration.get("execution_status"),
            "experiment_status": experiment.get("status"),
            "validation_status": validation.get("status"),
        },
    })
    return value
