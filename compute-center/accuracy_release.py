#!/usr/bin/env python3
"""Apply accuracy evidence to the institutional decision release report."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_LEVEL = {
    "experimental": 0,
    "controlled-preview": 1,
    "production": 2,
    "decision-grade": 3,
}
_REQUIRED = {"exploratory": 0, "formal": 2, "high_stakes": 3}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _level_name(level: int) -> str:
    return next(name for name, value in _LEVEL.items() if value == level)


def _maturity_check(
    decision_class: str,
    maturity: Mapping[str, Any],
) -> tuple[dict[str, Any], bool, str, str]:
    observed = str(maturity.get("evidence_maturity") or "experimental")
    required = _level_name(_REQUIRED.get(decision_class, 2))
    passed = _LEVEL.get(observed, -1) >= _LEVEL[required]
    return (
        {
            "code": "EVIDENCE_MATURITY",
            "status": "PASS" if passed else "FAIL",
            "blocking": decision_class in {"formal", "high_stakes"} and not passed,
            "message": (
                f"Evidence maturity={observed}; required for "
                f"{decision_class}={required}."
            ),
        },
        passed,
        observed,
        required,
    )


def _requested_checks(
    decision_class: str,
    calibration: Mapping[str, Any],
    experiment: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if calibration.get("requested"):
        executed = calibration.get("execution_status") == "EXECUTED"
        checks.append(
            {
                "code": "CALIBRATION_EXECUTION",
                "status": "PASS" if executed else "FAIL",
                "blocking": not executed,
                "message": "Requested calibration must execute, not merely validate a profile.",
            }
        )
    if experiment.get("requested"):
        passed = (
            experiment.get("status") == "PASS"
            and experiment.get("executed_replications")
            == experiment.get("requested_replications")
        )
        checks.append(
            {
                "code": "REPLICATION_EXECUTION",
                "status": "PASS" if passed else "FAIL",
                "blocking": decision_class == "high_stakes" and not passed,
                "message": (
                    f"Requested replications={experiment.get('requested_replications')}; "
                    f"executed={experiment.get('executed_replications')}."
                ),
            }
        )
    if validation.get("requested"):
        passed = validation.get("status") == "PASS"
        checks.append(
            {
                "code": "OUT_OF_SAMPLE_VALIDATION",
                "status": "PASS" if passed else "FAIL",
                "blocking": decision_class in {"formal", "high_stakes"} and not passed,
                "message": f"Executable validation status={validation.get('status')}.",
            }
        )
    return checks


def _release_status(checks: list[Mapping[str, Any]]) -> str:
    blocking = any(
        row.get("blocking") and row.get("status") == "FAIL" for row in checks
    )
    conditional = any(
        row.get("status") == "WARN"
        or (not row.get("blocking") and row.get("status") == "FAIL")
        for row in checks
    )
    if blocking:
        return "DECISION_BLOCKED"
    return "DECISION_CONDITIONAL" if conditional else "DECISION_RELEASED"


def apply_accuracy_release_gate(
    report: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(report)
    checks = [dict(row) for row in value.get("checks", [])]
    decision_class = str(value.get("decision_class") or "formal")
    maturity = _mapping(result.get("maturity_assessment"))
    calibration = _mapping(result.get("calibration_assurance"))
    experiment = _mapping(result.get("experiment_execution"))
    validation = _mapping(result.get("validation_assurance"))

    maturity_row, maturity_pass, observed, required = _maturity_check(
        decision_class, maturity
    )
    checks.append(maturity_row)
    checks.extend(
        _requested_checks(decision_class, calibration, experiment, validation)
    )
    release_status = _release_status(checks)
    constraints = dict(value.get("constraints") or {})
    constraints.update(
        {
            "formal_decision_use_allowed": release_status == "DECISION_RELEASED",
            "must_complete_accuracy_evidence": not maturity_pass,
            "required_evidence_maturity": required,
            "observed_evidence_maturity": observed,
        }
    )
    value.update(
        {
            "schema_version": "compute-quality-report-v3",
            "checks": checks,
            "release_status": release_status,
            "decision_grade": release_status == "DECISION_RELEASED",
            "constraints": constraints,
            "accuracy_gate": {
                "engineering_maturity": maturity.get("engineering_maturity"),
                "evidence_maturity": observed,
                "required_evidence_maturity": required,
                "calibration_status": calibration.get("execution_status"),
                "experiment_status": experiment.get("status"),
                "validation_status": validation.get("status"),
            },
        }
    )
    return value
