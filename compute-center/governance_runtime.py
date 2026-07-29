#!/usr/bin/env python3
"""Install executable model, assumption, experiment and quality governance."""
from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from accuracy_runtime import (
    derive_evidence_maturity,
    execute_calibration,
    execute_experiment,
    execute_validation,
)
from assumption_library import assess_assumptions
from assumption_runtime import build_assumption_plan
from constraint_engine import ConstraintViolation, enforce_constraints, independent_post_check
from credibility_engine import build_credibility_case
from experiment_assurance import assess_experiment
from library_runtime import resolve_library_selection
from model_governance import (
    lifecycle_status,
    registered_model,
    validate_ticket_governance,
)


def _constraint_paths(row: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("field", "left", "right"):
        if isinstance(row.get(key), str):
            paths.append(str(row[key]))
    if isinstance(row.get("fields"), list):
        paths.extend(str(item) for item in row["fields"])
    return paths


def _pre_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    hard_constraints = [
        row
        for row in profile.get("hard_constraints", [])
        if isinstance(row, Mapping)
        and not any(path.startswith("results.") for path in _constraint_paths(row))
    ]
    return {
        "hard_constraints": hard_constraints,
        "soft_constraints": list(profile.get("soft_constraints") or []),
        "independent_post_check": True,
    }


def _default_constraint_report(phase: str) -> dict[str, Any]:
    return {
        "schema_version": "compute-constraint-report-v1",
        "phase": phase,
        "status": "NOT_PROVIDED",
        "hard_constraint_count": 0,
        "violation_count": 0,
        "checks": [],
        "violations": [],
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class _GovernedExecution:
    """One isolated governance transaction around one fixed compute operation."""

    def __init__(
        self,
        compute_runner: Any,
        original: Callable[[dict[str, Any], Path], dict[str, Any]],
        ticket: dict[str, Any],
        output_dir: Path,
    ) -> None:
        self.compute_runner = compute_runner
        self.original = original
        self.ticket = ticket
        self.output_dir = output_dir
        self.governance = validate_ticket_governance(ticket)
        self.original_inputs = dict(_mapping(ticket.get("inputs")))
        self.operation = str(ticket.get("operation") or "")
        self.mode = str(self.original_inputs.get("mode") or "") or None
        self.model = registered_model(self.operation, self.mode)
        self.lifecycle = lifecycle_status(self.model)
        self.decision_class = str(
            _mapping(ticket.get("quality_profile")).get("decision_class") or "formal"
        )
        self.constraint_profile = (
            dict(ticket["constraint_profile"])
            if isinstance(ticket.get("constraint_profile"), Mapping)
            else None
        )

    def _error(self, message: str, exc: Exception | None = None) -> Exception:
        error = self.compute_runner.ComputeError(message)
        if exc is not None:
            error.__cause__ = exc
        return error

    def _validate_lifecycle(self) -> None:
        if self.lifecycle["status"] in {"MODEL_SUSPENDED", "MODEL_RETIRED"}:
            raise self._error(
                f"{self.lifecycle['status']}: {self.model['model_id']}"
            )

    def _resolve_library(self) -> dict[str, Any]:
        try:
            return resolve_library_selection(self.ticket)
        except Exception as exc:
            raise self._error(f"LIBRARY_SELECTION_FAILED: {exc}", exc)

    def _preuse_assurance(self) -> dict[str, Any]:
        library_selection = self._resolve_library()
        assumption_plan = build_assumption_plan(self.ticket)
        assumption_assurance = assess_assumptions(self.ticket)
        experiment_assurance = assess_experiment(self.ticket)
        if self.decision_class == "high_stakes" and not library_selection.get("strategy"):
            raise self._error("HIGH_STAKES_STRATEGY_NOT_SELECTED")
        blocked = {
            "ASSUMPTION_PLAN_UNRESOLVED": assumption_plan["status"] == "BLOCKED",
            "ASSUMPTION_GOVERNANCE_FAILED": assumption_assurance["status"] == "BLOCKED",
            "EXPERIMENT_GOVERNANCE_FAILED": experiment_assurance["status"] == "BLOCKED",
        }
        for code, active in blocked.items():
            if active:
                raise self._error(code)
        credibility = build_credibility_case(
            self.ticket,
            self.model,
            assumption_assurance,
            experiment_assurance,
        )
        if self.decision_class == "high_stakes" and credibility["status"] == "BLOCKED":
            raise self._error("CREDIBILITY_PREUSE_FAILED")
        return {
            "library_selection": library_selection,
            "assumption_plan": assumption_plan,
            "assumption_assurance": assumption_assurance,
            "experiment_assurance": experiment_assurance,
            "credibility": credibility,
        }

    def _precheck(
        self,
        inputs: Mapping[str, Any],
        *,
        phase: str,
        failure_code: str,
    ) -> dict[str, Any]:
        if self.constraint_profile is None:
            return _default_constraint_report(phase)
        subject = {"inputs": dict(inputs), **dict(inputs)}
        try:
            report = enforce_constraints(subject, _pre_profile(self.constraint_profile))
        except ConstraintViolation as exc:
            raise self._error(f"{failure_code}: {exc}", exc)
        report["phase"] = phase
        return report

    def _handler(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        handler = self.compute_runner.OPERATIONS.get(self.operation)
        if not callable(handler):
            raise self._error(
                f"registered operation handler not found: {self.operation}"
            )
        return handler

    def _calibrate(
        self,
        handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        profile = self.ticket.get("calibration_profile")
        profile = profile if isinstance(profile, Mapping) else None
        try:
            calibrated_inputs, calibration = execute_calibration(
                handler, self.original_inputs, profile
            )
        except Exception as exc:
            raise self._error(f"CALIBRATION_EXECUTION_FAILED: {exc}", exc)
        return dict(calibrated_inputs), calibration

    def _execute_experiment(
        self,
        handler: Callable[[Mapping[str, Any]], dict[str, Any]],
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile = self.ticket.get("experiment_profile")
        profile = profile if isinstance(profile, Mapping) else None
        try:
            report = execute_experiment(handler, inputs, profile)
        except Exception as exc:
            raise self._error(f"EXPERIMENT_EXECUTION_FAILED: {exc}", exc)
        if report.get("status") == "FAIL" and self.decision_class == "high_stakes":
            raise self._error("EXPERIMENT_PRECISION_FAILED")
        return report

    def _execute_operation(self, calibrated_inputs: Mapping[str, Any]) -> dict[str, Any]:
        executed_ticket = copy.deepcopy(self.ticket)
        executed_ticket["inputs"] = dict(calibrated_inputs)
        return self.original(executed_ticket, self.output_dir)

    def _postcheck(
        self,
        inputs: Mapping[str, Any],
        results: Mapping[str, Any],
    ) -> dict[str, Any]:
        phase = "post_solver_independent_recheck"
        if self.constraint_profile is None:
            return _default_constraint_report(phase)
        subject = {
            "inputs": dict(inputs),
            "results": dict(results),
            **dict(inputs),
            **dict(results),
        }
        report = independent_post_check(subject, self.constraint_profile)
        if report["status"] != "PASS":
            identifiers = ", ".join(
                str(item.get("id") or "<unnamed>")
                for item in report["violations"]
            )
            raise self._error("CONSTRAINT_POSTCHECK_FAILED: " + identifiers)
        return report

    def _validate_results(self, results: Mapping[str, Any]) -> dict[str, Any]:
        profile = self.ticket.get("validation_profile")
        profile = profile if isinstance(profile, Mapping) else None
        try:
            return execute_validation(results, profile)
        except Exception as exc:
            raise self._error(f"VALIDATION_EXECUTION_FAILED: {exc}", exc)

    def _maturity(
        self,
        calibration: Mapping[str, Any],
        validation: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> dict[str, Any]:
        report = derive_evidence_maturity(
            self.model,
            calibration,
            validation,
            experiment,
            _mapping(self.ticket.get("quality_profile")),
        )
        if (
            self.decision_class == "high_stakes"
            and report["evidence_maturity"] != "decision-grade"
        ):
            raise self._error("EVIDENCE_MATURITY_INSUFFICIENT_FOR_HIGH_STAKES")
        return report

    def _attach_reports(
        self,
        result: dict[str, Any],
        reports: Mapping[str, Any],
    ) -> None:
        result.update(
            {
                "model_governance": {
                    **self.governance,
                    "lifecycle": self.lifecycle,
                },
                "library_selection": reports["library_selection"],
                "assumption_plan": reports["assumption_plan"],
                "assumption_assurance": reports["assumption_assurance"],
                "experiment_assurance": reports["experiment_assurance"],
                "experiment_execution": reports["experiment_execution"],
                "credibility_case": reports["credibility"],
                "constraint_assurance": {
                    "pre_execution_original_inputs": reports["pre_report"],
                    "pre_execution_calibrated_inputs": reports["calibrated_pre_report"],
                    "post_execution": reports["post_report"],
                },
                "calibration_assurance": reports["calibration"],
                "validation_assurance": reports["validation"],
                "maturity_assessment": reports["maturity"],
            }
        )

    def _write_reports(self, result: Mapping[str, Any], reports: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "compute-model-governance.json": result["model_governance"],
            "compute-library-selection.json": reports["library_selection"],
            "compute-assumption-plan.json": reports["assumption_plan"],
            "compute-assumption-assurance.json": reports["assumption_assurance"],
            "compute-experiment-assurance.json": reports["experiment_assurance"],
            "compute-experiment-execution.json": reports["experiment_execution"],
            "compute-credibility-case.json": reports["credibility"],
            "compute-constraint-precheck.json": reports["pre_report"],
            "compute-constraint-calibrated-precheck.json": reports[
                "calibrated_pre_report"
            ],
            "compute-constraint-postcheck.json": reports["post_report"],
            "compute-calibration-assurance.json": reports["calibration"],
            "compute-validation-result.json": reports["validation"],
            "compute-maturity-assessment.json": reports["maturity"],
            "compute-result.json": result,
        }
        for filename, value in files.items():
            self.compute_runner._write_json(self.output_dir / filename, value)

    def _update_audit(self, reports: Mapping[str, Any]) -> None:
        audit_path = self.output_dir / "compute-audit.json"
        if not audit_path.is_file():
            return
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            return
        audit.update(
            {
                "model_id": self.governance["model_id"],
                "model_engineering_maturity": reports["maturity"][
                    "engineering_maturity"
                ],
                "model_evidence_maturity": reports["maturity"]["evidence_maturity"],
                "lifecycle_status": self.lifecycle["status"],
                "library_selection_status": reports["library_selection"]["status"],
                "library_selection_sha256": reports["library_selection"][
                    "selection_sha256"
                ],
                "assumption_plan_status": reports["assumption_plan"]["status"],
                "assumption_assurance_status": reports["assumption_assurance"][
                    "status"
                ],
                "assumption_snapshot_sha256": reports["assumption_assurance"][
                    "resolved_snapshot_sha256"
                ],
                "experiment_assurance_status": reports["experiment_assurance"][
                    "status"
                ],
                "experiment_execution_status": reports["experiment_execution"][
                    "status"
                ],
                "executed_replications": reports["experiment_execution"].get(
                    "executed_replications", 0
                ),
                "calibration_execution_status": reports["calibration"][
                    "execution_status"
                ],
                "validation_status": reports["validation"]["status"],
                "credibility_case_status": reports["credibility"]["status"],
                "credibility_gap_count": reports["credibility"]["gap_count"],
                "constraint_original_precheck_status": reports["pre_report"]["status"],
                "constraint_calibrated_precheck_status": reports[
                    "calibrated_pre_report"
                ]["status"],
                "constraint_postcheck_status": reports["post_report"]["status"],
            }
        )
        self.compute_runner._write_json(audit_path, audit)

    def run(self) -> dict[str, Any]:
        self._validate_lifecycle()
        reports = self._preuse_assurance()
        reports["pre_report"] = self._precheck(
            self.original_inputs,
            phase="pre_execution_original_inputs",
            failure_code="CONSTRAINT_PRECHECK_FAILED",
        )
        handler = self._handler()
        calibrated_inputs, reports["calibration"] = self._calibrate(handler)
        reports["calibrated_pre_report"] = self._precheck(
            calibrated_inputs,
            phase="pre_execution_calibrated_inputs",
            failure_code="CALIBRATED_CONSTRAINT_PRECHECK_FAILED",
        )
        reports["experiment_execution"] = self._execute_experiment(
            handler, calibrated_inputs
        )
        result = self._execute_operation(calibrated_inputs)
        results = _mapping(result.get("results"))
        reports["post_report"] = self._postcheck(calibrated_inputs, results)
        reports["validation"] = self._validate_results(results)
        reports["maturity"] = self._maturity(
            reports["calibration"],
            reports["validation"],
            reports["experiment_execution"],
        )
        self._attach_reports(result, reports)
        self._write_reports(result, reports)
        self._update_audit(reports)
        self.compute_runner._write_manifest(self.output_dir)
        return result


def install(compute_runner: Any) -> None:
    if getattr(compute_runner, "_governance_runtime_installed", False):
        return
    original = compute_runner.run_ticket

    def governed_run_ticket(
        ticket: dict[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        return _GovernedExecution(compute_runner, original, ticket, output_dir).run()

    compute_runner.run_ticket = governed_run_ticket
    compute_runner._governance_runtime_installed = True
