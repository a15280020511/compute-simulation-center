#!/usr/bin/env python3
"""Install executable model, library, assumption, experiment, validation and constraint governance."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from accuracy_runtime import derive_evidence_maturity, execute_calibration, execute_experiment, execute_validation
from assumption_library import assess_assumptions
from assumption_runtime import build_assumption_plan
from constraint_engine import ConstraintViolation, enforce_constraints, independent_post_check
from credibility_engine import build_credibility_case
from experiment_assurance import assess_experiment
from library_runtime import resolve_library_selection
from model_governance import lifecycle_status, registered_model, validate_ticket_governance


def _constraint_paths(row: Mapping[str, Any]) -> list[str]:
    paths = []
    for key in ("field", "left"):
        if isinstance(row.get(key), str):
            paths.append(str(row[key]))
    if isinstance(row.get("right"), str):
        paths.append(str(row["right"]))
    if isinstance(row.get("fields"), list):
        paths.extend(str(item) for item in row["fields"])
    return paths


def _pre_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in profile.get("hard_constraints", []) if isinstance(row, Mapping) and not any(path.startswith("results.") for path in _constraint_paths(row))]
    return {"hard_constraints": rows, "soft_constraints": list(profile.get("soft_constraints") or []), "independent_post_check": True}


def _default_constraint_report(phase: str) -> dict[str, Any]:
    return {"schema_version": "compute-constraint-report-v1", "phase": phase, "status": "NOT_PROVIDED", "hard_constraint_count": 0, "violation_count": 0, "checks": [], "violations": []}


def install(compute_runner: Any) -> None:
    if getattr(compute_runner, "_governance_runtime_installed", False):
        return
    original = compute_runner.run_ticket

    def governed_run_ticket(ticket: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        governance = validate_ticket_governance(ticket)
        original_inputs = ticket.get("inputs") if isinstance(ticket.get("inputs"), Mapping) else {}
        operation = str(ticket.get("operation") or "")
        mode = str(original_inputs.get("mode") or "") or None
        model = registered_model(operation, mode)
        lifecycle = lifecycle_status(model)
        if lifecycle["status"] in {"MODEL_SUSPENDED", "MODEL_RETIRED"}:
            raise compute_runner.ComputeError(f"{lifecycle['status']}: {model['model_id']}")

        try:
            library_selection = resolve_library_selection(ticket)
        except Exception as exc:
            raise compute_runner.ComputeError(f"LIBRARY_SELECTION_FAILED: {exc}") from exc

        assumption_plan = build_assumption_plan(ticket)
        assumption_assurance = assess_assumptions(ticket)
        experiment_assurance = assess_experiment(ticket)
        decision_class = str(((ticket.get("quality_profile") or {}).get("decision_class") if isinstance(ticket.get("quality_profile"), Mapping) else None) or "formal")
        if decision_class == "high_stakes" and not library_selection.get("strategy"):
            raise compute_runner.ComputeError("HIGH_STAKES_STRATEGY_NOT_SELECTED")
        if assumption_plan["status"] == "BLOCKED":
            raise compute_runner.ComputeError("ASSUMPTION_PLAN_UNRESOLVED")
        if assumption_assurance["status"] == "BLOCKED":
            raise compute_runner.ComputeError("ASSUMPTION_GOVERNANCE_FAILED")
        if experiment_assurance["status"] == "BLOCKED":
            raise compute_runner.ComputeError("EXPERIMENT_GOVERNANCE_FAILED")

        preliminary_credibility = build_credibility_case(ticket, model, assumption_assurance, experiment_assurance)
        if decision_class == "high_stakes" and preliminary_credibility["status"] == "BLOCKED":
            raise compute_runner.ComputeError("CREDIBILITY_PREUSE_FAILED")

        profile = ticket.get("constraint_profile") if isinstance(ticket.get("constraint_profile"), Mapping) else None
        pre_report = _default_constraint_report("pre_execution_original_inputs")
        if profile is not None:
            try:
                pre_report = enforce_constraints({"inputs": dict(original_inputs), **dict(original_inputs)}, _pre_profile(profile))
                pre_report["phase"] = "pre_execution_original_inputs"
            except ConstraintViolation as exc:
                raise compute_runner.ComputeError(f"CONSTRAINT_PRECHECK_FAILED: {exc}") from exc

        handler = compute_runner.OPERATIONS.get(operation)
        if not callable(handler):
            raise compute_runner.ComputeError(f"registered operation handler not found: {operation}")
        calibration_profile = ticket.get("calibration_profile") if isinstance(ticket.get("calibration_profile"), Mapping) else None
        try:
            calibrated_inputs, calibration = execute_calibration(handler, original_inputs, calibration_profile)
        except Exception as exc:
            raise compute_runner.ComputeError(f"CALIBRATION_EXECUTION_FAILED: {exc}") from exc

        calibrated_pre_report = _default_constraint_report("pre_execution_calibrated_inputs")
        if profile is not None:
            try:
                calibrated_pre_report = enforce_constraints({"inputs": dict(calibrated_inputs), **dict(calibrated_inputs)}, _pre_profile(profile))
                calibrated_pre_report["phase"] = "pre_execution_calibrated_inputs"
            except ConstraintViolation as exc:
                raise compute_runner.ComputeError(f"CALIBRATED_CONSTRAINT_PRECHECK_FAILED: {exc}") from exc

        experiment_profile = ticket.get("experiment_profile") if isinstance(ticket.get("experiment_profile"), Mapping) else None
        try:
            experiment_execution = execute_experiment(handler, calibrated_inputs, experiment_profile)
        except Exception as exc:
            raise compute_runner.ComputeError(f"EXPERIMENT_EXECUTION_FAILED: {exc}") from exc
        if experiment_execution.get("status") == "FAIL" and decision_class == "high_stakes":
            raise compute_runner.ComputeError("EXPERIMENT_PRECISION_FAILED")

        executed_ticket = copy.deepcopy(ticket)
        executed_ticket["inputs"] = calibrated_inputs
        result = original(executed_ticket, output_dir)
        results = result.get("results") if isinstance(result.get("results"), Mapping) else {}

        post_report = _default_constraint_report("post_solver_independent_recheck")
        if profile is not None:
            post_report = independent_post_check({"inputs": dict(calibrated_inputs), "results": dict(results), **dict(calibrated_inputs), **dict(results)}, profile)
            if post_report["status"] != "PASS":
                identifiers = ", ".join(str(item.get("id") or "<unnamed>") for item in post_report["violations"])
                raise compute_runner.ComputeError("CONSTRAINT_POSTCHECK_FAILED: " + identifiers)

        validation_profile = ticket.get("validation_profile") if isinstance(ticket.get("validation_profile"), Mapping) else None
        try:
            validation = execute_validation(results, validation_profile)
        except Exception as exc:
            raise compute_runner.ComputeError(f"VALIDATION_EXECUTION_FAILED: {exc}") from exc

        quality_profile = ticket.get("quality_profile") if isinstance(ticket.get("quality_profile"), Mapping) else {}
        maturity = derive_evidence_maturity(model, calibration, validation, experiment_execution, quality_profile)
        if decision_class == "high_stakes" and maturity["evidence_maturity"] != "decision-grade":
            raise compute_runner.ComputeError("EVIDENCE_MATURITY_INSUFFICIENT_FOR_HIGH_STAKES")

        result["model_governance"] = {**governance, "lifecycle": lifecycle}
        result["library_selection"] = library_selection
        result["assumption_plan"] = assumption_plan
        result["assumption_assurance"] = assumption_assurance
        result["experiment_assurance"] = experiment_assurance
        result["experiment_execution"] = experiment_execution
        result["credibility_case"] = preliminary_credibility
        result["constraint_assurance"] = {"pre_execution_original_inputs": pre_report, "pre_execution_calibrated_inputs": calibrated_pre_report, "post_execution": post_report}
        result["calibration_assurance"] = calibration
        result["validation_assurance"] = validation
        result["maturity_assessment"] = maturity

        output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "compute-model-governance.json": result["model_governance"],
            "compute-library-selection.json": library_selection,
            "compute-assumption-plan.json": assumption_plan,
            "compute-assumption-assurance.json": assumption_assurance,
            "compute-experiment-assurance.json": experiment_assurance,
            "compute-experiment-execution.json": experiment_execution,
            "compute-credibility-case.json": preliminary_credibility,
            "compute-constraint-precheck.json": pre_report,
            "compute-constraint-calibrated-precheck.json": calibrated_pre_report,
            "compute-constraint-postcheck.json": post_report,
            "compute-calibration-assurance.json": calibration,
            "compute-validation-result.json": validation,
            "compute-maturity-assessment.json": maturity,
            "compute-result.json": result,
        }
        for filename, value in files.items():
            compute_runner._write_json(output_dir / filename, value)

        audit_path = output_dir / "compute-audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(audit, dict):
                audit.update({
                    "model_id": governance["model_id"],
                    "model_engineering_maturity": maturity["engineering_maturity"],
                    "model_evidence_maturity": maturity["evidence_maturity"],
                    "lifecycle_status": lifecycle["status"],
                    "library_selection_status": library_selection["status"],
                    "library_selection_sha256": library_selection["selection_sha256"],
                    "assumption_plan_status": assumption_plan["status"],
                    "assumption_assurance_status": assumption_assurance["status"],
                    "assumption_snapshot_sha256": assumption_assurance["resolved_snapshot_sha256"],
                    "experiment_assurance_status": experiment_assurance["status"],
                    "experiment_execution_status": experiment_execution["status"],
                    "executed_replications": experiment_execution.get("executed_replications", 0),
                    "calibration_execution_status": calibration["execution_status"],
                    "validation_status": validation["status"],
                    "credibility_case_status": preliminary_credibility["status"],
                    "credibility_gap_count": preliminary_credibility["gap_count"],
                    "constraint_original_precheck_status": pre_report["status"],
                    "constraint_calibrated_precheck_status": calibrated_pre_report["status"],
                    "constraint_postcheck_status": post_report["status"],
                })
                compute_runner._write_json(audit_path, audit)
        compute_runner._write_manifest(output_dir)
        return result

    compute_runner.run_ticket = governed_run_ticket
    compute_runner._governance_runtime_installed = True
