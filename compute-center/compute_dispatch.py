#!/usr/bin/env python3
"""Production dispatcher with strict preflight, executable accuracy controls and diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_runner  # noqa: E402
from accuracy_release import apply_accuracy_release_gate  # noqa: E402
from compute_diagnostics import write_failure, write_success  # noqa: E402
from compute_preflight import assess as assess_preflight  # noqa: E402
from compute_preflight import canonical_sha as canonical_preflight_sha  # noqa: E402
from material_package_validation import validate_material_package  # noqa: E402
from quality_gate import build_quality_report  # noqa: E402
from relay_contracts import build_data_gap_plan, build_expert_review_request  # noqa: E402
from tool_registry import register_into  # noqa: E402

register_into(compute_runner.OPERATIONS)

run_ticket = compute_runner.run_ticket
validate_ticket = compute_runner.validate_ticket
ComputeError = compute_runner.ComputeError
OPERATIONS = compute_runner.OPERATIONS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_ticket(ticket_path: Path) -> dict[str, Any]:
    if ticket_path.stat().st_size > compute_runner.MAX_TICKET_BYTES:
        raise ComputeError(f"ticket exceeds {compute_runner.MAX_TICKET_BYTES} bytes")
    value = json.loads(ticket_path.read_text(encoding="utf-8"), parse_constant=compute_runner._reject_constant)
    if not isinstance(value, dict):
        raise ComputeError("ticket root must be an object")
    return value


def _refresh_manifest(output_dir: Path) -> None:
    compute_runner._write_manifest(output_dir)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _validate_relayed_material(output_dir: Path, ticket_path: Path) -> dict[str, Any]:
    package_root = ticket_path.parent / "compute-input" / "material-package"
    receipt_path = output_dir / "compute-material-package-validation.json"
    if not package_root.exists():
        receipt = {
            "schema_version": "compute-material-package-validation-receipt-v2",
            "status": "NOT_PRESENT",
            "package_root": "compute-input/material-package",
            "numeric_execution_blocked": False,
            "reason": "No GPTs-relayed material package was supplied for this ticket.",
            "runtime_network_used": False,
            "direct_center_connection": False,
            "model_calls": 0,
        }
        _write_json(receipt_path, receipt)
        print(json.dumps({
            "material_package_status": "NOT_PRESENT",
            "material_package_validation_file": receipt_path.name,
        }, ensure_ascii=False))
        return receipt
    if not package_root.is_dir():
        raise ComputeError("MATERIAL_PACKAGE_INVALID: compute-input/material-package is not a directory")
    try:
        receipt = validate_material_package(package_root)
    except Exception as exc:
        raise ComputeError(f"MATERIAL_PACKAGE_BLOCKED:{type(exc).__name__}:{exc}") from exc
    _write_json(receipt_path, receipt)
    print(json.dumps({
        "material_package_status": receipt["status"],
        "material_package_id": receipt["package_id"],
        "material_package_sha256": receipt["content_sha256"],
        "material_package_validation_file": receipt_path.name,
        "runtime_network_used": False,
    }, ensure_ascii=False))
    return receipt


def _write_preflight(output_dir: Path, ticket: Mapping[str, Any]) -> dict[str, Any]:
    result = assess_preflight(ticket)
    _write_json(output_dir / "compute-preflight.json", result)
    data_gap_plan = build_data_gap_plan(ticket, result)
    _write_json(output_dir / "compute-data-gap-plan.json", data_gap_plan)
    print(json.dumps({
        "preflight_status": result["status"],
        "execution_allowed": result["execution_allowed"],
        "enforcement": result["policy"].get("enforcement"),
        "issue_count": len(result["issues"]),
        "recommended_operations": result["recommended_operations"],
        "relay_owner": data_gap_plan["relay_owner"],
        "center_direct_contact_allowed": False,
    }, ensure_ascii=False))
    return result


def _attach_governance_to_transfer(
    output_dir: Path,
    result: dict[str, Any],
    preflight: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    ticket: Mapping[str, Any] | None = None,
) -> None:
    preflight_sha256 = canonical_preflight_sha(preflight)
    quality_sha256 = canonical_preflight_sha(quality_report)
    blocking_count = sum(bool(item.get("blocking")) for item in preflight.get("issues", []))
    result["preflight"] = {
        "status": preflight["status"],
        "execution_allowed": preflight["execution_allowed"],
        "requires_user_approval": preflight["requires_user_approval"],
        "issue_count": len(preflight["issues"]),
        "blocking_issue_count": blocking_count,
        "recommended_operations": list(preflight["recommended_operations"]),
        "data_summary": dict(preflight["data_summary"]),
        "preflight_sha256": preflight_sha256,
        "file": "compute-preflight.json",
    }
    result["quality"] = {
        "decision_class": quality_report["decision_class"],
        "release_status": quality_report["release_status"],
        "decision_grade": quality_report["decision_grade"],
        "formal_decision_use_allowed": quality_report["constraints"]["formal_decision_use_allowed"],
        "must_collect_more_feedback": quality_report["constraints"].get("must_collect_more_feedback", False),
        "must_recalibrate_before_reuse": quality_report["constraints"].get("must_recalibrate_before_reuse", False),
        "must_complete_accuracy_evidence": quality_report["constraints"].get("must_complete_accuracy_evidence", False),
        "required_evidence_maturity": quality_report["constraints"].get("required_evidence_maturity"),
        "observed_evidence_maturity": quality_report["constraints"].get("observed_evidence_maturity"),
        "quality_report_sha256": quality_sha256,
        "quality_report_file": "compute-quality-report.json",
        "calibration_feedback_file": "compute-calibration.json",
        "accuracy_release_file": "compute-accuracy-release.json",
    }
    result["relay_contract"] = {
        "sole_relay": "gpts-usage-center",
        "direct_center_contact_allowed": False,
        "data_gap_plan_file": "compute-data-gap-plan.json",
        "expert_review_request_file": "compute-expert-review-request.json",
        "material_package_validation_file": "compute-material-package-validation.json",
    }
    if isinstance(ticket, Mapping):
        pipeline = ticket.get("pipeline")
        if isinstance(pipeline, Mapping):
            result["pipeline"] = dict(pipeline)
            result["pipeline_sha256"] = canonical_preflight_sha(pipeline)
        data_context = ticket.get("data_context")
        if isinstance(data_context, Mapping):
            result["data_context"] = dict(data_context)
            result["data_context_sha256"] = canonical_preflight_sha(data_context)

    package_payload = {key: value for key, value in result.items() if key != "package_sha256"}
    result["package_sha256"] = canonical_preflight_sha(package_payload)
    _write_json(output_dir / "compute-result.json", result)

    audit_path = output_dir / "compute-audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(audit, dict):
            audit.update({
                "preflight_status": preflight["status"],
                "preflight_sha256": preflight_sha256,
                "quality_release_status": quality_report["release_status"],
                "quality_report_sha256": quality_sha256,
                "formal_decision_use_allowed": quality_report["constraints"]["formal_decision_use_allowed"],
                "required_evidence_maturity": quality_report["constraints"].get("required_evidence_maturity"),
                "observed_evidence_maturity": quality_report["constraints"].get("observed_evidence_maturity"),
                "package_sha256": result["package_sha256"],
                "sole_relay": "gpts-usage-center",
                "direct_center_contact_allowed": False,
                "material_package_validation_file": "compute-material-package-validation.json",
            })
            if "pipeline_sha256" in result:
                audit["pipeline_sha256"] = result["pipeline_sha256"]
            if "data_context_sha256" in result:
                audit["data_context_sha256"] = result["data_context_sha256"]
            _write_json(audit_path, audit)

    summary_path = output_dir / "compute-summary.md"
    if summary_path.is_file():
        summary = summary_path.read_text(encoding="utf-8").rstrip()
        summary += (
            f"\n- Preflight status: `{preflight['status']}`"
            f"\n- Preflight enforcement: `{preflight['policy'].get('enforcement')}`"
            f"\n- Preflight SHA256: `{preflight_sha256}`"
            f"\n- Decision class: `{quality_report['decision_class']}`"
            f"\n- Decision release: `{quality_report['release_status']}`"
            f"\n- Decision grade: `{quality_report['decision_grade']}`"
            f"\n- Evidence maturity: `{quality_report['constraints'].get('observed_evidence_maturity')}`"
            f"\n- Required evidence maturity: `{quality_report['constraints'].get('required_evidence_maturity')}`"
            f"\n- Quality report SHA256: `{quality_sha256}`"
            f"\n- Package SHA256: `{result['package_sha256']}`"
            "\n- Material package validation: `compute-material-package-validation.json`"
            "\n- Sole cross-center relay: `gpts-usage-center`"
            "\n- Direct center contact: `forbidden`"
        )
        pipeline = result.get("pipeline")
        if isinstance(pipeline, Mapping):
            summary += f"\n- Pipeline ID: `{pipeline.get('pipeline_id') or 'unknown'}`\n- Stage ID: `{pipeline.get('stage_id') or 'unknown'}`"
        summary += "\n"
        summary_path.write_text(summary, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one independent deterministic compute ticket.")
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--output-dir", default="compute-artifacts")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ticket_path = Path(args.ticket)
    started_at = _utc_now()
    started = time.perf_counter()
    stage = "load_ticket"
    ticket: Mapping[str, Any] | None = None

    try:
        ticket = _load_ticket(ticket_path)
        stage = "validate_ticket"
        validate_ticket(ticket)
        stage = "material_package_validation"
        material_receipt = _validate_relayed_material(output_dir, ticket_path)
        stage = "data_preflight"
        preflight = _write_preflight(output_dir, ticket)
        if not preflight["execution_allowed"]:
            raise ComputeError(f"PREFLIGHT_BLOCKED:{preflight['status']}; GPTs must resolve data gaps or obtain required user approval")
        stage = "execute_operation"
        result = run_ticket(dict(ticket), output_dir)
        result["material_package"] = {
            "status": material_receipt["status"],
            "validation_file": "compute-material-package-validation.json",
            "content_sha256": material_receipt.get("content_sha256"),
            "gpts_relay_attestation": material_receipt.get("gpts_relay_attestation"),
            "runtime_network_used": False,
        }
        stage = "quality_gate"
        quality_report = build_quality_report(ticket, result, preflight)
        quality_report = apply_accuracy_release_gate(quality_report, result)
        _write_json(output_dir / "compute-calibration.json", quality_report["calibration_feedback"])
        _write_json(output_dir / "compute-accuracy-release.json", quality_report["accuracy_gate"])
        _write_json(output_dir / "compute-quality-report.json", quality_report)
        _attach_governance_to_transfer(output_dir, result, preflight, quality_report, ticket)
        expert_request = build_expert_review_request(ticket, result, preflight)
        expert_request["quality_release_status"] = quality_report["release_status"]
        expert_request["formal_decision_use_allowed"] = quality_report["constraints"]["formal_decision_use_allowed"]
        expert_request["observed_evidence_maturity"] = quality_report["constraints"].get("observed_evidence_maturity")
        expert_request["required_evidence_maturity"] = quality_report["constraints"].get("required_evidence_maturity")
        expert_request["quality_report_file"] = "compute-quality-report.json"
        expert_request["material_package_validation_file"] = "compute-material-package-validation.json"
        _write_json(output_dir / "compute-expert-review-request.json", expert_request)
        elapsed = time.perf_counter() - started
        write_success(output_dir, ticket=ticket, result=result, elapsed_seconds=elapsed)
        stage = "write_manifest"
        _refresh_manifest(output_dir)
        print(json.dumps({
            "status": "COMPUTE_COMPLETED",
            "release_status": quality_report["release_status"],
            "formal_decision_use_allowed": quality_report["constraints"]["formal_decision_use_allowed"],
            "evidence_maturity": quality_report["constraints"].get("observed_evidence_maturity"),
            "material_package_status": material_receipt["status"],
            "output_dir": str(output_dir),
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - started
        write_failure(output_dir, exc=exc, stage=stage, started_at=started_at, elapsed_seconds=elapsed, ticket_path=ticket_path, ticket=ticket)
        try:
            _refresh_manifest(output_dir)
        except Exception as manifest_exc:
            print(f"SECONDARY ERROR [manifest] {type(manifest_exc).__name__}: {manifest_exc}", file=sys.stderr)
            traceback.print_exception(type(manifest_exc), manifest_exc, manifest_exc.__traceback__, file=sys.stderr)
        print(f"ERROR [{type(exc).__name__}] stage={stage}: {exc}", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
