#!/usr/bin/env python3
"""Synchronize ticket, model, and all-operation fixtures for strategic policy."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPERATION = "strategic_policy_analysis"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_ticket_schema() -> None:
    path = HERE / "compute-ticket.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    operations = schema["properties"]["operation"]["enum"]
    if OPERATION not in operations:
        operations.append(OPERATION)
        operations.sort()
    write_json(path, schema)


def update_model_registry() -> None:
    path = HERE / "model-registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in registry["models"] if row.get("operation") != OPERATION]
    rows.append({
        "model_id": "strategic_policy_analysis-registered-v1",
        "operation": OPERATION,
        "mode": "*",
        "version": "1.0.0",
        "maturity": "controlled-preview",
        "engineering_maturity": "production",
        "evidence_maturity": "experimental",
        "risk_tier": "high",
        "calibration_supported": False,
        "allowed_backends": [],
        "intended_use": [
            "Bounded offline strategic, policy, entity-resolution, graph, and structured intelligence decision support."
        ],
        "prohibited_use": [
            "Personal identity targeting or surveillance",
            "Ticket-supplied code, agents, solver programs, files, URLs, or remote ontologies",
            "Autonomous high-stakes decisions",
            "Claims of equivalence to proprietary consulting or intelligence systems"
        ],
        "theoretical_basis": "Registered game theory, decision analysis, policy counterfactual, entity resolution, graph analytics, and structured analytic techniques.",
        "known_failure_conditions": [
            "Inputs outside registered bounds",
            "Unvalidated causal or behavioral assumptions",
            "Insufficient provenance or lawful-basis evidence",
            "Use outside the fixed mode contract"
        ],
        "assurance_owner": "compute-quality-gate",
        "revalidation_trigger": [
            "dependency_change",
            "data_drift",
            "performance_degradation",
            "scope_change"
        ]
    })
    registry["models"] = sorted(rows, key=lambda row: str(row.get("operation") or ""))
    write_json(path, registry)


def update_all_operation_fixture() -> None:
    path = HERE / "validate_all_operations.py"
    text = path.read_text(encoding="utf-8")
    if f'"{OPERATION}":' in text:
        return
    anchor = '        "sector_model_analysis": {"mode": "nash_bimatrix_equilibria", "row_payoffs": [[3, 0], [5, 1]], "column_payoffs": [[3, 5], [0, 1]]},\n'
    addition = (
        '        "strategic_policy_analysis": {"mode": "issue_tree_coverage", '
        '"root": "profit decline", "branches": ['
        '{"name": "revenue", "weight": 0.6, "evidence_count": 2}, '
        '{"name": "cost", "weight": 0.4, "evidence_count": 1}]},\n'
    )
    if anchor not in text:
        raise RuntimeError("all-operation fixture anchor not found")
    path.write_text(text.replace(anchor, anchor + addition, 1), encoding="utf-8")


def main() -> int:
    update_ticket_schema()
    update_model_registry()
    update_all_operation_fixture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
