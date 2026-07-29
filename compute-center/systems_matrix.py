#!/usr/bin/env python3
"""Validate and expose the systems-theory computation matrix.

This module is control-plane only. It does not execute models, install packages,
access a network, or choose a model from free-form user text.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MATRIX_PATH = HERE / "systems-computation-matrix.json"
CAPABILITIES_PATH = HERE / "compute-capabilities.json"
ALLOWED_STAGES = {
    "formulate",
    "estimate_or_calibrate",
    "simulate_or_optimize",
    "stress_and_validate",
    "realized_outcome_feedback",
}
ALLOWED_GATES = {
    "input_quality",
    "assumption_register",
    "constraint_feasibility",
    "identifiability",
    "uncertainty",
    "calibration",
    "stress_test",
    "external_validation",
    "feedback_monitoring",
}
ALLOWED_SYSTEM_LEVELS = {
    "observation",
    "mechanism",
    "state-estimation",
    "state-evolution",
    "decision",
    "validation",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path.name}")
    return value


def capability_operation_ids() -> set[str]:
    catalog = _load_object(CAPABILITIES_PATH)
    rows = catalog.get("operations")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("compute capability catalog has no operations")
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("compute capability row must be an object")
        operation = str(row.get("id") or "")
        if not operation or operation in result:
            raise RuntimeError(f"invalid or duplicate capability operation: {operation}")
        result.add(operation)
    declared_count = catalog.get("operation_count")
    if declared_count != len(result):
        raise RuntimeError(
            f"compute capability operation_count mismatch: declared={declared_count} actual={len(result)}"
        )
    return result


def load_systems_matrix() -> dict[str, Any]:
    matrix = _load_object(MATRIX_PATH)
    if matrix.get("schema_version") != "compute-systems-matrix-v1":
        raise RuntimeError("invalid systems computation matrix schema")
    if matrix.get("runtime_network_policy") != "deny":
        raise RuntimeError("systems computation matrix must remain offline")

    stages = matrix.get("lifecycle_stages")
    if not isinstance(stages, list) or set(stages) != ALLOWED_STAGES or len(stages) != len(ALLOWED_STAGES):
        raise RuntimeError("systems lifecycle stages are incomplete or duplicated")

    gate_catalog = matrix.get("required_gate_catalog")
    if not isinstance(gate_catalog, Mapping) or set(gate_catalog) != ALLOWED_GATES:
        raise RuntimeError("systems gate catalog is incomplete")

    routes = matrix.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        raise RuntimeError("systems computation matrix has no routes")
    operation_ids = capability_operation_ids()
    route_ids = {str(name) for name in routes}
    if route_ids != operation_ids:
        missing = sorted(operation_ids - route_ids)
        extra = sorted(route_ids - operation_ids)
        raise RuntimeError(f"systems matrix/catalog mismatch: missing={missing} extra={extra}")

    for operation, raw in routes.items():
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"systems route must be an object: {operation}")
        problem_class = str(raw.get("problem_class") or "")
        system_level = str(raw.get("system_level") or "")
        feedback_structure = str(raw.get("feedback_structure") or "")
        gates = raw.get("required_gates")
        if not problem_class or not feedback_structure:
            raise RuntimeError(f"systems route is incomplete: {operation}")
        if system_level not in ALLOWED_SYSTEM_LEVELS:
            raise RuntimeError(f"invalid system level for {operation}: {system_level}")
        if not isinstance(gates, list) or len(gates) != len(set(gates)):
            raise RuntimeError(f"required_gates must be a unique array: {operation}")
        unknown = sorted(set(gates) - ALLOWED_GATES)
        if unknown:
            raise RuntimeError(f"unknown systems gate for {operation}: {unknown}")
        if system_level in {"decision", "state-evolution"} and "assumption_register" not in gates:
            raise RuntimeError(f"dynamic or decision route lacks assumption governance: {operation}")
        if system_level == "decision" and "stress_test" not in gates:
            raise RuntimeError(f"decision route lacks stress testing: {operation}")
    return matrix


def route_for_operation(operation: str) -> dict[str, Any]:
    matrix = load_systems_matrix()
    routes = matrix["routes"]
    if operation not in routes:
        raise RuntimeError(f"operation is not mapped in systems matrix: {operation}")
    route = copy.deepcopy(dict(routes[operation]))
    route["schema_version"] = "compute-systems-route-v1"
    route["operation"] = operation
    route["lifecycle_stages"] = list(matrix["lifecycle_stages"])
    route["runtime_network_policy"] = "deny"
    route["arbitrary_code_allowed"] = False
    return route


def route_for_ticket(ticket: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(ticket.get("operation") or "")
    if not operation:
        raise RuntimeError("ticket operation is required for systems routing")
    route = route_for_operation(operation)
    inputs = ticket.get("inputs")
    route["mode"] = str(inputs.get("mode") or "") or None if isinstance(inputs, Mapping) else None
    profile = ticket.get("quality_profile")
    decision_class = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    route["decision_class"] = decision_class
    route["publication_gate"] = {
        "exploratory": "method-and-limitations-disclosure",
        "formal": "frozen-benchmark-or-holdout-required",
        "high-stakes": "independent-validation-and-human-approval-required",
    }.get(decision_class, "unknown-decision-class-blocked")
    return route


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    route = sub.add_parser("route")
    route.add_argument("--operation", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        matrix = load_systems_matrix()
        print(json.dumps({
            "status": "PASS",
            "schema_version": matrix["schema_version"],
            "operation_count": len(matrix["routes"]),
            "gate_count": len(matrix["required_gate_catalog"]),
        }, ensure_ascii=False))
        return 0
    print(json.dumps(route_for_operation(args.operation), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
