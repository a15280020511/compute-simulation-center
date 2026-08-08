#!/usr/bin/env python3
"""Backward-compatible facade over Compute Capability Manager V2."""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from capability_manager import load_registered_operations, requirements_for_ticket, runtime_plan
from dynamic_family_router import FAMILY_BY_OPERATION, family_runtime_metadata
from governance_runtime import install as install_governance_runtime
from json_normalization import wrap_operation
from systems_matrix import load_systems_matrix, route_for_ticket

HERE = Path(__file__).resolve().parent
DYNAMIC_PIPELINE_ID = "dynamic-auto-v1"
DYNAMIC_STAGE_ID = "dynamic"
DYNAMIC_REQUIREMENT = HERE / "requirements-ortools.txt"
REQUIREMENT_RE = re.compile(r"^requirements-[a-z0-9-]+\.txt$")
ALLOWED_DYNAMIC_PYTHON = {"3.12", "3.13"}
DEFAULT_DYNAMIC_PYTHON = "3.12"


def _dynamic_orchestration_requested(ticket: Mapping[str, Any]) -> bool:
    pipeline = ticket.get("pipeline")
    return bool(
        isinstance(pipeline, Mapping)
        and str(pipeline.get("pipeline_id") or "") == DYNAMIC_PIPELINE_ID
        and str(pipeline.get("stage_id") or "") == DYNAMIC_STAGE_ID
    )


def _dynamic_family_requirements(metadata: Mapping[str, Any]) -> list[str]:
    rows = metadata.get("requirements", [])
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise RuntimeError("dynamic family requirements must be an array")
    result: list[str] = []
    for raw in rows:
        name = str(raw)
        if not REQUIREMENT_RE.fullmatch(name):
            raise RuntimeError(f"invalid dynamic family requirement name: {name}")
        candidate = HERE / name
        if candidate.parent != HERE or not candidate.is_file():
            raise RuntimeError(f"dynamic family requirement bundle is missing: {name}")
        rendered = str(candidate)
        if rendered not in result:
            result.append(rendered)
    return result


def _dynamic_family_python(metadata: Mapping[str, Any]) -> str:
    version = str(metadata.get("python_version") or DEFAULT_DYNAMIC_PYTHON)
    if version not in ALLOWED_DYNAMIC_PYTHON:
        raise RuntimeError(f"dynamic family Python runtime is not allowlisted: {version}")
    return version


def register_into(target: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> None:
    for name, handler in load_registered_operations().items():
        managed_handler = wrap_operation(handler)
        if name in target and target[name] is not managed_handler:
            raise RuntimeError(f"conflicting compute operation registration: {name}")
        target[name] = managed_handler
    try:
        import compute_runner
        schema = copy.deepcopy(compute_runner.SCHEMA)
        schema["properties"]["operation"]["enum"] = sorted(target)
        Draft202012Validator.check_schema(schema)
        compute_runner.VALIDATOR = Draft202012Validator(schema)
        install_governance_runtime(compute_runner)
        if getattr(compute_runner, "_governance_runtime_installed", False) is not True:
            raise RuntimeError("governance runtime installation was not confirmed")
    except Exception as exc:
        raise RuntimeError("GOVERNANCE_RUNTIME_INSTALLATION_FAILED") from exc


def requirement_files_for_ticket(ticket: Mapping[str, Any]) -> list[str]:
    if _dynamic_orchestration_requested(ticket):
        # Resolve the structured family before installing any optional bundle so
        # unsupported dynamic operations fail closed at dependency planning time.
        family = family_runtime_metadata(ticket)
        if not DYNAMIC_REQUIREMENT.is_file():
            raise RuntimeError("dynamic orchestration requirement bundle is missing")
        result = [str(DYNAMIC_REQUIREMENT)]
        for requirement in _dynamic_family_requirements(family):
            if requirement not in result:
                result.append(requirement)
        return result
    return requirements_for_ticket(ticket)


def managed_runtime_plan(ticket: Mapping[str, Any]) -> dict[str, Any]:
    if _dynamic_orchestration_requested(ticket):
        family = family_runtime_metadata(ticket)
        return {
            "schema_version": "compute-runtime-plan-v2",
            "operation": str(ticket.get("operation") or ""),
            "mode": None,
            "capability_pack": "dynamic-orchestration",
            "dynamic_family": family["family"],
            "dynamic_entry_contract": family["entry_contract"],
            "dynamic_policy_file": family["policy_file"],
            "dynamic_graph_file": family["graph_file"],
            "python_version": _dynamic_family_python(family),
            "requirements": requirement_files_for_ticket(ticket),
            "network_policy": "deny",
            "deterministic": True,
            "limits": {
                "max_seconds": 120,
                "max_memory_mb": 4096,
                "max_stages": 8,
            },
            "maturity": "controlled-preview",
            "rollback": {
                "stable_module": "pipeline_engine",
                "strategy": "disable-dynamic-stage-and-git-revert",
            },
            "managed": True,
            "arbitrary_code_allowed": False,
            "arbitrary_requirements_allowed": False,
            "dynamic_operation_discovery_allowed": False,
            "automatic_parallel_execution": False,
            "selection_engine": "ortools-cp-sat",
            "graph_engine": "networkx",
            "systems_route": route_for_ticket(ticket),
        }
    plan = runtime_plan(ticket)
    plan["systems_route"] = route_for_ticket(ticket)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    requirements = sub.add_parser("requirements")
    requirements.add_argument("--ticket", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--ticket", required=True)
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "validate":
        operations = load_registered_operations()
        matrix = load_systems_matrix()
        if not DYNAMIC_REQUIREMENT.is_file():
            raise SystemExit("dynamic orchestration requirement bundle is missing")
        print(json.dumps({
            "status": "PASS",
            "manager_version": 2,
            "registered_operations": sorted(operations),
            "systems_matrix_schema": matrix["schema_version"],
            "systems_matrix_operation_count": len(matrix["routes"]),
            "dynamic_orchestration": {
                "pipeline_id": DYNAMIC_PIPELINE_ID,
                "stage_id": DYNAMIC_STAGE_ID,
                "requirement": DYNAMIC_REQUIREMENT.name,
                "selection_engine": "ortools-cp-sat",
                "graph_engine": "networkx",
                "families": dict(sorted(FAMILY_BY_OPERATION.items())),
                "python_allowlist": sorted(ALLOWED_DYNAMIC_PYTHON),
            },
        }, ensure_ascii=False))
        return 0
    ticket = json.loads(Path(args.ticket).read_text(encoding="utf-8"))
    if not isinstance(ticket, Mapping):
        raise SystemExit("ticket must be a JSON object")
    if args.command == "requirements":
        for requirement in requirement_files_for_ticket(ticket):
            print(requirement)
        return 0
    print(json.dumps(managed_runtime_plan(ticket), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
