#!/usr/bin/env python3
"""Backward-compatible facade over Compute Capability Manager V2."""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from capability_environment import core_requirement_files, metadata_for_ticket, prepare as prepare_capability_environment
from capability_manager import load_registered_operations, requirements_for_ticket, runtime_plan
from dynamic_family_router import FAMILY_BY_OPERATION, family_runtime_metadata
from governance_runtime import install as install_governance_runtime
from json_normalization import wrap_operation
from systems_matrix import load_systems_matrix, route_for_ticket

HERE = Path(__file__).resolve().parent
DYNAMIC_PIPELINE_ID = "dynamic-auto-v1"
DYNAMIC_STAGE_ID = "dynamic"
DYNAMIC_REQUIREMENT = HERE / "requirements-ortools.txt"


def _dynamic_orchestration_requested(ticket: Mapping[str, Any]) -> bool:
    pipeline = ticket.get("pipeline")
    return bool(
        isinstance(pipeline, Mapping)
        and str(pipeline.get("pipeline_id") or "") == DYNAMIC_PIPELINE_ID
        and str(pipeline.get("stage_id") or "") == DYNAMIC_STAGE_ID
    )


def register_into(target: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> None:
    for name, handler in load_registered_operations().items():
        if name == "causal_policy_evaluation":
            # The registered causal implementation remains the algorithm source of truth,
            # but production dispatch crosses the fixed gateway so DoWhy's older SciPy
            # constraint cannot contaminate the core numerical runtime.
            from causal_policy_gateway import causal_policy_evaluation

            handler = causal_policy_evaluation
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


def _core_managed_requirements(ticket: Mapping[str, Any]) -> list[str]:
    return core_requirement_files(ticket, requirements_for_ticket(ticket))


def requirement_files_for_ticket(ticket: Mapping[str, Any]) -> list[str]:
    base_requirements = _core_managed_requirements(ticket)
    if _dynamic_orchestration_requested(ticket):
        # Validate the structured family before installing dependencies so unsupported
        # dynamic operations fail closed at dependency planning time. Dynamic families
        # always need OR-Tools. Dependency-incompatible packs such as causal-policy are
        # excluded here and prepared by capability_environment.py in a fixed venv.
        family_runtime_metadata(ticket)
        if not DYNAMIC_REQUIREMENT.is_file():
            raise RuntimeError("dynamic orchestration requirement bundle is missing")
        selected = [str(DYNAMIC_REQUIREMENT), *base_requirements]
    else:
        selected = list(base_requirements)

    unique: list[str] = []
    for raw in selected:
        path = Path(raw)
        if not path.is_file():
            raise RuntimeError(f"managed requirement bundle is missing: {path.name}")
        rendered = str(path)
        if rendered not in unique:
            unique.append(rendered)
    return unique


def managed_runtime_plan(ticket: Mapping[str, Any]) -> dict[str, Any]:
    isolated_environment = metadata_for_ticket(ticket)
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
            "requirements": requirement_files_for_ticket(ticket),
            "isolated_environment": isolated_environment,
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
    plan["requirements"] = requirement_files_for_ticket(ticket)
    plan["isolated_environment"] = isolated_environment
    plan["systems_route"] = route_for_ticket(ticket)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    requirements = sub.add_parser("requirements")
    requirements.add_argument("--ticket", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--ticket", required=True)
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
            },
            "isolated_environments": {
                operation: metadata_for_ticket({"operation": operation})
                for operation in ("causal_policy_evaluation",)
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
    environment_preparation = prepare_capability_environment(ticket)
    plan = managed_runtime_plan(ticket)
    plan["capability_environment_preparation"] = environment_preparation
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
