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

from capability_manager import load_registered_operations, requirements_for_ticket, runtime_plan
from governance_runtime import install as install_governance_runtime
from json_normalization import wrap_operation


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
    return requirements_for_ticket(ticket)


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
        print(json.dumps({"status": "PASS", "manager_version": 2, "registered_operations": sorted(operations)}))
        return 0
    ticket = json.loads(Path(args.ticket).read_text(encoding="utf-8"))
    if not isinstance(ticket, Mapping):
        raise SystemExit("ticket must be a JSON object")
    if args.command == "requirements":
        for requirement in requirements_for_ticket(ticket):
            print(requirement)
        return 0
    print(json.dumps(runtime_plan(ticket), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
