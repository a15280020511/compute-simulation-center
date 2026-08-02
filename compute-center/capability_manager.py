#!/usr/bin/env python3
"""Compute Capability Manager V2.

The manager is repository-controlled. It validates capability packs, resolves the
single pinned dependency bundle for a ticket, exposes bounded resource policy,
and produces a deterministic runtime plan. It never installs arbitrary packages
or imports ticket-supplied modules.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "tool-registry.json"
THINK_TANK_REGISTRY_PATH = HERE / "think-tank-mode-registry.json"
MODULE_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
REQUIREMENT_RE = re.compile(r"^requirements-[a-z0-9-]+\.txt$")
MODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
ALLOWED_NETWORK_POLICIES = {"deny"}
ALLOWED_MATURITY = {"production", "controlled-preview"}


def _merge_think_tank_registry(value: dict[str, Any]) -> dict[str, Any]:
    if not THINK_TANK_REGISTRY_PATH.is_file():
        return value
    extension = json.loads(THINK_TANK_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(extension, Mapping) or extension.get("schema_version") != "think-tank-mode-registry-v1":
        raise RuntimeError("invalid think-tank mode registry schema")
    if extension.get("network_policy") != "deny" or extension.get("arbitrary_code_allowed") is not False:
        raise RuntimeError("think-tank mode registry violates offline or arbitrary-code policy")
    target_id = str(extension.get("target_group") or "")
    groups = value.get("groups")
    if not isinstance(groups, list):
        raise RuntimeError("compute tool registry has no groups")
    matches = [group for group in groups if isinstance(group, Mapping) and group.get("id") == target_id]
    if len(matches) != 1:
        raise RuntimeError("think-tank mode registry target group is missing or ambiguous")
    target = matches[0]
    modes = extension.get("modes")
    requirements = extension.get("mode_requirements")
    if not isinstance(modes, Mapping) or not modes or not isinstance(requirements, Mapping):
        raise RuntimeError("think-tank mode registry is incomplete")
    if set(modes) != set(requirements):
        raise RuntimeError("think-tank mode and requirement maps must have identical keys")
    existing_modes = target.get("modes") or {}
    existing_requirements = target.get("mode_requirements") or {}
    if not isinstance(existing_modes, Mapping) or not isinstance(existing_requirements, Mapping):
        raise RuntimeError("target group has invalid mode maps")
    duplicates = sorted(set(existing_modes) & set(modes))
    if duplicates:
        raise RuntimeError(f"think-tank modes conflict with stable registry: {duplicates}")
    target["modes"] = {**dict(existing_modes), **copy.deepcopy(dict(modes))}
    target["mode_requirements"] = {
        **dict(existing_requirements),
        **copy.deepcopy(dict(requirements)),
    }
    return value


def load_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "compute-tool-registry-v1":
        raise RuntimeError("invalid compute tool registry schema")
    if value.get("manager_version") != 2:
        raise RuntimeError("compute capability manager_version must be 2")
    if value.get("arbitrary_modules_allowed") is not False or value.get("arbitrary_requirements_allowed") is not False:
        raise RuntimeError("arbitrary compute extensions must remain disabled")
    if not isinstance(value.get("groups"), list) or not value["groups"]:
        raise RuntimeError("compute tool registry has no groups")
    return _merge_think_tank_registry(value)


def _requirement_files(rows: Any, group_id: str) -> list[str]:
    if not isinstance(rows, list):
        raise RuntimeError(f"requirements must be arrays for group {group_id}")
    result = []
    for filename in rows:
        name = str(filename)
        if not REQUIREMENT_RE.fullmatch(name) or not (HERE / name).is_file():
            raise RuntimeError(f"invalid or missing registered requirement file: {name}")
        result.append(name)
    return result


def _validated_modes(group: Mapping[str, Any], group_id: str) -> dict[str, dict[str, Any]]:
    modes = group.get("modes") or {}
    if not isinstance(modes, Mapping):
        raise RuntimeError(f"modes must be an object for group {group_id}")
    result: dict[str, dict[str, Any]] = {}
    for mode, raw_metadata in modes.items():
        if not MODE_RE.fullmatch(str(mode)) or not isinstance(raw_metadata, Mapping):
            raise RuntimeError(f"invalid mode metadata in group {group_id}: {mode}")
        metadata = dict(raw_metadata)
        if str(metadata.get("maturity") or "") not in ALLOWED_MATURITY:
            raise RuntimeError(f"invalid mode maturity in group {group_id}: {mode}")
        if str(metadata.get("network_policy") or "") not in ALLOWED_NETWORK_POLICIES:
            raise RuntimeError(f"invalid mode network policy in group {group_id}: {mode}")
        if not isinstance(metadata.get("deterministic"), bool):
            raise RuntimeError(f"mode deterministic flag is required in group {group_id}: {mode}")
        limits = metadata.get("limits")
        if not isinstance(limits, Mapping) or not limits:
            raise RuntimeError(f"mode limits are required in group {group_id}: {mode}")
        if any(not isinstance(item, int) or item <= 0 for item in limits.values()):
            raise RuntimeError(f"mode limits must be positive integers in group {group_id}: {mode}")
        result[str(mode)] = metadata
    return result


def _validated_group(
    raw: Mapping[str, Any],
    *,
    seen_ids: set[str],
    seen_operations: set[str],
) -> dict[str, Any]:
    group = dict(raw)
    group_id = str(group.get("id") or "")
    module_name = str(group.get("module") or "")
    operations = [str(item) for item in group.get("operations") or []]
    if not group_id or group_id in seen_ids:
        raise RuntimeError(f"invalid or duplicate tool group id: {group_id}")
    if not MODULE_RE.fullmatch(module_name):
        raise RuntimeError(f"invalid registered module name: {module_name}")
    if not operations or len(set(operations)) != len(operations):
        raise RuntimeError(f"invalid operation list for group {group_id}")
    duplicate = sorted(set(operations) & seen_operations)
    if duplicate:
        raise RuntimeError(f"operations registered by multiple groups: {duplicate}")

    modes = _validated_modes(group, group_id)
    mode_requirements = group.get("mode_requirements") or {}
    if not isinstance(mode_requirements, Mapping):
        raise RuntimeError(f"mode_requirements must be an object for group {group_id}")
    validated_requirements = {
        str(mode): _requirement_files(rows, group_id)
        for mode, rows in mode_requirements.items()
    }
    undeclared = sorted(set(validated_requirements) - set(modes))
    if undeclared:
        raise RuntimeError(
            f"mode_requirements references undeclared mode {undeclared[0]} in group {group_id}"
        )
    rollback = group.get("rollback")
    if not isinstance(rollback, Mapping) or not rollback.get("stable_module"):
        raise RuntimeError(f"rollback metadata is required for group {group_id}")

    group["default_requirements"] = _requirement_files(
        group.get("default_requirements") or [], group_id
    )
    group["mode_requirements"] = validated_requirements
    group["modes"] = modes
    return group


def validated_groups() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_operations: set[str] = set()
    for raw in load_registry()["groups"]:
        if not isinstance(raw, Mapping):
            raise RuntimeError("tool registry group must be an object")
        group = _validated_group(
            raw,
            seen_ids=seen_ids,
            seen_operations=seen_operations,
        )
        seen_ids.add(str(group["id"]))
        seen_operations.update(str(item) for item in group["operations"])
        groups.append(group)
    return groups


def load_registered_operations() -> dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]]:
    result: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {}
    for group in validated_groups():
        module = importlib.import_module(str(group["module"]))
        module_operations = getattr(module, "OPERATIONS", None)
        if not isinstance(module_operations, Mapping):
            raise RuntimeError(f"registered module has no OPERATIONS mapping: {group['module']}")
        expected = set(str(item) for item in group["operations"])
        observed = set(str(item) for item in module_operations)
        if expected != observed:
            raise RuntimeError(
                f"tool registry/module mismatch for {group['id']}: expected={sorted(expected)} observed={sorted(observed)}"
            )
        for name, handler in module_operations.items():
            if name in result or not callable(handler):
                raise RuntimeError(f"invalid or conflicting registered operation: {name}")
            result[str(name)] = handler
    return result


def group_for_operation(operation: str) -> dict[str, Any] | None:
    for group in validated_groups():
        if operation in group["operations"]:
            return group
    return None


def requirements_for_ticket(ticket: Mapping[str, Any]) -> list[str]:
    operation = str(ticket.get("operation") or "")
    inputs = ticket.get("inputs")
    mode = str(inputs.get("mode") or "") if isinstance(inputs, Mapping) else ""
    group = group_for_operation(operation)
    if group is None:
        return []
    selected = list(group.get("default_requirements") or [])
    mode_map = group.get("mode_requirements") or {}
    if mode and mode in mode_map:
        selected = list(mode_map[mode])
    return [str(HERE / filename) for filename in selected]


def runtime_plan(ticket: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(ticket.get("operation") or "")
    inputs = ticket.get("inputs")
    mode = str(inputs.get("mode") or "") if isinstance(inputs, Mapping) else ""
    group = group_for_operation(operation)
    if group is None:
        return {
            "schema_version": "compute-runtime-plan-v2",
            "operation": operation,
            "mode": mode or None,
            "capability_pack": "core",
            "requirements": [],
            "network_policy": "deny",
            "deterministic": True,
            "managed": False,
        }
    mode_metadata = (group.get("modes") or {}).get(mode, {}) if mode else {}
    return {
        "schema_version": "compute-runtime-plan-v2",
        "operation": operation,
        "mode": mode or None,
        "capability_pack": group["id"],
        "requirements": requirements_for_ticket(ticket),
        "network_policy": mode_metadata.get("network_policy", group.get("network_policy", "deny")),
        "deterministic": bool(mode_metadata.get("deterministic", group.get("deterministic", True))),
        "limits": dict(mode_metadata.get("limits") or group.get("resource_limits") or {}),
        "maturity": mode_metadata.get("maturity", group.get("maturity", "production")),
        "rollback": dict(group["rollback"]),
        "managed": True,
        "arbitrary_code_allowed": False,
        "arbitrary_requirements_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    requirements = sub.add_parser("requirements")
    requirements.add_argument("--ticket", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--ticket", required=True)
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
