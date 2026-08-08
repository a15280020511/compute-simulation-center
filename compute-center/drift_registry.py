#!/usr/bin/env python3
"""Narrow controlled-preview registry view for the dynamic drift family."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "institutional-expansion-mode-registry.json"
ALLOWED_MODES = ("evidently_data_drift", "river_adwin_drift")


class DriftRegistryError(RuntimeError):
    pass


def load_drift_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != "institutional-expansion-mode-registry-v1":
        raise DriftRegistryError("invalid institutional expansion registry schema")
    if value.get("target_operation") != "finance_decision_analysis":
        raise DriftRegistryError("drift registry target operation mismatch")
    if value.get("status") != "controlled-preview":
        raise DriftRegistryError("drift modes must remain controlled-preview")
    if value.get("network_policy") != "deny" or value.get("arbitrary_code_allowed") is not False:
        raise DriftRegistryError("unsafe institutional drift registry policy")
    modes = value.get("modes")
    requirements = value.get("mode_requirements")
    if not isinstance(modes, Mapping) or not isinstance(requirements, Mapping):
        raise DriftRegistryError("institutional drift registry maps are missing")
    selected_modes: dict[str, dict[str, Any]] = {}
    selected_requirements: dict[str, list[str]] = {}
    for name in ALLOWED_MODES:
        raw = modes.get(name)
        raw_files = requirements.get(name)
        if not isinstance(raw, Mapping) or not isinstance(raw_files, list) or not raw_files:
            raise DriftRegistryError(f"drift mode is not fully registered: {name}")
        metadata = dict(raw)
        if metadata.get("maturity") != "controlled-preview":
            raise DriftRegistryError(f"drift mode must remain controlled-preview: {name}")
        if metadata.get("network_policy") != "deny" or metadata.get("deterministic") is not True:
            raise DriftRegistryError(f"unsafe drift mode policy: {name}")
        files: list[str] = []
        for raw_file in raw_files:
            file_name = str(raw_file)
            if not file_name.startswith("requirements-") or not file_name.endswith(".txt"):
                raise DriftRegistryError(f"invalid drift dependency file: {file_name}")
            if not (HERE / file_name).is_file():
                raise DriftRegistryError(f"missing drift dependency file: {file_name}")
            if file_name not in files:
                files.append(file_name)
        selected_modes[name] = metadata
        selected_requirements[name] = files
    return {
        "schema_version": "dynamic-drift-registry-view-v1",
        "target_operation": "finance_decision_analysis",
        "status": "controlled-preview",
        "network_policy": "deny",
        "arbitrary_code_allowed": False,
        "modes": selected_modes,
        "mode_requirements": selected_requirements,
    }


def drift_modes() -> tuple[str, ...]:
    return tuple(ALLOWED_MODES)


def drift_requirements() -> list[str]:
    registry = load_drift_registry()
    result: list[str] = []
    for name in ALLOWED_MODES:
        for file_name in registry["mode_requirements"][name]:
            if file_name not in result:
                result.append(file_name)
    return result
