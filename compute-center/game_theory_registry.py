#!/usr/bin/env python3
"""Single-source loader for controlled-preview game-theory modes."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "game-theory-mode-registry.json"
MODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
REQUIREMENT_RE = re.compile(r"^requirements-[a-z0-9-]+\.txt$")


class GameTheoryRegistryError(RuntimeError):
    pass


def load_game_theory_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "game-theory-mode-registry-v1":
        raise GameTheoryRegistryError("invalid game-theory mode registry schema")
    if value.get("target_group") != "decision-intelligence" or value.get("target_operation") != "finance_decision_analysis":
        raise GameTheoryRegistryError("game-theory mode registry target mismatch")
    if value.get("status") != "controlled-preview":
        raise GameTheoryRegistryError("game-theory modes must remain controlled-preview")
    if value.get("network_policy") != "deny" or value.get("arbitrary_code_allowed") is not False:
        raise GameTheoryRegistryError("unsafe game-theory registry policy")
    modes = value.get("modes")
    requirements = value.get("mode_requirements")
    if not isinstance(modes, Mapping) or not modes:
        raise GameTheoryRegistryError("game-theory mode registry has no modes")
    if not isinstance(requirements, Mapping) or set(requirements) != set(modes):
        raise GameTheoryRegistryError("game-theory mode requirements must exactly cover modes")
    validated_modes: dict[str, dict[str, Any]] = {}
    validated_requirements: dict[str, list[str]] = {}
    for raw_name, raw_metadata in modes.items():
        name = str(raw_name)
        if not MODE_RE.fullmatch(name) or not isinstance(raw_metadata, Mapping):
            raise GameTheoryRegistryError(f"invalid game-theory mode entry: {name}")
        metadata = dict(raw_metadata)
        if metadata.get("maturity") != "controlled-preview":
            raise GameTheoryRegistryError(f"game-theory mode must remain controlled-preview: {name}")
        if metadata.get("network_policy") != "deny" or metadata.get("deterministic") is not True:
            raise GameTheoryRegistryError(f"unsafe game-theory mode policy: {name}")
        if metadata.get("user_defined_game_code_allowed") is not False:
            raise GameTheoryRegistryError(f"game-theory mode must forbid user-defined game code: {name}")
        raw_files = requirements[name]
        if not isinstance(raw_files, list) or not raw_files:
            raise GameTheoryRegistryError(f"game-theory mode requires pinned dependency files: {name}")
        files: list[str] = []
        for raw_file in raw_files:
            file_name = str(raw_file)
            if not REQUIREMENT_RE.fullmatch(file_name):
                raise GameTheoryRegistryError(f"invalid game-theory requirement file: {file_name}")
            if not (HERE / file_name).is_file():
                raise GameTheoryRegistryError(f"missing game-theory requirement file: {file_name}")
            if file_name not in files:
                files.append(file_name)
        validated_modes[name] = metadata
        validated_requirements[name] = files
    return {**value, "modes": validated_modes, "mode_requirements": validated_requirements}


def game_theory_modes() -> tuple[str, ...]:
    return tuple(sorted(load_game_theory_registry()["modes"]))


def game_theory_requirements(mode_names: tuple[str, ...] | list[str] | None = None) -> list[str]:
    registry = load_game_theory_registry()
    names = list(mode_names) if mode_names is not None else list(registry["modes"])
    unknown = sorted(set(names) - set(registry["modes"]))
    if unknown:
        raise GameTheoryRegistryError(f"unknown game-theory modes: {unknown}")
    result: list[str] = []
    for name in names:
        for file_name in registry["mode_requirements"][name]:
            if file_name not in result:
                result.append(file_name)
    return result
