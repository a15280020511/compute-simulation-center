#!/usr/bin/env python3
"""Fast operation-specific and model-governance validation before ticket acceptance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from capability_manager import load_registry
from game_theory_registry import game_theory_modes, load_game_theory_registry
from model_governance import GovernanceError, validate_ticket_governance

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "operation-input-schemas.json").read_text(encoding="utf-8"))
REGISTRY = load_registry()
GAME_REGISTRY = load_game_theory_registry()


def _controlled_preview_modes(operation: str) -> set[str]:
    if str(GAME_REGISTRY.get("target_operation") or "") != operation:
        return set()
    return set(game_theory_modes())


def _managed_schema(operation: str) -> Mapping[str, Any] | None:
    for raw in REGISTRY.get("groups", []):
        if not isinstance(raw, Mapping) or operation not in raw.get("operations", []):
            continue
        if raw.get("input_validation") != "mode_allowlist":
            return None
        modes = raw.get("modes")
        if not isinstance(modes, Mapping) or not modes:
            raise ValueError(f"managed operation has no mode allowlist: {operation}")
        allowed_modes = {str(item) for item in modes}
        allowed_modes.update(_controlled_preview_modes(operation))
        return {
            "type": "object",
            "required": ["mode"],
            "properties": {"mode": {"enum": sorted(allowed_modes)}},
            "additionalProperties": True,
            "maxProperties": 100,
        }
    return None


def validate_operation_inputs(ticket: Mapping[str, Any]) -> None:
    operation = str(ticket.get("operation") or "")
    schema = _managed_schema(operation)
    if schema is None:
        schema = CATALOG.get("operations", {}).get(operation)
    if not isinstance(schema, Mapping):
        raise ValueError(f"operation has no input schema: {operation}")
    validator = Draft202012Validator(dict(schema))
    errors = sorted(validator.iter_errors(ticket.get("inputs")), key=lambda item: list(item.absolute_path))
    if errors:
        rendered = []
        for error in errors[:20]:
            suffix = ".".join(str(item) for item in error.absolute_path)
            path = "inputs" + (f".{suffix}" if suffix else "")
            rendered.append(f"{path}: {error.message}")
        raise ValueError("; ".join(rendered))
    try:
        validate_ticket_governance(ticket)
    except GovernanceError as exc:
        raise ValueError(str(exc)) from exc