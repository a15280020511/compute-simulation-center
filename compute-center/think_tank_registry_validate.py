#!/usr/bin/env python3
"""Validate the governed top think-tank capability extension."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from capability_manager import load_registry, validated_groups

HERE = Path(__file__).resolve().parent
PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+==[^=\s]+$")
EXPECTED_EXTENSION_MODES = 53
EXPECTED_EFFECTIVE_MODES = 175


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} root must be an object")
    return value


def validate() -> dict[str, Any]:
    extension = _load("think-tank-mode-registry.json")
    capabilities = _load("compute-capabilities.json")
    methods = _load("method-registry.json")
    modes = extension.get("modes")
    requirements = extension.get("mode_requirements")
    if not isinstance(modes, Mapping) or not isinstance(requirements, Mapping):
        raise RuntimeError("think-tank extension maps are invalid")
    if len(modes) != EXPECTED_EXTENSION_MODES or set(modes) != set(requirements):
        raise RuntimeError("think-tank extension count or key parity is invalid")
    for mode, metadata in modes.items():
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"invalid metadata for {mode}")
        if metadata.get("maturity") != "controlled-preview" or metadata.get("network_policy") != "deny":
            raise RuntimeError(f"unsafe maturity or network policy for {mode}")
        if metadata.get("deterministic") is not True:
            raise RuntimeError(f"new think-tank mode must be deterministic: {mode}")
        rows = requirements[mode]
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError(f"mode must resolve exactly one dependency pack: {mode}")
        path = HERE / str(rows[0])
        if not path.is_file():
            raise RuntimeError(f"missing dependency pack for {mode}: {path.name}")
        pins = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not pins or any(not PIN_RE.fullmatch(line) for line in pins):
            raise RuntimeError(f"dependency pack is not exactly pinned: {path.name}")

    effective = load_registry()
    target = next(group for group in effective["groups"] if group.get("id") == "decision-intelligence")
    effective_mode_count = sum(len(group.get("modes") or {}) for group in effective["groups"])
    if effective_mode_count != EXPECTED_EFFECTIVE_MODES:
        raise RuntimeError(f"effective mode count mismatch: {effective_mode_count}")
    if not set(modes).issubset(set(target.get("modes") or {})):
        raise RuntimeError("effective registry does not expose all extension modes")
    validated_groups()

    if capabilities.get("extension_mode_count") != EXPECTED_EXTENSION_MODES:
        raise RuntimeError("capability extension_mode_count mismatch")
    if capabilities.get("effective_managed_mode_count") != EXPECTED_EFFECTIVE_MODES:
        raise RuntimeError("capability effective_managed_mode_count mismatch")
    finance = next(row for row in capabilities["operations"] if row.get("id") == "finance_decision_analysis")
    if not set(modes).issubset(set(finance.get("modes") or [])):
        raise RuntimeError("capability catalog omits extension modes")

    installed = methods.get("installed_method_packs")
    if not isinstance(installed, list):
        raise RuntimeError("method registry has no installed packs")
    pack_ids = {str(row.get("id") or "") for row in installed if isinstance(row, Mapping)}
    required_packs = {
        "thinktank-data-engineering",
        "thinktank-econometrics-business",
        "thinktank-finance-risk",
        "thinktank-decision-optimization",
        "thinktank-hierarchical-bayesian",
        "thinktank-raster-spatial",
        "thinktank-global-discovery",
    }
    if not required_packs.issubset(pack_ids):
        raise RuntimeError("method registry omits think-tank packs")

    return {
        "status": "PASS",
        "extension_modes": len(modes),
        "effective_managed_modes": effective_mode_count,
        "dependency_packs": len({rows[0] for rows in requirements.values()}),
        "network_policy": "deny",
        "arbitrary_code_allowed": False,
    }


def main() -> int:
    print(json.dumps(validate(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
