#!/usr/bin/env python3
"""Validate the Exa-discovered institutional toolkit registry and package contracts."""
from __future__ import annotations

import json
import re
from pathlib import Path

from institutional_toolkit_operations import HANDLERS

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "institutional-toolkit-mode-registry.json"
CATALOG = HERE / "institutional-tool-catalog.json"
BASE_REGISTRY = HERE / "tool-registry.json"
THINK_TANK_REGISTRY = HERE / "think-tank-mode-registry.json"
PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+==[^=\s]+$")


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    base = json.loads(BASE_REGISTRY.read_text(encoding="utf-8"))
    think_tank = json.loads(THINK_TANK_REGISTRY.read_text(encoding="utf-8"))
    modes = registry.get("modes")
    requirements = registry.get("mode_requirements")
    assert registry.get("schema_version") == "institutional-toolkit-mode-registry-v1"
    assert registry.get("target_group") == "decision-intelligence"
    assert registry.get("status") == "controlled-preview"
    assert registry.get("network_policy") == "deny"
    assert registry.get("arbitrary_code_allowed") is False
    assert isinstance(modes, dict) and len(modes) == 41
    assert isinstance(requirements, dict) and set(requirements) == set(modes) == set(HANDLERS)
    stable_modes = set()
    for group in base.get("groups", []):
        stable_modes.update((group.get("modes") or {}).keys())
    think_tank_modes = set((think_tank.get("modes") or {}).keys())
    assert not set(modes) & stable_modes
    assert not set(modes) & think_tank_modes
    requirement_files = sorted({item for rows in requirements.values() for item in rows})
    assert len(requirement_files) == 10
    package_rows = {}
    for filename in requirement_files:
        path = HERE / filename
        assert path.is_file(), filename
        rows = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert rows and all(PIN_RE.fullmatch(row) for row in rows), filename
        package_rows[filename] = rows
    for mode, metadata in modes.items():
        assert metadata.get("maturity") == "controlled-preview", mode
        assert metadata.get("network_policy") == "deny", mode
        assert metadata.get("deterministic") is True, mode
        limits = metadata.get("limits")
        assert isinstance(limits, dict) and limits
        assert all(isinstance(value, int) and value > 0 for value in limits.values())
        assert len(requirements[mode]) == 1
    assert catalog.get("schema_version") == "exa-institutional-tool-catalog-v1"
    assert catalog.get("accepted_count") == 41
    assert len(catalog.get("accepted", [])) == 41
    assert {row["mode"] for row in catalog["accepted"]} == set(modes)
    assert catalog.get("search_rounds") == 2
    assert catalog.get("candidate_result_slots") == 200
    result = {
        "status": "PASS",
        "mode_count": len(modes),
        "requirement_pack_count": len(requirement_files),
        "accepted_tool_count": catalog["accepted_count"],
        "rejected_group_count": catalog["rejected_count"],
        "network_policy": registry["network_policy"],
        "arbitrary_code_allowed": registry["arbitrary_code_allowed"],
        "ticket_selected_dependencies": False,
        "requirement_files": requirement_files,
        "packages": package_rows,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
