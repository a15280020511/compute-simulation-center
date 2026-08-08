#!/usr/bin/env python3
"""Validate stable think-tank capabilities plus governed preview overlays."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from capability_manager import load_registry, validated_groups

HERE = Path(__file__).resolve().parent
PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+==[^=\s]+$")
EXPECTED_EXTENSION_MODES = 61
ASSURANCE_MODES = {
    "benchmark_comparison",
    "bounded_linear_kalman_filter",
    "calibration_diagnostics",
    "cross_model_agreement",
    "prediction_interval_validation",
    "probabilistic_forecast_scoring",
    "realized_outcome_feedback",
    "vva_acceptance_gate",
}


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} root must be an object")
    return value


def _validate_pinned_requirement_file(filename: str, *, mode: str) -> None:
    path = HERE / filename
    if not path.is_file():
        raise RuntimeError(f"missing dependency pack for {mode}: {path.name}")
    pins = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pins or any(not PIN_RE.fullmatch(line) for line in pins):
        raise RuntimeError(f"dependency pack is not exactly pinned: {path.name}")


def _validate_indirect_overlay() -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    overlay = _load("indirect-intelligence-mode-registry.json")
    if overlay.get("schema_version") != "indirect-intelligence-mode-registry-v1":
        raise RuntimeError("indirect intelligence overlay schema is invalid")
    if overlay.get("target_group") != "decision-intelligence":
        raise RuntimeError("indirect intelligence overlay target group is invalid")
    if overlay.get("network_policy") != "deny" or overlay.get("arbitrary_code_allowed") is not False:
        raise RuntimeError("indirect intelligence overlay violates offline or arbitrary-code policy")
    modes = overlay.get("modes")
    requirements = overlay.get("mode_requirements")
    if not isinstance(modes, Mapping) or not modes or not isinstance(requirements, Mapping):
        raise RuntimeError("indirect intelligence overlay maps are invalid")
    if set(modes) != set(requirements):
        raise RuntimeError("indirect intelligence overlay mode/requirement keys differ")
    for mode, metadata in modes.items():
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"invalid indirect intelligence metadata for {mode}")
        if metadata.get("maturity") != "controlled-preview":
            raise RuntimeError(f"indirect intelligence mode must remain controlled-preview: {mode}")
        if metadata.get("network_policy") != "deny" or metadata.get("deterministic") is not True:
            raise RuntimeError(f"unsafe indirect intelligence mode metadata: {mode}")
        limits = metadata.get("limits")
        if not isinstance(limits, Mapping) or not limits:
            raise RuntimeError(f"indirect intelligence mode limits are missing: {mode}")
        rows = requirements[mode]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"indirect intelligence dependency bundle is empty: {mode}")
        for filename in rows:
            _validate_pinned_requirement_file(str(filename), mode=str(mode))
    return overlay, modes, requirements


def validate() -> dict[str, Any]:
    extension = _load("think-tank-mode-registry.json")
    capabilities = _load("compute-capabilities.json")
    methods = _load("method-registry.json")
    _overlay, overlay_modes, overlay_requirements = _validate_indirect_overlay()

    modes = extension.get("modes")
    requirements = extension.get("mode_requirements")
    if not isinstance(modes, Mapping) or not isinstance(requirements, Mapping):
        raise RuntimeError("think-tank extension maps are invalid")
    if len(modes) != EXPECTED_EXTENSION_MODES or set(modes) != set(requirements):
        raise RuntimeError("think-tank extension count or key parity is invalid")
    if not ASSURANCE_MODES.issubset(set(modes)):
        raise RuntimeError("forecast assurance modes are missing from the extension registry")
    for mode, metadata in modes.items():
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"invalid metadata for {mode}")
        if metadata.get("maturity") != "controlled-preview" or metadata.get("network_policy") != "deny":
            raise RuntimeError(f"unsafe maturity or network policy for {mode}")
        if metadata.get("deterministic") is not True:
            raise RuntimeError(f"new think-tank mode must be deterministic: {mode}")
        rows = requirements[mode]
        if not isinstance(rows, list):
            raise RuntimeError(f"dependency map must be an array: {mode}")
        if mode in ASSURANCE_MODES:
            if rows:
                raise RuntimeError(
                    f"repository-native assurance mode must not install an extra dependency pack: {mode}"
                )
            continue
        if len(rows) != 1:
            raise RuntimeError(f"package-backed mode must resolve exactly one dependency pack: {mode}")
        _validate_pinned_requirement_file(str(rows[0]), mode=str(mode))

    effective = load_registry()
    target = next(
        group for group in effective["groups"] if group.get("id") == "decision-intelligence"
    )
    effective_mode_count = sum(
        len(group.get("modes") or {}) for group in effective["groups"]
    )
    catalog_effective_mode_count = capabilities.get("effective_managed_mode_count")
    if not isinstance(catalog_effective_mode_count, int) or catalog_effective_mode_count < EXPECTED_EXTENSION_MODES:
        raise RuntimeError("capability effective_managed_mode_count is invalid")

    finance = next(
        row
        for row in capabilities["operations"]
        if row.get("id") == "finance_decision_analysis"
    )
    static_finance_modes = set(finance.get("modes") or [])
    if not set(modes).issubset(static_finance_modes):
        raise RuntimeError("capability catalog omits stable think-tank extension modes")
    if set(overlay_modes) & static_finance_modes:
        raise RuntimeError(
            "controlled-preview overlay must remain outside the static capability baseline"
        )

    overlay_delta = len(set(overlay_modes) - static_finance_modes)
    if effective_mode_count != catalog_effective_mode_count + overlay_delta:
        raise RuntimeError(
            f"effective mode count mismatch: registry={effective_mode_count}, "
            f"static_capability_catalog={catalog_effective_mode_count}, "
            f"controlled_preview_overlay_delta={overlay_delta}"
        )
    target_modes = set(target.get("modes") or {})
    if not set(modes).issubset(target_modes):
        raise RuntimeError("effective registry does not expose all stable extension modes")
    if not set(overlay_modes).issubset(target_modes):
        raise RuntimeError("effective registry does not expose all indirect intelligence overlay modes")
    validated_groups()

    if capabilities.get("extension_mode_count") != EXPECTED_EXTENSION_MODES:
        raise RuntimeError("capability extension_mode_count mismatch")

    installed = methods.get("installed_method_packs")
    if not isinstance(installed, list):
        raise RuntimeError("method registry has no installed packs")
    pack_ids = {
        str(row.get("id") or "") for row in installed if isinstance(row, Mapping)
    }
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

    dependency_files = {rows[0] for rows in requirements.values() if rows}
    overlay_dependency_files = {
        filename
        for rows in overlay_requirements.values()
        for filename in rows
    }
    return {
        "status": "PASS",
        "extension_modes": len(modes),
        "repository_native_assurance_modes": len(ASSURANCE_MODES),
        "static_effective_managed_modes": catalog_effective_mode_count,
        "controlled_preview_overlay_modes": len(overlay_modes),
        "effective_managed_modes": effective_mode_count,
        "dependency_packs": len(dependency_files),
        "overlay_dependency_packs": len(overlay_dependency_files),
        "network_policy": "deny",
        "arbitrary_code_allowed": False,
    }


def main() -> int:
    print(json.dumps(validate(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
