#!/usr/bin/env python3
"""One-shot, idempotent registrar for forecast assurance and center boundaries."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTER = ROOT / "compute-center"

MODES = {
    "benchmark_comparison": {"max_candidates": 200},
    "bounded_linear_kalman_filter": {"max_dimension": 20, "max_steps": 10000},
    "calibration_diagnostics": {"max_observations": 20000, "max_bins": 50},
    "cross_model_agreement": {"max_models": 50, "max_observations": 20000},
    "prediction_interval_validation": {"max_observations": 20000},
    "probabilistic_forecast_scoring": {"max_observations": 20000, "max_classes": 50},
    "realized_outcome_feedback": {"max_observations": 20000},
    "vva_acceptance_gate": {"max_checks": 20},
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def patch_gateway() -> None:
    path = CENTER / "decision_intelligence_gateway.py"
    text = path.read_text(encoding="utf-8")
    marker = "from assurance_operations import HANDLERS as ASSURANCE_HANDLERS\n"
    if marker not in text:
        anchor = "from compute_runner import ComputeError\n"
        text = text.replace(anchor, anchor + marker)
    text = text.replace(
        "PREVIEW_MODES = set(THINK_TANK_HANDLERS)",
        "PREVIEW_MODES = set(THINK_TANK_HANDLERS) | set(ASSURANCE_HANDLERS)",
    )
    if "**ASSURANCE_HANDLERS," not in text:
        text = text.replace("    **THINK_TANK_HANDLERS,\n", "    **THINK_TANK_HANDLERS,\n    **ASSURANCE_HANDLERS,\n")
    path.write_text(text, encoding="utf-8")


def patch_registry() -> None:
    path = CENTER / "think-tank-mode-registry.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    requirements = value.setdefault("mode_requirements", {})
    modes = value.setdefault("modes", {})
    for mode, limits in MODES.items():
        requirements.setdefault(mode, [])
        modes.setdefault(mode, {
            "maturity": "controlled-preview",
            "network_policy": "deny",
            "deterministic": True,
            "limits": limits,
        })
    value["mode_requirements"] = {key: requirements[key] for key in sorted(requirements)}
    value["modes"] = {key: modes[key] for key in sorted(modes)}
    write_json(path, value)


def patch_capability_catalog() -> None:
    path = CENTER / "compute-capabilities.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    operation = next(row for row in value["operations"] if row.get("id") == "finance_decision_analysis")
    existing = list(operation.get("modes") or [])
    added = [mode for mode in MODES if mode not in existing]
    operation["modes"] = sorted(existing + added)
    if added:
        value["extension_mode_count"] = int(value.get("extension_mode_count", 0)) + len(added)
        value["effective_managed_mode_count"] = int(value.get("effective_managed_mode_count", 0)) + len(added)
    packages = value.setdefault("runtime_packages", {})
    packages["repository-native-assurance"] = "forecast scoring, calibration, interval validation, realized-outcome feedback, cross-model agreement, VV&A and bounded linear state estimation"
    assessment = value.setdefault("toolkit_assessment", {})
    scope = assessment.setdefault("scope", [])
    statement = "forecast scoring, calibration diagnostics, prediction-interval validation, realized-outcome feedback, benchmark comparison, cross-model agreement, VV&A and bounded generic state estimation"
    if statement not in scope:
        scope.append(statement)
    not_intended = assessment.setdefault("not_intended_for", [])
    restriction = "live person or military target tracking, weapons integration or autonomous operational control"
    if restriction not in not_intended:
        not_intended.append(restriction)
    write_json(path, value)


def patch_docs() -> None:
    path = CENTER / "README.md"
    text = path.read_text(encoding="utf-8")
    heading = "## Forecast assurance and VV&A"
    if heading not in text:
        text += """

## Forecast assurance and VV&A

Eight controlled-preview modes are exposed through `finance_decision_analysis`: probability scoring, calibration diagnostics, prediction-interval validation, realized-outcome feedback, benchmark comparison, cross-model agreement, a verification/validation/accreditation gate, and a bounded linear Kalman filter. They run offline on supplied structured data, make zero model calls, accept no ticket code, and do not support live feeds, person identification, target designation, weapons integration or autonomous control.

The authoritative allocation between the Compute Center and Intelligence Center is recorded in repository-root `CENTER_CAPABILITY_OWNERSHIP.json`.
"""
        path.write_text(text, encoding="utf-8")


def patch_count_contracts() -> None:
    allowed = {".py", ".yml", ".yaml", ".json", ".md", ".txt"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed or ".git" in path.parts:
            continue
        if path in {CENTER / "compute-capabilities.json", CENTER / "think-tank-mode-registry.json"}:
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        text = re.sub(r'("extension_mode_count"\s*:\s*)53\b', r'\g<1>61', text)
        text = re.sub(r"(['\"]extension_mode_count['\"]\s*(?:==|:|=)\s*)53\b", r"\g<1>61", text)
        text = re.sub(r"(['\"]extension_modes['\"]\s*(?:==|:|=)\s*)53\b", r"\g<1>61", text)
        text = re.sub(r'("effective_managed_mode_count"\s*:\s*)175\b', r'\g<1>183', text)
        text = re.sub(r"(['\"]effective_managed_mode_count['\"]\s*(?:==|:|=)\s*)175\b", r"\g<1>183", text)
        text = re.sub(r"(['\"]effective_managed_modes?['\"]\s*(?:==|:|=)\s*)175\b", r"\g<1>183", text)
        text = re.sub(r"(?i)(extension modes?\s*[:=/|` ]+?)53\b", r"\g<1>61", text)
        text = re.sub(r"(?i)(effective(?: managed)? modes?\s*[:=/|` ]+?)175\b", r"\g<1>183", text)
        if "think-tank" in text.lower() or "THINK_TANK" in text:
            text = text.replace("== 53", "== 61").replace("!= 53", "!= 61")
        if text != original:
            path.write_text(text, encoding="utf-8")


def main() -> int:
    patch_gateway()
    patch_registry()
    patch_capability_catalog()
    patch_docs()
    patch_count_contracts()
    print(json.dumps({"status": "PASS", "added_modes": sorted(MODES), "extension_modes": 61, "effective_modes": 183}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
