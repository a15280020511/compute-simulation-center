#!/usr/bin/env python3
"""Generate the effective compute capability catalog from governed registries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAPABILITIES = HERE / "compute-capabilities.json"
BASE_REGISTRY = HERE / "tool-registry.json"
THINK_TANK_REGISTRY = HERE / "think-tank-mode-registry.json"
INSTITUTIONAL_REGISTRY = HERE / "institutional-toolkit-mode-registry.json"
CACHE_WORKFLOWS = (
    ROOT / ".github" / "workflows" / "compute-ticket.yml",
    ROOT / ".github" / "workflows" / "compute-validate.yml",
)
INSTITUTIONAL_REQUIREMENTS = (
    "requirements-institutional-economics.txt",
    "requirements-institutional-forecasting.txt",
    "requirements-institutional-decision.txt",
    "requirements-institutional-spatial.txt",
    "requirements-institutional-energy.txt",
    "requirements-institutional-climate-health.txt",
    "requirements-institutional-finance.txt",
    "requirements-institutional-knowledge.txt",
    "requirements-institutional-engineering.txt",
    "requirements-institutional-assurance.txt",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def mode_map(registry: Mapping[str, Any], label: str) -> dict[str, Any]:
    modes = registry.get("modes")
    requirements = registry.get("mode_requirements")
    if not isinstance(modes, dict) or not isinstance(requirements, dict) or set(modes) != set(requirements):
        raise RuntimeError(f"{label} registry is invalid")
    return modes


def update_cache_contract(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    required_paths = [f"compute-center/{name}" for name in INSTITUTIONAL_REQUIREMENTS]
    if all(item in text for item in required_paths):
        return
    marker_index = None
    marker_indent = None
    for index, line in enumerate(lines):
        if line.strip() == "cache-dependency-path: |":
            marker_index = index
            marker_indent = len(line) - len(line.lstrip())
            break
    if marker_index is None or marker_indent is None:
        raise RuntimeError(f"{path.name} has no cache-dependency-path block")
    end = marker_index + 1
    child_indent = marker_indent + 2
    while end < len(lines):
        line = lines[end]
        if not line.strip():
            break
        indent = len(line) - len(line.lstrip())
        if indent <= marker_indent:
            break
        end += 1
    present = {line.strip() for line in lines[marker_index + 1 : end]}
    additions = [" " * child_indent + item for item in required_paths if item not in present]
    lines[end:end] = additions
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    capabilities = load(CAPABILITIES)
    base = load(BASE_REGISTRY)
    think_tank = load(THINK_TANK_REGISTRY)
    institutional = load(INSTITUTIONAL_REGISTRY)
    think_tank_modes = mode_map(think_tank, "think-tank")
    institutional_modes = mode_map(institutional, "institutional")
    if set(think_tank_modes) & set(institutional_modes):
        raise RuntimeError("preview extension modes overlap")

    managed_mode_count = sum(len(group.get("modes") or {}) for group in base.get("groups", []))
    extension_mode_count = len(think_tank_modes) + len(institutional_modes)
    capabilities["schema_version"] = "compute-capabilities-v9"
    capabilities["managed_mode_count"] = managed_mode_count
    capabilities["think_tank_extension_mode_count"] = len(think_tank_modes)
    capabilities["institutional_extension_mode_count"] = len(institutional_modes)
    capabilities["extension_mode_count"] = extension_mode_count
    capabilities["effective_managed_mode_count"] = managed_mode_count + extension_mode_count

    runtime = capabilities.setdefault("runtime_packages", {})
    runtime.update(
        {
            "institutional-economics": "pyfixest, DoubleML, EconML, semopy and PyBLP",
            "institutional-forecasting": "StatsForecast, HierarchicalForecast, arch, PyOD, pyextremes and xskillscore",
            "institutional-decision": "EMA Workbench, pymcdm and Nashpy",
            "institutional-spatial": "GeoPandas, mgwr, momepy, spreg, spopt, MovingPandas and segregation",
            "institutional-infrastructure": "PyPSA, pandapower, WNTR, xclim and Starsim",
            "institutional-finance": "QuantLib and pyvinecopulib",
            "institutional-knowledge": "Splink, RapidFuzz, pySHACL, RDFLib and datasketch",
            "institutional-engineering": "python-control, reliability, Stockpyl, Ciw and JobShopLib",
            "institutional-assurance": "Fairlearn, Cleanlab, SHAP and Copulas",
        }
    )

    assessment = capabilities.setdefault("toolkit_assessment", {})
    scope = assessment.setdefault("scope", [])
    additions = [
        "causal machine learning, structural equations and industrial-organization instruments",
        "large-scale hierarchical forecasting, volatility, anomaly and extreme-value analysis",
        "deep-uncertainty exploration, comprehensive MCDA and matrix-game equilibrium",
        "urban morphology, mobility, segregation, spatial econometrics and facility location",
        "energy dispatch, electric power flow, water resilience, climate indices and group-level epidemic scenarios",
        "derivatives pricing, copula dependence and financial engineering",
        "record linkage, fuzzy entity matching, RDF/SHACL validation and approximate similarity",
        "reliability, control, multi-echelon inventory, queueing networks and job-shop scheduling",
        "fairness, label quality, model explanation and synthetic-data diagnostics",
    ]
    for row in additions:
        if row not in scope:
            scope.append(row)
    assessment["target_scope_gaps"] = [
        "domain-real benchmarks for all newly introduced controlled-preview modes",
        "long-running shadow validation and realized-outcome feedback",
        "specialized multi-physics digital-twin and military-grade wargaming backends",
    ]
    not_intended = assessment.setdefault("not_intended_for", [])
    native_restriction = "FMU or other ticket-supplied native binary execution"
    if native_restriction not in not_intended:
        not_intended.append(native_restriction)

    limits = capabilities.setdefault("limits", {})
    limits.update(
        {
            "institutional_rows": 50000,
            "institutional_columns": 50,
            "institutional_scenarios": 5000,
            "institutional_graph_triples": 50000,
            "institutional_agents": 100000,
        }
    )

    operations = capabilities.get("operations")
    if not isinstance(operations, list):
        raise RuntimeError("capability operations are invalid")
    finance = next(
        (row for row in operations if isinstance(row, dict) and row.get("id") == "finance_decision_analysis"),
        None,
    )
    if finance is None:
        raise RuntimeError("finance_decision_analysis is missing")
    existing_modes = finance.get("modes")
    if not isinstance(existing_modes, list):
        raise RuntimeError("finance_decision_analysis modes are invalid")
    finance["modes"] = list(existing_modes) + sorted(set(institutional_modes) - set(existing_modes))
    finance["use_when"] = (
        "financial, commercial, econometric, policy, social, strategic, spatial, infrastructure, climate, health, "
        "knowledge-engineering or model-governance evidence requires bounded institutional-style analysis"
    )
    finance["typical_output"] = (
        "validated evidence tables, statistical and causal estimates, forecasts, risk metrics, infrastructure and spatial "
        "results, optimized decisions, simulations and governance diagnostics"
    )
    restrictions = finance.setdefault("restrictions", [])
    for row in ("no ticket-supplied dependencies", "no external native binaries"):
        if row not in restrictions:
            restrictions.append(row)

    capabilities["operation_count"] = len(operations)
    CAPABILITIES.write_text(
        json.dumps(capabilities, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for workflow in CACHE_WORKFLOWS:
        update_cache_contract(workflow)
    print(
        json.dumps(
            {
                "status": "PASS",
                "operations": len(operations),
                "managed_modes": managed_mode_count,
                "think_tank_modes": len(think_tank_modes),
                "institutional_modes": len(institutional_modes),
                "effective_modes": managed_mode_count + extension_mode_count,
                "cache_workflows": [path.name for path in CACHE_WORKFLOWS],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
