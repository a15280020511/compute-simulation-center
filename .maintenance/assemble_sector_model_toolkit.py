from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTER = ROOT / "compute-center"
MODES = {
    "doubleml_plr": ("requirements-sector-causal-econ.txt", {"max_rows": 5000, "max_features": 30}),
    "quantecon_markov_chain": ("requirements-sector-causal-econ.txt", {"max_states": 50, "max_steps": 10000}),
    "nash_bimatrix_equilibria": ("requirements-sector-causal-econ.txt", {"max_actions": 20, "max_equilibria": 50}),
    "ema_robust_regret": ("requirements-sector-causal-econ.txt", {"max_alternatives": 50, "max_scenarios": 500}),
    "pypsa_linear_power_flow": ("requirements-sector-energy.txt", {"max_buses": 2, "max_lines": 1}),
    "pandapower_ac_power_flow": ("requirements-sector-grid.txt", {"max_buses": 2, "max_lines": 1}),
    "wntr_hydraulic_snapshot": ("requirements-sector-water.txt", {"max_nodes": 2, "max_pipes": 1}),
    "pywr_resource_allocation": ("requirements-sector-water.txt", {"max_nodes": 2, "max_timesteps": 1}),
    "gstools_random_field": ("requirements-sector-geostat.txt", {"max_cells": 2500, "max_dimensions": 2}),
    "pykrige_interpolation": ("requirements-sector-geostat.txt", {"max_observations": 2000, "max_predictions": 2000}),
    "brightway_matrix_lca": ("requirements-sector-lca.txt", {"max_activities": 100, "max_biosphere_flows": 100}),
}


def load(name: str):
    return json.loads((CENTER / name).read_text(encoding="utf-8"))


def dump(name: str, value) -> None:
    (CENTER / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"missing patch anchor: {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


registry = load("tool-registry.json")
if any(group.get("id") == "sector-models" for group in registry["groups"]):
    raise SystemExit("sector-models group already exists")
registry["groups"].append({
    "id": "sector-models",
    "module": "sector_model_operations",
    "operations": ["sector_model_analysis"],
    "input_validation": "mode_allowlist",
    "default_requirements": [],
    "mode_requirements": {mode: [requirement] for mode, (requirement, _) in MODES.items()},
    "network_policy": "deny",
    "deterministic": True,
    "maturity": "controlled-preview",
    "resource_limits": {"max_seconds": 120, "max_memory_mb": 4096},
    "rollback": {"stable_module": "sector_model_operations", "strategy": "git-revert"},
    "modes": {
        mode: {
            "maturity": "controlled-preview",
            "network_policy": "deny",
            "deterministic": True,
            "limits": limits,
        }
        for mode, (_, limits) in MODES.items()
    },
})
dump("tool-registry.json", registry)

capabilities = load("compute-capabilities.json")
capabilities["schema_version"] = "compute-capabilities-v9"
capabilities["operation_count"] = 27
capabilities["managed_mode_count"] = 80
capabilities["extension_mode_count"] = 38
capabilities["effective_managed_mode_count"] = 118
packages = capabilities.setdefault("runtime_packages", {})
packages.update({
    "DoubleML": "0.11.3",
    "QuantEcon": "0.11.4",
    "Nashpy": "0.0.43",
    "EMA Workbench": "3.0.0",
    "PyPSA": "1.2.4",
    "pandapower": "3.5.4",
    "WNTR": "1.5.0",
    "Pywr": "1.31.1",
    "GSTools": "1.7.0",
    "PyKrige": "1.7.3",
    "Brightway 2.5": "1.1.1",
})
if any(row.get("id") == "sector_model_analysis" for row in capabilities["operations"]):
    raise SystemExit("sector_model_analysis already exists in capabilities")
capabilities["operations"].append({
    "id": "sector_model_analysis",
    "description": "断网执行因果机器学习、马尔可夫链、双矩阵博弈、深度不确定性稳健决策、电力系统、电网潮流、水力网络、水资源分配、地质统计和生命周期评估。",
    "modes": list(MODES),
    "dependency_policy": "每个模式只安装一个仓库固定的精确版本依赖包；禁止票据指定包或模块。",
    "limits": {
        "maximum_runtime_seconds": 120,
        "maximum_memory_mb": 4096,
        "runtime_network_policy": "deny",
        "arbitrary_code_allowed": False,
        "external_database_fetch_allowed": False,
    },
    "maturity": "controlled-preview",
    "required_evidence": [
        "显式单位和输入结构",
        "引擎版本回执",
        "数值收敛或可行性状态",
        "模型假设和适用边界",
        "独立验证或冻结基准后方可用于正式决策",
    ],
})
dump("compute-capabilities.json", capabilities)

matrix = load("systems-computation-matrix.json")
matrix["routes"]["sector_model_analysis"] = {
    "problem_class": "sector-specific-modeling",
    "system_level": "decision",
    "feedback_structure": "sector-network-and-scenario-feedback",
    "required_gates": [
        "input_quality", "assumption_register", "constraint_feasibility", "identifiability",
        "uncertainty", "calibration", "stress_test", "external_validation", "feedback_monitoring"
    ],
}
dump("systems-computation-matrix.json", matrix)

methods = load("method-registry.json")
new_packs = [
    {"id":"sector-causal-economics-strategy","status":"installed","requirements":"requirements-sector-causal-econ.txt","operations":["sector_model_analysis"],"capabilities":["double/debiased machine learning","Markov chains","bimatrix equilibrium","robust regret screening"],"network_policy":"deny"},
    {"id":"sector-energy-systems","status":"installed","requirements":"requirements-sector-energy.txt","operations":["sector_model_analysis"],"capabilities":["linear power flow","power-system network modeling"],"network_policy":"deny"},
    {"id":"sector-distribution-grid","status":"installed","requirements":"requirements-sector-grid.txt","operations":["sector_model_analysis"],"capabilities":["AC power flow","distribution-grid diagnostics"],"network_policy":"deny"},
    {"id":"sector-water-systems","status":"installed","requirements":"requirements-sector-water.txt","operations":["sector_model_analysis"],"capabilities":["hydraulic network simulation","water-resource allocation"],"network_policy":"deny"},
    {"id":"sector-geostatistics","status":"installed","requirements":"requirements-sector-geostat.txt","operations":["sector_model_analysis"],"capabilities":["random fields","variograms","ordinary kriging"],"network_policy":"deny"},
    {"id":"sector-lifecycle-assessment","status":"installed","requirements":"requirements-sector-lca.txt","operations":["sector_model_analysis"],"capabilities":["Brightway-compatible matrix inventory","impact characterization","activity contribution analysis"],"network_policy":"deny"},
]
existing_pack_ids = {row.get("id") for row in methods["installed_method_packs"]}
if existing_pack_ids.intersection(row["id"] for row in new_packs):
    raise SystemExit("sector method pack already exists")
methods["installed_method_packs"].extend(new_packs)
dump("method-registry.json", methods)

models = load("model-registry.json")
if any(row.get("operation") == "sector_model_analysis" for row in models["models"]):
    raise SystemExit("sector model registry entry already exists")
models["models"].append({
    "model_id": "sector_model_analysis-registered-v1",
    "operation": "sector_model_analysis",
    "maturity": "controlled-preview",
    "engineering_maturity": "controlled-preview",
    "evidence_maturity": "experimental",
    "risk_tier": "high",
    "calibration_supported": False,
    "allowed_backends": [],
    "theoretical_basis": "Fixed adapters for DoubleML, QuantEcon, Nashpy, EMA Workbench, PyPSA, pandapower, WNTR, Pywr, GSTools, PyKrige and Brightway-compatible matrix LCA.",
    "prohibited_use": [
        "Ticket-supplied code or packages",
        "External database or network access",
        "Autonomous power, water or infrastructure control",
        "Formal policy or engineering claims without independent validation",
    ],
})
dump("model-registry.json", models)

validate_path = CENTER / "validate_all_operations.py"
replace_once(
    validate_path,
    '        "bayesian_network_inference": {"mode": "fixed_network_inference", "nodes": ["A", "B"], "edges": [["A", "B"]], "cpds": [{"variable": "A", "variable_card": 2, "values": [[0.6], [0.4]]}, {"variable": "B", "variable_card": 2, "values": [[0.9, 0.2], [0.1, 0.8]], "evidence": ["A"], "evidence_card": [2]}], "query_variables": ["B"], "evidence": {"A": 1}},\n',
    '        "bayesian_network_inference": {"mode": "fixed_network_inference", "nodes": ["A", "B"], "edges": [["A", "B"]], "cpds": [{"variable": "A", "variable_card": 2, "values": [[0.6], [0.4]]}, {"variable": "B", "variable_card": 2, "values": [[0.9, 0.2], [0.1, 0.8]], "evidence": ["A"], "evidence_card": [2]}], "query_variables": ["B"], "evidence": {"A": 1}},\n        "sector_model_analysis": {"mode": "nash_bimatrix_equilibria", "row_payoffs": [[3, 0], [5, 1]], "column_payoffs": [[3, 5], [0, 1]]},\n',
)

replace_once(CENTER / "think_tank_registry_validate.py", "EXPECTED_EFFECTIVE_MODES = 107", "EXPECTED_EFFECTIVE_MODES = 118")
replace_once(CENTER / "tests/test_think_tank_operations.py", 'result["effective_managed_modes"], 107', 'result["effective_managed_modes"], 118')
replace_once(CENTER / "tests/test_governance_catalogs.py", 'report["operation_count"], 26', 'report["operation_count"], 27')
replace_once(CENTER / "tests/test_governance_catalogs.py", 'report["managed_mode_count"], 69', 'report["managed_mode_count"], 80')
replace_once(CENTER / "tests/test_governance_catalogs.py", 'report["installed_method_pack_count"], 12', 'report["installed_method_pack_count"], 18')
replace_once(CENTER / "tests/test_governance_v2.py", 'report["covered_operation_count"], 26', 'report["covered_operation_count"], 27')

workflow = ROOT / ".github/workflows/compute-all-operations-validate.yml"
replace_once(
    workflow,
    '            compute-center/requirements-constraints.txt\n',
    '            compute-center/requirements-constraints.txt\n            compute-center/requirements-sector-causal-econ.txt\n',
)
replace_once(
    workflow,
    '          python -m pip install --disable-pip-version-check --no-input -r compute-center/requirements-constraints.txt\n',
    '          python -m pip install --disable-pip-version-check --no-input -r compute-center/requirements-constraints.txt\n          python -m pip install --disable-pip-version-check --no-input -r compute-center/requirements-sector-causal-econ.txt\n',
)
replace_once(workflow, "Execute all 26 production operations", "Execute all 27 production operations")
replace_once(workflow, "operation_summary['operation_count_expected'] == 26", "operation_summary['operation_count_expected'] == 27")
replace_once(workflow, "operation_summary['operation_count_executed'] == 26", "operation_summary['operation_count_executed'] == 27")
replace_once(workflow, "operation_summary['passed'] == 26", "operation_summary['passed'] == 27")
replace_once(workflow, "len(operation_summary['registry']) == 26", "len(operation_summary['registry']) == 27")
replace_once(workflow, "len(operation_summary['rows']) == 26", "len(operation_summary['rows']) == 27")
replace_once(workflow, "'operations': 26, 'managed_modes': 69", "'operations': 27, 'managed_modes': 80")
