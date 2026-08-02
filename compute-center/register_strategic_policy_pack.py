#!/usr/bin/env python3
"""One-shot deterministic registrar for the strategic-policy capability pack."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODES = [
    "open_spiel_policy_evaluation", "pygambit_pure_equilibria",
    "axelrod_strategy_tournament", "negmas_bilateral_bargaining",
    "scml_supply_chain_competition", "pyblp_price_counterfactual",
    "pymc_marketing_budget_allocation", "biogeme_choice_share",
    "pyagrum_bayesian_evidence", "scikit_criteria_method_agreement",
    "clingo_rule_action_set", "z3_constraint_counterexample",
    "hark_household_policy_response", "taxcalc_policy_counterfactual",
    "policyengine_transfer_counterfactual", "splink_entity_resolution",
    "rapidfuzz_record_collision", "datasketch_set_similarity",
    "rdflib_claim_evidence_graph", "pyshacl_graph_validation",
    "owlready2_ontology_summary", "igraph_link_analysis",
    "problog_evidence_probability", "issue_tree_coverage",
    "value_driver_tree", "source_reliability_matrix",
    "claim_evidence_contradiction", "event_timeline_collision",
    "red_team_challenge_matrix", "net_assessment_balance",
]
REQUIREMENTS = {
    "open_spiel_policy_evaluation": ["requirements-strategy-open-spiel.txt"],
    "pygambit_pure_equilibria": ["requirements-strategy-pygambit.txt"],
    "axelrod_strategy_tournament": ["requirements-strategy-axelrod.txt"],
    "negmas_bilateral_bargaining": ["requirements-strategy-negmas.txt"],
    "scml_supply_chain_competition": ["requirements-strategy-scml.txt"],
    "pyblp_price_counterfactual": ["requirements-strategy-pyblp.txt"],
    "pymc_marketing_budget_allocation": ["requirements-strategy-pymc-marketing.txt"],
    "biogeme_choice_share": ["requirements-strategy-biogeme.txt"],
    "pyagrum_bayesian_evidence": ["requirements-strategy-pyagrum.txt"],
    "scikit_criteria_method_agreement": ["requirements-strategy-scikit-criteria.txt"],
    "clingo_rule_action_set": ["requirements-strategy-clingo.txt"],
    "z3_constraint_counterexample": ["requirements-strategy-z3.txt"],
    "hark_household_policy_response": ["requirements-policy-econ-ark.txt"],
    "taxcalc_policy_counterfactual": ["requirements-policy-taxcalc.txt"],
    "policyengine_transfer_counterfactual": ["requirements-policy-policyengine.txt"],
    "splink_entity_resolution": ["requirements-intelligence-splink.txt"],
    "rapidfuzz_record_collision": ["requirements-intelligence-rapidfuzz.txt"],
    "datasketch_set_similarity": ["requirements-intelligence-datasketch.txt"],
    "rdflib_claim_evidence_graph": ["requirements-graph-rdflib.txt"],
    "pyshacl_graph_validation": ["requirements-graph-pyshacl.txt"],
    "owlready2_ontology_summary": ["requirements-graph-owlready2.txt"],
    "igraph_link_analysis": ["requirements-graph-igraph.txt"],
    "problog_evidence_probability": ["requirements-intelligence-problog.txt"],
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def register_tool_group() -> None:
    path = ROOT / "tool-registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["groups"] = [row for row in registry["groups"] if row.get("id") != "strategic-policy-intelligence"]
    registry["groups"].append({
        "id": "strategic-policy-intelligence",
        "module": "strategic_policy_intelligence_operations",
        "operations": ["strategic_policy_analysis"],
        "input_validation": "mode_allowlist",
        "default_requirements": [],
        "mode_requirements": REQUIREMENTS,
        "network_policy": "deny",
        "deterministic": True,
        "maturity": "controlled-preview",
        "resource_limits": {"max_seconds": 120, "max_memory_mb": 4096},
        "rollback": {"stable_module": "strategic_policy_intelligence_operations", "strategy": "git-revert"},
        "modes": {
            mode: {
                "maturity": "controlled-preview",
                "network_policy": "deny",
                "deterministic": True,
                "limits": {"max_seconds": 120, "max_memory_mb": 4096},
            }
            for mode in MODES
        },
    })
    write_json(path, registry)


def register_capability() -> None:
    path = ROOT / "compute-capabilities.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    prior = next((row for row in catalog["operations"] if row.get("id") == "strategic_policy_analysis"), None)
    catalog["operations"] = [row for row in catalog["operations"] if row.get("id") != "strategic_policy_analysis"]
    catalog["operations"].append({
        "id": "strategic_policy_analysis",
        "description": "断网执行博弈、谈判、市场竞争、营销配置、金融政策反事实、实体消歧、证据图谱、规则推理和公开结构化情报分析。",
        "modes": MODES,
        "dependency_policy": "每个包支持模式只安装仓库固定的精确版本依赖；纯方法模式仅使用核心数值栈。",
        "limits": {
            "maximum_runtime_seconds": 120,
            "maximum_memory_mb": 4096,
            "runtime_network_policy": "deny",
            "arbitrary_code_allowed": False,
            "ticket_supplied_agents_allowed": False,
            "remote_graph_or_ontology_loading_allowed": False,
        },
        "maturity": "controlled-preview",
        "required_evidence": [
            "输入字段、权重、规则、时间和实体范围",
            "依赖引擎版本与模式回执",
            "假设、适用边界和反例",
            "涉及个人数据时必须具备合法来源、用途限制和最小化证明",
            "高风险决策必须独立验证并由人类批准",
        ],
    })
    if prior is None:
        catalog["managed_mode_count"] = int(catalog.get("managed_mode_count", 0)) + len(MODES)
        catalog["effective_managed_mode_count"] = int(catalog.get("effective_managed_mode_count", 0)) + len(MODES)
    catalog["operation_count"] = len(catalog["operations"])
    catalog.setdefault("runtime_packages", {}).update({
        "OpenSpiel": "2.0.1", "PyGambit": "16.7.0", "NegMAS": "0.15.7",
        "SCML": "0.8.3", "Axelrod": "4.14.0", "PyBLP": "1.2.0",
        "PyMC-Marketing": "0.19.4", "Biogeme": "3.3.3", "pyAgrum": "3.0.0",
        "Scikit-Criteria": "0.9", "clingo": "5.8.0", "Z3 Solver": "5.0.0.0",
        "Econ-ARK": "0.17.2", "Tax-Calculator": "6.7.3",
        "PolicyEngine Core": "3.30.3", "Splink": "4.0.16",
        "RapidFuzz": "3.14.5", "datasketch": "2.0.0", "RDFLib": "7.6.0",
        "PySHACL": "0.40.1", "Owlready2": "0.51", "python-igraph": "1.0.0",
        "ProbLog": "2.2.10",
    })
    scope = catalog.setdefault("toolkit_assessment", {}).setdefault("scope", [])
    addition = "strategic games, negotiation, policy counterfactuals, entity resolution, evidence graphs and structured intelligence analysis"
    if addition not in scope:
        scope.append(addition)
    write_json(path, catalog)


def register_system_route() -> None:
    path = ROOT / "systems-computation-matrix.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    matrix["routes"]["strategic_policy_analysis"] = {
        "problem_class": "strategic-policy-and-intelligence-analysis",
        "system_level": "decision",
        "feedback_structure": "adversarial-evidence-and-policy-response",
        "required_gates": [
            "input_quality", "assumption_register", "constraint_feasibility",
            "uncertainty", "stress_test", "external_validation",
        ],
    }
    write_json(path, matrix)


def update_cache_inventory() -> None:
    path = REPO / ".github/workflows/compute-ticket.yml"
    text = path.read_text(encoding="utf-8")
    anchor = "            compute-center/requirements-global-spreg.txt\n"
    additions = "".join(
        f"            compute-center/{name}\n"
        for name in sorted({item for values in REQUIREMENTS.values() for item in values})
    )
    if "compute-center/requirements-strategy-open-spiel.txt" not in text:
        if anchor not in text:
            raise RuntimeError("compute-ticket cache anchor not found")
        text = text.replace(anchor, anchor + additions, 1)
    path.write_text(text, encoding="utf-8")


def repair_global_modes() -> None:
    path = ROOT / "think_tank_global_operations.py"
    text = path.read_text(encoding="utf-8")
    old = '''    parameters = np.asarray(variogram.parameters, dtype=float)\n    experimental = np.asarray(variogram.experimental, dtype=float)\n    bins = np.asarray(variogram.bins, dtype=float)\n    if not np.all(np.isfinite(parameters)):\n        raise ComputeError("variogram fit returned non-finite parameters")\n    return {\n'''
    new = '''    parameters = np.asarray(variogram.parameters, dtype=float)\n    experimental = np.asarray(variogram.experimental, dtype=float)\n    bins = np.asarray(variogram.bins, dtype=float)\n    parameter_finite = bool(np.all(np.isfinite(parameters)))\n    valid_lags = np.isfinite(experimental) & np.isfinite(bins)\n    dropped_lags = int(np.size(valid_lags) - np.count_nonzero(valid_lags))\n    experimental = experimental[valid_lags]\n    bins = bins[valid_lags]\n    return {\n'''
    if old in text:
        text = text.replace(old, new, 1)
        text = text.replace(
            '        "parameters": parameters.tolist(),\n        "lag_bins": bins.tolist(),',
            '        "parameters": parameters.tolist() if parameter_finite else [],\n'
            '        "fit_status": "fitted" if parameter_finite else "non-identifiable-from-input",\n'
            '        "dropped_non_finite_lags": dropped_lags,\n'
            '        "lag_bins": bins.tolist(),',
            1,
        )
    old = '''    for trajectory in results.trajectories:\n        array = np.asarray(trajectory.compartments, dtype=float)\n        index = dict(trajectory.compartment_idx)\n        if array.ndim == 3:\n            array = array.sum(axis=1)\n        if array.ndim != 2 or "I" not in index or "R" not in index:\n            raise ComputeError("Epydemix returned an unexpected trajectory shape")\n        infected_paths.append(array[:, int(index["I"])])\n        recovered_final.append(float(array[-1, int(index["R"])]))\n'''
    new = '''    for trajectory in results.trajectories:\n        compartments = trajectory.compartments\n        if isinstance(compartments, Mapping):\n            if "I" not in compartments or "R" not in compartments:\n                raise ComputeError("Epydemix returned incomplete compartment data")\n            infected_array = np.asarray(compartments["I"], dtype=float)\n            recovered_array = np.asarray(compartments["R"], dtype=float)\n            if infected_array.ndim > 1:\n                infected_array = infected_array.sum(axis=tuple(range(1, infected_array.ndim)))\n            if recovered_array.ndim > 1:\n                recovered_array = recovered_array.sum(axis=tuple(range(1, recovered_array.ndim)))\n            infected_paths.append(infected_array.reshape(-1))\n            recovered_final.append(float(recovered_array.reshape(-1)[-1]))\n            continue\n        array = np.asarray(compartments, dtype=float)\n        index = dict(trajectory.compartment_idx)\n        if array.ndim == 3:\n            array = array.sum(axis=1)\n        if array.ndim != 2 or "I" not in index or "R" not in index:\n            raise ComputeError("Epydemix returned an unexpected trajectory shape")\n        infected_paths.append(array[:, int(index["I"])])\n        recovered_final.append(float(array[-1, int(index["R"])]))\n'''
    if old in text:
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def harden_graph_validation() -> None:
    path = ROOT / "strategic_policy_intelligence_operations.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '    if any(token in data_turtle + shapes_turtle for token in ("http://", "https://", "file:")): raise ComputeError("remote or file IRIs are forbidden in graph validation")',
        '    if "file:" in data_turtle + shapes_turtle or "owl:imports" in data_turtle + shapes_turtle: raise ComputeError("file access and ontology imports are forbidden in graph validation")',
    )
    text = text.replace(
        'advanced=False, js=False)',
        'advanced=False, js=False, do_owl_imports=False)',
    )
    path.write_text(text, encoding="utf-8")


def restore_permanent_workflows() -> None:
    candidate = REPO / ".github/workflows/strategic-policy-candidate-compatibility.yml"
    if candidate.exists():
        candidate.unlink()
    permanent = subprocess.check_output(
        ["git", "show", "origin/main:.github/workflows/global-think-tank-toolkit-validate.yml"],
        text=True,
    )
    (REPO / ".github/workflows/global-think-tank-toolkit-validate.yml").write_text(permanent, encoding="utf-8")


def main() -> int:
    register_tool_group()
    register_capability()
    register_system_route()
    update_cache_inventory()
    repair_global_modes()
    harden_graph_validation()
    restore_permanent_workflows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
