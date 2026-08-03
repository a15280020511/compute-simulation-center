#!/usr/bin/env python3
"""One-time deterministic registrar for the final fixed capability pack.

This script is removed before production merge. It only edits repository-owned
control-plane files and is idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

MODES = [
    "quantlib_option_greeks",
    "quantlib_bond_duration",
    "active_inference_policy_choice",
    "pyod_anomaly_screen",
    "market_basket_association_rules",
    "replicator_dynamics",
    "finite_population_fixation",
    "prospect_theory_choice",
    "collective_action_threshold",
    "rumor_correction_dynamics",
    "trust_reputation_update",
    "group_consensus_pressure",
]

MODE_REQUIREMENTS = {
    "quantlib_option_greeks": ["requirements-final-quantlib.txt"],
    "quantlib_bond_duration": ["requirements-final-quantlib.txt"],
    "active_inference_policy_choice": ["requirements-final-pymdp.txt"],
    "pyod_anomaly_screen": ["requirements-final-pyod.txt"],
    "market_basket_association_rules": ["requirements-final-mlxtend.txt"],
}

PACKAGES = {
    "QuantLib": "1.43; fixed option and bond analytics",
    "inferactively-pymdp": "1.0.3; bounded one-step active-inference policy evaluation",
    "PyOD": "3.6.2; classic allowlisted ECOD, IForest and KNN anomaly screening only",
    "mlxtend": "0.25.0; frequent-itemset and association-rule analysis",
}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_runtime() -> None:
    path = ROOT / "strategic_policy_intelligence_operations.py"
    text = path.read_text(encoding="utf-8")
    import_line = (
        "from behavior_finance_intelligence_operations import "
        "HANDLERS as BEHAVIOR_FINANCE_INTELLIGENCE_HANDLERS\n"
    )
    if import_line not in text:
        anchor = "from compute_runner import ComputeError\n"
        if anchor not in text:
            raise RuntimeError("strategic runtime import anchor not found")
        text = text.replace(anchor, anchor + import_line, 1)
    update_line = "HANDLERS.update(BEHAVIOR_FINANCE_INTELLIGENCE_HANDLERS)\n\n"
    if update_line not in text:
        anchor = "SUPPORTED_MODES = tuple(sorted(HANDLERS))"
        if anchor not in text:
            raise RuntimeError("strategic runtime handler anchor not found")
        text = text.replace(anchor, update_line + anchor, 1)
    path.write_text(text, encoding="utf-8")


def patch_tool_registry() -> None:
    path = ROOT / "tool-registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    group = next(row for row in payload["groups"] if row.get("id") == "strategic-policy-intelligence")
    group.setdefault("mode_requirements", {}).update(MODE_REQUIREMENTS)
    modes = group.setdefault("modes", {})
    for mode in MODES:
        modes[mode] = {
            "maturity": "controlled-preview",
            "network_policy": "deny",
            "deterministic": True,
            "limits": {"max_seconds": 120, "max_memory_mb": 4096},
        }
    write_json(path, payload)


def patch_capabilities() -> None:
    path = ROOT / "compute-capabilities.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    operation = next(row for row in payload["operations"] if row.get("id") == "strategic_policy_analysis")
    existing = list(operation.get("modes", []))
    added = [mode for mode in MODES if mode not in existing]
    operation["modes"] = existing + added
    operation["description"] = (
        "断网执行博弈、谈判、市场竞争、营销配置、金融政策反事实、量化定价、"
        "异常筛查、实体消歧、证据图谱、群体行为机制和公开结构化情报分析。"
    )
    if added:
        payload["managed_mode_count"] = int(payload["managed_mode_count"]) + len(added)
        payload["effective_managed_mode_count"] = int(payload["effective_managed_mode_count"]) + len(added)
    payload.setdefault("runtime_packages", {}).update(PACKAGES)
    scope = payload.setdefault("toolkit_assessment", {}).setdefault("scope", [])
    statement = (
        "fixed derivatives and bond analytics, active-inference choice, classic anomaly screening, "
        "market-basket rules, evolutionary strategy and bounded group-behavior scenarios"
    )
    if statement not in scope:
        scope.append(statement)
    boundaries = payload.setdefault("safety_boundaries", [])
    additions = [
        "no live brokerage, order placement, personalized suitability or guaranteed-return claims",
        "no autonomous behavioral agent loop, individual psychological diagnosis or real-group deterministic prediction",
        "no persuasion targeting, information operations or automatic enforcement from anomaly scores",
    ]
    for addition in additions:
        if addition not in boundaries:
            boundaries.append(addition)
    write_json(path, payload)


def patch_compute_ticket_workflow() -> None:
    path = REPO / ".github" / "workflows" / "compute-ticket.yml"
    text = path.read_text(encoding="utf-8")
    additions = [
        "compute-center/requirements-final-quantlib.txt",
        "compute-center/requirements-final-pymdp.txt",
        "compute-center/requirements-final-pyod.txt",
        "compute-center/requirements-final-mlxtend.txt",
    ]
    missing = [item for item in additions if item not in text]
    if missing:
        anchor = "            compute-center/requirements-intelligence-problog.txt\n"
        if anchor not in text:
            raise RuntimeError("compute-ticket dependency cache anchor not found")
        inserted = "".join(f"            {item}\n" for item in missing)
        text = text.replace(anchor, anchor + inserted, 1)
    path.write_text(text, encoding="utf-8")


def patch_strategic_test() -> None:
    path = ROOT / "tests" / "test_strategic_policy_intelligence.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("self.assertEqual(len(HANDLERS), 30)", "self.assertEqual(len(HANDLERS), 42)")
    marker = '            "red_team_challenge_matrix",\n'
    additions = ''.join(f'            "{mode}",\n' for mode in MODES)
    if '            "quantlib_option_greeks",\n' not in text:
        if marker not in text:
            raise RuntimeError("strategic test expected-mode anchor not found")
        text = text.replace(marker, marker + additions, 1)
    old_source = 'source = (ROOT / "strategic_policy_intelligence_operations.py").read_text(encoding="utf-8")'
    new_source = '''source = "\\n".join((\n            (ROOT / "strategic_policy_intelligence_operations.py").read_text(encoding="utf-8"),\n            (ROOT / "behavior_finance_intelligence_operations.py").read_text(encoding="utf-8"),\n        ))'''
    if old_source in text:
        text = text.replace(old_source, new_source, 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    patch_runtime()
    patch_tool_registry()
    patch_capabilities()
    patch_compute_ticket_workflow()
    patch_strategic_test()
    print(json.dumps({"status": "PASS", "registered_modes": MODES, "mode_count": len(MODES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
