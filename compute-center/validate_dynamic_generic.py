#!/usr/bin/env python3
"""Deterministic acceptance checks for generic dynamic orchestration."""
from __future__ import annotations

import ast
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_pipeline_planner import plan_dynamic_pipeline
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{name} root must be an object")
    return value


def _base_ticket() -> dict[str, Any]:
    return {
        "task_id": "dynamic-generic-acceptance-base",
        "operation": "scenario_compare",
        "inputs": {
            "model": {
                "intercept": 10.0,
                "coefficients": {"demand": 2.0, "cost": -1.0},
            },
            "scenarios": [
                {"name": "weak", "values": {"demand": 1.0, "cost": 4.0}},
                {"name": "base", "values": {"demand": 2.0, "cost": 3.0}},
                {"name": "strong", "values": {"demand": 4.0, "cost": 1.0}},
                {"name": "stretch", "values": {"demand": 5.0, "cost": 2.0}},
            ],
        },
        "quality_profile": {
            "decision_class": "exploratory",
            "probabilistic_claim": False,
        },
    }


def _assert_source_safety() -> None:
    for name in ("dynamic_pipeline_planner.py", "pipeline_adapters.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        for forbidden in (
            "import requests",
            "import socket",
            "urllib.request",
            "subprocess.",
            "pickle.loads",
            "import mcp",
        ):
            assert forbidden not in text, (name, forbidden)
        tree = ast.parse(text)
        direct = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert {"eval", "exec", "compile"}.isdisjoint(direct), name


def _assert_policy_and_graph() -> None:
    policy = _load("dynamic-orchestration-policy.json")
    graph = _load("dynamic-capability-graph.json")
    contracts = _load("operation-contract-registry.json")

    assert policy["schema_version"] == "compute-dynamic-orchestration-policy-v5"
    assert policy["status"] == "controlled-preview"
    assert policy["planner"] == "ortools-cp-sat"
    assert policy["graph_engine"] == "networkx"
    assert policy["network_policy"] == "deny"
    assert policy["model_calls"] == 0
    assert policy["objective_text_routing_allowed"] is False
    assert policy["structured_signals_only"] is True
    assert policy["dynamic_operation_discovery_allowed"] is False
    assert policy["ticket_supplied_code_allowed"] is False
    assert policy["automatic_parallel_execution"] is False
    assert policy["cycles_allowed"] is False
    assert policy["solver_policy"]["require_optimal_status"] is True
    assert policy["solver_policy"]["num_search_workers"] == 1
    assert policy["solver_policy"]["exhaustive_cross_check_max_optional_nodes"] >= 4
    assert set(policy["allowed_operations"]) == {
        "scenario_compare",
        "descriptive_statistics",
        "sensitivity_analysis",
        "monte_carlo",
        "constrained_optimization",
    }
    assert list(policy["selection_policy"]["stage_rules"]) == [
        "scenario_statistics",
        "sensitivity",
        "risk_simulation",
        "decision_optimization",
    ]

    expected_nodes = [
        "scenarios",
        "scenario_statistics",
        "sensitivity",
        "risk_simulation",
        "decision_optimization",
    ]
    assert graph["schema_version"] == "compute-dynamic-capability-graph-v2"
    assert graph["status"] == "controlled-preview"
    assert graph["graph_engine"] == "networkx"
    assert graph["selection_engine"] == "ortools-cp-sat"
    assert [node["id"] for node in graph["nodes"]] == expected_nodes
    assert graph["precedence"] == [
        ["scenarios", "scenario_statistics"],
        ["scenario_statistics", "sensitivity"],
        ["sensitivity", "risk_simulation"],
        ["risk_simulation", "decision_optimization"],
    ]
    assert graph["safety"]["full_graph_must_be_single_serial_chain"] is True
    assert graph["safety"]["automatic_parallel_execution"] is False
    assert graph["safety"]["dynamic_operation_discovery_allowed"] is False

    assert contracts["schema_version"] == "compute-operation-contract-registry-v1"
    for operation in policy["allowed_operations"]:
        assert operation in contracts["contracts"], operation


def _cases() -> tuple[list[dict[str, Any]], list[list[str]]]:
    base = _base_ticket()
    cases: list[dict[str, Any]] = []

    a = copy.deepcopy(base)
    a["task_id"] = "dynamic-generic-acceptance-a"
    a["inputs"]["scenarios"] = a["inputs"]["scenarios"][:2]
    cases.append(a)

    b = copy.deepcopy(base)
    b["task_id"] = "dynamic-generic-acceptance-b"
    b["inputs"]["scenarios"] = b["inputs"]["scenarios"][:3]
    cases.append(b)

    c = copy.deepcopy(base)
    c["task_id"] = "dynamic-generic-acceptance-c"
    c["inputs"]["scenarios"] = c["inputs"]["scenarios"][:2]
    c["quality_profile"]["probabilistic_claim"] = True
    cases.append(c)

    d = copy.deepcopy(base)
    d["task_id"] = "dynamic-generic-acceptance-d"
    d["inputs"]["scenarios"] = d["inputs"]["scenarios"][:3]
    d["data_context"] = {
        "variables": [
            {"name": "u1", "required": True, "source_type": "proxy", "confidence": "medium"},
            {"name": "u2", "required": True, "source_type": "proxy", "confidence": "medium"},
        ]
    }
    cases.append(d)

    e = copy.deepcopy(base)
    e["task_id"] = "dynamic-generic-acceptance-e"
    cases.append(e)

    f = copy.deepcopy(base)
    f["task_id"] = "dynamic-generic-acceptance-f"
    f["inputs"]["scenarios"] = f["inputs"]["scenarios"][:3]
    f["inputs"]["dynamic_context"] = {
        "continuous_decision_optimization": True,
        "allow_continuous_interpolation": True,
        "controllable_variables": ["demand", "cost"],
    }
    cases.append(f)

    g = copy.deepcopy(base)
    g["task_id"] = "dynamic-generic-acceptance-g"
    g["quality_profile"]["probabilistic_claim"] = True
    g["inputs"]["dynamic_context"] = {
        "continuous_decision_optimization": True,
        "allow_continuous_interpolation": True,
        "controllable_variables": ["demand", "cost"],
    }
    cases.append(g)

    expected = [
        ["scenarios"],
        ["scenarios", "sensitivity"],
        ["scenarios", "risk_simulation"],
        ["scenarios", "sensitivity", "risk_simulation"],
        ["scenarios", "scenario_statistics", "sensitivity"],
        ["scenarios", "sensitivity", "decision_optimization"],
        ["scenarios", "scenario_statistics", "sensitivity", "risk_simulation", "decision_optimization"],
    ]
    return cases, expected


def _assert_optimal_plans() -> dict[str, Any]:
    cases, expected = _cases()
    plans = [plan_dynamic_pipeline(item) for item in cases]
    observed = [plan["stage_order"] for plan in plans]
    assert observed == expected, (observed, expected)
    assert len({tuple(order) for order in observed}) == len(expected)
    for plan in plans:
        optimization = plan["optimization"]
        cross = optimization["exhaustive_cross_check"]
        assert plan["selection_engine"] == "ortools-cp-sat"
        assert plan["graph_engine"] == "networkx"
        assert optimization["solver_status"] == "OPTIMAL"
        assert optimization["global_optimal_proven"] is True
        assert cross["performed"] is True
        assert cross["passed"] is True
        assert cross["optional_node_count"] == 4
        assert optimization["objective_value"] == cross["best_objective"]
        assert optimization["solver_policy"]["num_search_workers"] == 1
        assert plan["objective_text_used"] is False
        assert plan["network_policy"] == "deny"
        assert plan["automatic_parallel_execution"] is False
        assert plan["model_calls"] == 0
    return {
        "stage_orders": observed,
        "objective_values": [plan["optimization"]["objective_value"] for plan in plans],
    }


def _production_ticket() -> dict[str, Any]:
    value = copy.deepcopy(_base_ticket())
    value["task_id"] = "dynamic-production-sim-generic-20260807"
    value["objective"] = "Validate generic controlled-preview dynamic orchestration."
    value["quality_profile"].update({
        "probabilistic_claim": True,
        "benchmark_ids": [],
        "method_ids": [],
        "sample_ids": [],
        "rule_ids": [],
        "independent_cross_check_passed": False,
        "publication_policy": "status_only",
    })
    value["inputs"]["dynamic_context"] = {
        "continuous_decision_optimization": True,
        "allow_continuous_interpolation": True,
        "controllable_variables": ["demand", "cost"],
    }
    value["pipeline"] = {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "generic dynamic production dispatcher acceptance",
        "upstream_refs": [],
    }
    value["preflight_policy"] = {
        "enforcement": "advisory",
        "allow_assumptions": True,
        "allow_proxy": True,
        "require_user_approval_for_low_confidence": False,
        "max_assumption_ratio": 1.0,
        "small_sample_threshold": 2,
        "outlier_rate_threshold": 1.0,
    }
    return value


def _assert_dispatcher() -> dict[str, Any]:
    ticket = _production_ticket()
    requirements = requirement_files_for_ticket(ticket)
    assert len(requirements) == 1
    assert Path(requirements[0]).name == "requirements-ortools.txt"
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"

    root = HERE / "dynamic-production-sim"
    ticket_path = HERE / "dynamic-production-ticket.json"
    if root.exists():
        shutil.rmtree(root)
    ticket_path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        return_code = compute_dispatch.main(["--ticket", str(ticket_path), "--output-dir", str(root)])
        assert return_code == 0
        result = json.loads((root / "compute-result.json").read_text(encoding="utf-8"))
        state = json.loads((root / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "compute-audit.json").read_text(encoding="utf-8"))
        expected = [
            "scenarios",
            "scenario_statistics",
            "sensitivity",
            "risk_simulation",
            "decision_optimization",
        ]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        optimization = result["results"]["optimization"]
        assert optimization["solver_status"] == "OPTIMAL"
        assert optimization["global_optimal_proven"] is True
        assert optimization["exhaustive_cross_check"]["optional_node_count"] == 4
        assert optimization["exhaustive_cross_check"]["passed"] is True
        assert optimization["constrained_optimization"] is True
        assert result["results"]["final_stage"] == "decision_optimization"
        assert result["results"]["final_result"]["success"] is True
        assert "solution" in result["results"]["final_result"]
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert state["schema_version"] == "compute-dynamic-pipeline-state-v2"
        assert state["status"] == "PASS"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["network_used"] is False
        assert audit["model_calls"] == 0
        assert audit["global_optimal_proven"] is True
        assert result["quality"]["observed_evidence_maturity"] == "controlled-preview"
        return {
            "stage_order": expected,
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "final_stage": result["results"]["final_stage"],
            "release_status": result["quality"]["release_status"],
        }
    finally:
        ticket_path.unlink(missing_ok=True)
        if root.exists():
            shutil.rmtree(root)


def main() -> int:
    _assert_source_safety()
    _assert_policy_and_graph()
    plans = _assert_optimal_plans()
    dispatcher = _assert_dispatcher()
    print(json.dumps({
        "status": "PASS",
        "generic_optional_node_count": 4,
        "global_optimal_proven": True,
        "plans": plans,
        "dispatcher": dispatcher,
        "network_used": False,
        "model_calls": 0,
        "automatic_parallel_execution": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
