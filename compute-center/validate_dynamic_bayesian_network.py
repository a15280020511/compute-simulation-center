#!/usr/bin/env python3
"""Acceptance validator for the structured dynamic Bayesian-network family."""
from __future__ import annotations

import ast
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_bayesian_network_planner import plan_dynamic_bayesian_network
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _ticket() -> dict[str, Any]:
    count = 80
    a = [index % 2 for index in range(count)]
    b = [a[index] if index % 5 else 1 - a[index] for index in range(count)]
    c = [b[index] if index % 7 else 1 - b[index] for index in range(count)]
    return {
        "task_id": "dynamic-bayesian-production-sim-20260808",
        "objective": "Validate structured policy-optimal Bayesian DAG orchestration without objective-text routing.",
        "operation": "bayesian_network_inference",
        "inputs": {
            "mode": "bayesian_parameter_estimation",
            "edges": [["A", "B"], ["B", "C"]],
            "data": {"A": a, "B": b, "C": c},
            "query_variables": ["C"],
            "evidence": {"A": 1},
            "equivalent_sample_size": 5.0,
            "evidence_scenarios": [
                {"name": "A=0", "evidence": {"A": 0}},
                {"name": "A=1", "evidence": {"A": 1}}
            ],
            "virtual_evidence": [
                {"variable": "B", "probabilities": [0.25, 0.75], "state_names": [0, 1]}
            ],
            "dynamic_context": {
                "evidence_sensitivity": True,
                "virtual_evidence_update": True
            }
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "Bayesian family production dispatcher simulation with one branching dependency DAG executed strictly serially.",
            "upstream_refs": []
        },
        "preflight_policy": {
            "enforcement": "advisory",
            "allow_assumptions": True,
            "allow_proxy": True,
            "require_user_approval_for_low_confidence": False,
            "max_assumption_ratio": 1.0,
            "small_sample_threshold": 2,
            "outlier_rate_threshold": 1.0
        },
        "quality_profile": {
            "decision_class": "exploratory",
            "probabilistic_claim": False,
            "benchmark_ids": [],
            "method_ids": [],
            "sample_ids": [],
            "rule_ids": [],
            "independent_cross_check_passed": False,
            "publication_policy": "status_only"
        }
    }


def _assert_source_safety() -> None:
    for name in (
        "dynamic_family_router.py",
        "dynamic_bayesian_adapters.py",
        "dynamic_bayesian_network_planner.py",
    ):
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


def _assert_fail_closed_router() -> None:
    invalid = _ticket()
    invalid["task_id"] = "dynamic-bayesian-invalid-mode"
    invalid["inputs"]["mode"] = "fixed_network_inference"
    try:
        requirement_files_for_ticket(invalid)
    except DynamicFamilyRoutingError:
        return
    raise AssertionError("unadmitted Bayesian dynamic entry mode did not fail closed before dependency install")


def main() -> int:
    _assert_source_safety()
    _assert_fail_closed_router()
    ticket = _ticket()
    assert resolve_dynamic_family(ticket) == "bayesian-network"
    family = family_runtime_metadata(ticket)
    assert family == {
        "family": "bayesian-network",
        "entry_contract": "bayesian_network_inference",
        "policy_file": "dynamic-bayesian-policy.json",
        "graph_file": "dynamic-bayesian-capability-graph.json",
        "python_version": "3.12",
        "requirements": ["requirements-bayesian-network.txt"],
    }
    requirements = requirement_files_for_ticket(ticket)
    assert [Path(item).name for item in requirements] == [
        "requirements-ortools.txt",
        "requirements-bayesian-network.txt",
    ]
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["dynamic_family"] == "bayesian-network"
    assert runtime["dynamic_entry_contract"] == "bayesian_network_inference"
    assert runtime["python_version"] == "3.12"
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"
    assert runtime["automatic_parallel_execution"] is False
    assert platform.python_version_tuple()[:2] == ("3", "12")

    plan = plan_dynamic_bayesian_network(ticket)
    expected = [
        "parameter_estimation",
        "posterior_inference",
        "evidence_sensitivity",
        "virtual_evidence_update",
    ]
    assert plan["stage_order"] == expected
    assert plan["objective_text_used"] is False
    assert plan["stage_map"]["parameter_estimation"]["depends_on"] == []
    assert plan["stage_map"]["posterior_inference"]["depends_on"] == ["parameter_estimation"]
    assert plan["stage_map"]["evidence_sensitivity"]["depends_on"] == ["parameter_estimation"]
    assert plan["stage_map"]["virtual_evidence_update"]["depends_on"] == ["parameter_estimation"]
    optimization = plan["optimization"]
    assert optimization["solver_status"] == "OPTIMAL"
    assert optimization["global_optimal_proven"] is True
    cross = optimization["exhaustive_cross_check"]
    assert cross["performed"] is True
    assert cross["optional_node_count"] == 2
    assert cross["passed"] is True
    assert cross["unique_optimum"] is True
    assert optimization["objective_value"] == cross["best_objective"]

    root = HERE / "dynamic-bayesian-production-sim"
    ticket_path = HERE / "dynamic-bayesian-production-ticket.json"
    if root.exists():
        shutil.rmtree(root)
    ticket_path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        return_code = compute_dispatch.main(["--ticket", str(ticket_path), "--output-dir", str(root)])
        assert return_code == 0
        result = json.loads((root / "compute-result.json").read_text(encoding="utf-8"))
        state = json.loads((root / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "compute-audit.json").read_text(encoding="utf-8"))
        assert result["status"] == "success"
        assert result["operation"] == "bayesian_network_inference"
        assert result["results"]["dynamic_family"] == "bayesian-network"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["stage_dependencies"] == {
            "parameter_estimation": [],
            "posterior_inference": ["parameter_estimation"],
            "evidence_sensitivity": ["parameter_estimation"],
            "virtual_evidence_update": ["parameter_estimation"],
        }
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["final_stage"] == "posterior_inference"
        assert result["results"]["final_result"]["mode"] == "fixed_network_inference"
        assert result["results"]["final_result"]["model_valid"] is True
        assert result["results"]["causal_structure_claimed"] is False
        assert set(result["results"]["robustness_results"]) == {
            "evidence_sensitivity",
            "virtual_evidence_update",
        }
        assert result["results"]["robustness_results"]["evidence_sensitivity"]["scenario_count"] == 2
        assert result["results"]["robustness_results"]["virtual_evidence_update"]["virtual_evidence_count"] == 1
        assert len(result["results"]["stage_receipts"]) == 4
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        assert result["software"]["pgmpy"] == "1.1.2"
        assert state["status"] == "PASS"
        assert state["family"] == "bayesian-network"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["status"] == "PASS"
        assert audit["dynamic_family"] == "bayesian-network"
        assert audit["solver_status"] == "OPTIMAL"
        assert audit["global_optimal_proven"] is True
        assert audit["graph_contains_branching"] is True
        assert audit["causal_structure_claimed"] is False
        assert audit["network_used"] is False
        assert audit["model_calls"] == 0
        print(json.dumps({
            "status": "PASS",
            "dynamic_family": "bayesian-network",
            "python_version": runtime["python_version"],
            "stage_order": expected,
            "stage_dependencies": result["results"]["stage_dependencies"],
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "exhaustive_best_objective": cross["best_objective"],
            "global_optimal_proven": optimization["global_optimal_proven"],
            "graph_contains_branching": result["execution"]["graph_contains_branching"],
            "causal_structure_claimed": result["results"]["causal_structure_claimed"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"],
        }, ensure_ascii=False, indent=2))
    finally:
        ticket_path.unlink(missing_ok=True)
        if root.exists():
            shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
