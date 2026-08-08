#!/usr/bin/env python3
"""Acceptance validator for the structured dynamic causal-policy family."""
from __future__ import annotations

import ast
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_causal_policy_planner import plan_dynamic_causal_policy
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _ticket() -> dict[str, Any]:
    treatment = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    age = [20, 21, 24, 23, 27, 29, 31, 30, 34, 36, 39, 38, 42, 41, 45, 47, 49, 50, 53, 55, 57, 58, 61, 63]
    baseline = [1.0, -0.4, 0.8, -0.2, 0.5, -0.1, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.4, -0.2, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.2, 0.1, -0.1]
    outcome = [10.0 + 2.5 * t + 0.08 * a + e for t, a, e in zip(treatment, age, baseline, strict=True)]
    return {
        "task_id": "dynamic-causal-production-sim-20260808",
        "objective": "Validate structured policy-optimal causal orchestration without objective-text routing.",
        "operation": "causal_policy_evaluation",
        "inputs": {
            "mode": "backdoor_adjustment",
            "treatment": treatment,
            "outcome": outcome,
            "confounders": {"age": age},
            "dynamic_context": {
                "causal_diagnostics": True,
                "causal_refutation": True
            }
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "causal family production dispatcher simulation",
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
            "decision_class": "high_stakes",
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
    for name in ("dynamic_family_router.py", "dynamic_causal_policy_planner.py", "pipeline_adapters.py"):
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
    invalid["task_id"] = "dynamic-causal-invalid-mode"
    invalid["inputs"]["mode"] = "difference_in_differences_refuted"
    try:
        requirement_files_for_ticket(invalid)
    except DynamicFamilyRoutingError:
        return
    raise AssertionError("unadmitted causal dynamic mode did not fail closed before dependency install")


def main() -> int:
    _assert_source_safety()
    _assert_fail_closed_router()
    ticket = _ticket()
    assert resolve_dynamic_family(ticket) == "causal-policy"
    family = family_runtime_metadata(ticket)
    assert family == {
        "family": "causal-policy",
        "entry_contract": "causal_policy_evaluation",
        "policy_file": "dynamic-causal-policy.json",
        "graph_file": "dynamic-causal-capability-graph.json",
        "python_version": "3.13",
        "requirements": ["requirements-causal.txt"],
    }
    requirements = requirement_files_for_ticket(ticket)
    assert [Path(item).name for item in requirements] == ["requirements-ortools.txt", "requirements-causal.txt"]
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["dynamic_family"] == "causal-policy"
    assert runtime["dynamic_entry_contract"] == "causal_policy_evaluation"
    assert runtime["python_version"] == "3.13"
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"
    assert runtime["automatic_parallel_execution"] is False
    assert platform.python_version_tuple()[:2] == ("3", "13")

    plan = plan_dynamic_causal_policy(ticket)
    expected = ["outcome_statistics", "causal_estimate", "placebo_refutation"]
    assert plan["stage_order"] == expected
    assert plan["objective_text_used"] is False
    optimization = plan["optimization"]
    assert optimization["solver_status"] == "OPTIMAL"
    assert optimization["global_optimal_proven"] is True
    cross = optimization["exhaustive_cross_check"]
    assert cross["performed"] is True
    assert cross["optional_node_count"] == 2
    assert cross["passed"] is True
    assert optimization["objective_value"] == cross["best_objective"]

    root = HERE / "dynamic-causal-production-sim"
    ticket_path = HERE / "dynamic-causal-production-ticket.json"
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
        assert result["operation"] == "causal_policy_evaluation"
        assert result["results"]["dynamic_family"] == "causal-policy"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["final_stage"] == "causal_estimate"
        assert result["results"]["final_result"]["mode"] == "backdoor_adjustment"
        assert result["results"]["final_result"]["identified"] is True
        assert result["results"]["refutation_result"]["mode"] == "placebo_policy_test"
        assert result["results"]["refutation_result"]["repetitions"] == 200
        assert len(result["results"]["stage_receipts"]) == 3
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["software"]["dowhy"] == "0.14"
        assert state["status"] == "PASS"
        assert state["family"] == "causal-policy"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["status"] == "PASS"
        assert audit["dynamic_family"] == "causal-policy"
        assert audit["solver_status"] == "OPTIMAL"
        assert audit["global_optimal_proven"] is True
        assert audit["network_used"] is False
        assert audit["model_calls"] == 0
        print(json.dumps({
            "status": "PASS",
            "dynamic_family": "causal-policy",
            "python_version": runtime["python_version"],
            "stage_order": expected,
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "exhaustive_best_objective": cross["best_objective"],
            "global_optimal_proven": optimization["global_optimal_proven"],
            "causal_claim_allowed": result["results"]["final_result"]["causal_claim_allowed"],
            "refutation_passed": result["results"]["refutation_result"]["refutation_passed"],
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
