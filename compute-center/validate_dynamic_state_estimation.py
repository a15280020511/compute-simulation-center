#!/usr/bin/env python3
"""Acceptance validator for the structured dynamic state-estimation family."""
from __future__ import annotations

import ast
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from dynamic_state_estimation_planner import plan_dynamic_state_estimation
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _ticket() -> dict[str, Any]:
    return {
        "task_id": "dynamic-state-estimation-production-sim-20260808",
        "objective": "Validate bounded state-estimation orchestration from structured matrices only.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "bounded_linear_kalman_filter",
            "transition_matrix": [[1.0]],
            "observation_matrix": [[1.0]],
            "process_covariance": [[0.05]],
            "observation_covariance": [[0.2]],
            "initial_covariance": [[1.0]],
            "initial_state": [0.0],
            "observations": [[1.0], [1.4], [1.9], [2.3], [2.8], [3.2]],
            "drift_ratio_threshold": 4.0,
            "benchmark_state": [3.0],
            "benchmark_tolerance": 0.6,
            "dynamic_context": {
                "realized_feedback": True,
                "benchmark_check": True
            }
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "State estimation plus two validation branches executed strictly serially.",
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
        "dynamic_state_estimation_adapters.py",
        "dynamic_state_estimation_planner.py",
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
    invalid["task_id"] = "dynamic-state-estimation-invalid-mode"
    invalid["inputs"]["mode"] = "realized_outcome_feedback"
    try:
        requirement_files_for_ticket(invalid)
    except DynamicFamilyRoutingError:
        return
    raise AssertionError("unadmitted state-estimation dynamic entry mode did not fail closed")


def main() -> int:
    _assert_source_safety()
    _assert_fail_closed_router()
    ticket = _ticket()
    assert resolve_dynamic_family(ticket) == "state-estimation"
    family = family_runtime_metadata(ticket)
    assert family == {
        "family": "state-estimation",
        "entry_contract": "finance_decision_analysis:bounded_linear_kalman_filter",
        "policy_file": "dynamic-state-estimation-policy.json",
        "graph_file": "dynamic-state-estimation-capability-graph.json",
        "python_version": "3.12",
        "requirements": [],
    }
    requirements = requirement_files_for_ticket(ticket)
    assert [Path(item).name for item in requirements] == ["requirements-ortools.txt"]
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["dynamic_family"] == "state-estimation"
    assert runtime["dynamic_entry_contract"] == "finance_decision_analysis:bounded_linear_kalman_filter"
    assert runtime["python_version"] == "3.12"
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"
    assert runtime["automatic_parallel_execution"] is False
    assert platform.python_version_tuple()[:2] == ("3", "12")

    plan = plan_dynamic_state_estimation(ticket)
    expected = ["state_estimation", "realized_feedback", "benchmark_check"]
    assert plan["stage_order"] == expected
    assert plan["objective_text_used"] is False
    assert plan["stage_map"]["state_estimation"]["depends_on"] == []
    assert plan["stage_map"]["realized_feedback"]["depends_on"] == ["state_estimation"]
    assert plan["stage_map"]["benchmark_check"]["depends_on"] == ["state_estimation"]
    optimization = plan["optimization"]
    assert optimization["solver_status"] == "OPTIMAL"
    assert optimization["global_optimal_proven"] is True
    cross = optimization["exhaustive_cross_check"]
    assert cross["performed"] is True
    assert cross["optional_node_count"] == 2
    assert cross["passed"] is True
    assert cross["unique_optimum"] is True
    assert optimization["objective_value"] == cross["best_objective"]

    root = HERE / "dynamic-state-estimation-production-sim"
    ticket_path = HERE / "dynamic-state-estimation-production-ticket.json"
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
        assert result["operation"] == "finance_decision_analysis"
        assert result["results"]["dynamic_family"] == "state-estimation"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["stage_dependencies"] == {
            "state_estimation": [],
            "realized_feedback": ["state_estimation"],
            "benchmark_check": ["state_estimation"],
        }
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["final_stage"] == "state_estimation"
        assert result["results"]["final_result"]["mode"] == "bounded_linear_kalman_filter"
        assert result["results"]["final_result"]["fixed_offline_generic_state_estimation"] is True
        assert result["results"]["validation_results"]["realized_feedback"]["mode"] == "realized_outcome_feedback"
        assert result["results"]["validation_results"]["benchmark_check"]["mode"] == "benchmark_comparison"
        assert len(result["results"]["stage_receipts"]) == 3
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        assert state["status"] == "PASS"
        assert state["family"] == "state-estimation"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["status"] == "PASS"
        assert audit["dynamic_family"] == "state-estimation"
        assert audit["solver_status"] == "OPTIMAL"
        assert audit["global_optimal_proven"] is True
        assert audit["graph_contains_branching"] is True
        assert audit["fixed_offline_generic_state_estimation"] is True
        assert audit["live_feed_used"] is False
        assert audit["individual_or_target_tracking_allowed"] is False
        assert audit["network_used"] is False
        assert audit["model_calls"] == 0
        print(json.dumps({
            "status": "PASS",
            "dynamic_family": "state-estimation",
            "python_version": runtime["python_version"],
            "stage_order": expected,
            "stage_dependencies": result["results"]["stage_dependencies"],
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "exhaustive_best_objective": cross["best_objective"],
            "global_optimal_proven": optimization["global_optimal_proven"],
            "graph_contains_branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"],
            "live_feed_used": False,
        }, ensure_ascii=False, indent=2))
    finally:
        ticket_path.unlink(missing_ok=True)
        if root.exists():
            shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
