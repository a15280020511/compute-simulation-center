#!/usr/bin/env python3
"""Acceptance validator for the structured dynamic causal-policy family."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_causal_policy_planner import plan_dynamic_causal_policy
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _ticket() -> dict[str, Any]:
    count = 40
    confounder = [(index % 10) / 9.0 for index in range(count)]
    treatment = [
        1 if ((index * 7) % 10) < (3 + int(4 * confounder[index])) else 0
        for index in range(count)
    ]
    outcome = [
        2.5 * treatment[index]
        + 1.2 * confounder[index]
        + ((index % 3) - 1) * 0.05
        for index in range(count)
    ]
    return {
        "task_id": "dynamic-causal-production-sim-20260808",
        "objective": "Validate the structured policy-optimal causal family production path.",
        "operation": "causal_policy_evaluation",
        "inputs": {
            "mode": "backdoor_adjustment",
            "treatment": treatment,
            "outcome": outcome,
            "confounders": {"baseline_risk": confounder},
            "dynamic_context": {
                "causal_diagnostics": True,
                "estimator_cross_check": True,
                "placebo_refutation": True,
                "placebo_repetitions": 40,
                "placebo_seed": 17
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
        "dynamic_family_engine.py",
        "dynamic_causal_adapters.py",
        "dynamic_causal_policy_planner.py",
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
    ticket = _ticket()
    ticket["task_id"] = "dynamic-causal-invalid-mode"
    ticket["inputs"]["mode"] = "difference_in_differences_refuted"
    try:
        requirement_files_for_ticket(ticket)
    except DynamicFamilyRoutingError:
        return
    raise AssertionError("unadmitted dynamic causal mode did not fail closed before dependency install")


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
    }
    requirements = requirement_files_for_ticket(ticket)
    assert [Path(item).name for item in requirements] == [
        "requirements-ortools.txt",
        "requirements-causal.txt",
    ]
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["dynamic_family"] == "causal-policy"
    assert runtime["dynamic_entry_contract"] == "causal_policy_evaluation"
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"
    assert runtime["automatic_parallel_execution"] is False

    plan = plan_dynamic_causal_policy(ticket)
    expected = ["outcome_statistics", "alternate_estimate", "placebo_refutation", "primary_effect"]
    assert plan["stage_order"] == expected
    assert plan["objective_text_used"] is False
    optimization = plan["optimization"]
    assert optimization["solver_status"] == "OPTIMAL"
    assert optimization["global_optimal_proven"] is True
    cross = optimization["exhaustive_cross_check"]
    assert cross["performed"] is True
    assert cross["optional_node_count"] == 3
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
        assert result["results"]["final_stage"] == "primary_effect"
        assert result["results"]["final_result"]["mode"] == "backdoor_adjustment"
        assert len(result["results"]["stage_receipts"]) == 4
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        assert result["results"]["quality_gate"]["status"] in {"PASS", "WARN", "REJECT_CAUSAL_CLAIM"}
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
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
            "stage_order": expected,
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "exhaustive_best_objective": cross["best_objective"],
            "global_optimal_proven": optimization["global_optimal_proven"],
            "causal_quality_gate": result["results"]["quality_gate"],
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
