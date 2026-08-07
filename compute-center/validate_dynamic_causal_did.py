#!/usr/bin/env python3
"""Acceptance validator for the explicit DID causal dynamic family."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_causal_did_planner import plan_dynamic_causal_did
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _ticket(*, advanced: bool = True) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "treated_pre": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "treated_post": [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
        "control_pre": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        "control_post": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "bootstrap_samples": 300,
        "seed": 20260807,
        "pretrend_tolerance": 0.25,
    }
    if advanced:
        inputs["dynamic_context"] = {
            "causal_design": "difference_in_differences",
            "allow_causal_policy_evaluation": True,
        }
    return {
        "task_id": "dynamic-causal-did-production-sim-20260807" if advanced else "dynamic-causal-did-screening-sim-20260807",
        "objective": "Validate the explicit DID causal family production path without inferring an identification strategy from text.",
        "operation": "causal_screening",
        "inputs": inputs,
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "causal DID family production dispatcher simulation",
            "upstream_refs": [],
        },
        "preflight_policy": {
            "enforcement": "advisory",
            "allow_assumptions": True,
            "allow_proxy": True,
            "require_user_approval_for_low_confidence": False,
            "max_assumption_ratio": 1.0,
            "small_sample_threshold": 2,
            "outlier_rate_threshold": 1.0,
        },
        "quality_profile": {
            "decision_class": "exploratory",
            "probabilistic_claim": False,
            "benchmark_ids": [],
            "method_ids": [],
            "sample_ids": [],
            "rule_ids": [],
            "independent_cross_check_passed": False,
            "publication_policy": "status_only",
        },
    }


def _assert_source_safety() -> None:
    for name in ("dynamic_family_router.py", "dynamic_causal_did_planner.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        for forbidden in ("import requests", "import socket", "urllib.request", "subprocess.", "pickle.loads", "import mcp"):
            assert forbidden not in text, (name, forbidden)
        tree = ast.parse(text)
        direct = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert {"eval", "exec", "compile"}.isdisjoint(direct), name


def _assert_router_and_dependencies() -> None:
    screening = _ticket(advanced=False)
    advanced = _ticket(advanced=True)
    assert resolve_dynamic_family(screening) == "causal-did"
    assert resolve_dynamic_family(advanced) == "causal-did"
    assert [Path(item).name for item in requirement_files_for_ticket(screening)] == ["requirements-ortools.txt"]
    assert [Path(item).name for item in requirement_files_for_ticket(advanced)] == ["requirements-ortools.txt", "requirements-causal.txt"]
    metadata = family_runtime_metadata(advanced)
    assert metadata["family"] == "causal-did"
    assert metadata["entry_contract"] == "causal_screening"
    assert metadata["causal_design"] == "difference_in_differences"
    assert metadata["advanced_requested"] is True
    assert metadata["extra_requirements"] == ["requirements-causal.txt"]
    runtime = managed_runtime_plan(advanced)
    assert runtime["dynamic_family"] == "causal-did"
    assert runtime["dynamic_extra_requirements"] == ["requirements-causal.txt"]
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"

    invalid = _ticket(advanced=True)
    invalid["inputs"]["dynamic_context"]["causal_design"] = "instrumental_variable"
    try:
        requirement_files_for_ticket(invalid)
    except DynamicFamilyRoutingError:
        pass
    else:
        raise AssertionError("wrong causal design did not fail closed before dependency install")


def _assert_plans() -> None:
    screening = plan_dynamic_causal_did(_ticket(advanced=False))
    advanced = plan_dynamic_causal_did(_ticket(advanced=True))
    assert screening["stage_order"] == ["screening"]
    assert advanced["stage_order"] == ["screening", "did_policy_evaluation"]
    for plan in (screening, advanced):
        optimization = plan["optimization"]
        cross = optimization["exhaustive_cross_check"]
        assert plan["objective_text_used"] is False
        assert optimization["solver_status"] == "OPTIMAL"
        assert optimization["global_optimal_proven"] is True
        assert cross["performed"] is True
        assert cross["optional_node_count"] == 1
        assert cross["passed"] is True
        assert optimization["objective_value"] == cross["best_objective"]
        assert plan["network_policy"] == "deny"
        assert plan["automatic_parallel_execution"] is False
        assert plan["model_calls"] == 0


def _assert_dispatcher() -> dict[str, Any]:
    ticket = _ticket(advanced=True)
    root = HERE / "dynamic-causal-did-production-sim"
    ticket_path = HERE / "dynamic-causal-did-production-ticket.json"
    if root.exists():
        shutil.rmtree(root)
    ticket_path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        return_code = compute_dispatch.main(["--ticket", str(ticket_path), "--output-dir", str(root)])
        assert return_code == 0
        result = json.loads((root / "compute-result.json").read_text(encoding="utf-8"))
        state = json.loads((root / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "compute-audit.json").read_text(encoding="utf-8"))
        expected = ["screening", "did_policy_evaluation"]
        assert result["status"] == "success"
        assert result["operation"] == "causal_screening"
        assert result["results"]["dynamic_family"] == "causal-did"
        assert result["results"]["stage_order"] == expected
        optimization = result["results"]["optimization"]
        assert optimization["solver_status"] == "OPTIMAL"
        assert optimization["global_optimal_proven"] is True
        assert optimization["exhaustive_cross_check"]["passed"] is True
        assert optimization["selected_nodes"]["did_policy_evaluation"] is True
        assert len(result["results"]["stage_receipts"]) == 2
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        screening = result["results"]["stage_outputs"]["screening"]
        final = result["results"]["final_result"]
        assert screening["method"] == "difference_in_differences_screening"
        assert "not proof of causality" in screening["warning"]
        assert final["mode"] == "difference_in_differences_refuted"
        assert final["engine"]["network_used"] is False
        assert final["parallel_trends_passed"] is True
        assert final["causal_claim_allowed"] is True
        assert final["claim_type"] == "causal_effect"
        assert "Causal language is permitted only" in final["interpretation_boundary"]
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert state["status"] == "PASS"
        assert state["family"] == "causal-did"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["status"] == "PASS"
        assert audit["dynamic_family"] == "causal-did"
        assert audit["global_optimal_proven"] is True
        assert audit["network_used"] is False
        assert audit["model_calls"] == 0
        assert result["quality"]["observed_evidence_maturity"] == "controlled-preview"
        return {
            "stage_order": expected,
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "screening_status": screening["status"],
            "effect_estimate": final["effect"],
            "parallel_trends_passed": final["parallel_trends_passed"],
            "causal_claim_allowed": final["causal_claim_allowed"],
            "claim_type": final["claim_type"],
            "release_status": result["quality"]["release_status"],
        }
    finally:
        ticket_path.unlink(missing_ok=True)
        if root.exists():
            shutil.rmtree(root)


def main() -> int:
    _assert_source_safety()
    _assert_router_and_dependencies()
    _assert_plans()
    dispatcher = _assert_dispatcher()
    print(json.dumps({
        "status": "PASS",
        "dynamic_family": "causal-did",
        "global_optimal_proven": True,
        "dispatcher": dispatcher,
        "network_used": False,
        "model_calls": 0,
        "automatic_parallel_execution": False,
        "objective_text_routing": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
