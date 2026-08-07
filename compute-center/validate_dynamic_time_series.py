#!/usr/bin/env python3
"""Acceptance validator for the structured dynamic time-series family."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from dynamic_time_series_planner import plan_dynamic_time_series
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _ticket() -> dict[str, Any]:
    data = [100.0, 102.0, 101.5, 104.0, 105.5, 107.0, 108.0, 109.5, 111.0, 112.0, 114.0, 115.5]
    return {
        "task_id": "dynamic-time-series-production-sim-20260807",
        "objective": "Validate the structured policy-optimal time-series family production path.",
        "operation": "time_series_forecast",
        "inputs": {
            "data": data,
            "horizon": 5,
            "expected_mean": sum(data) / len(data),
            "mean_tolerance": 20.0,
            "dynamic_context": {"time_series_diagnostics": True},
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "time-series family production dispatcher simulation",
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
    for name in ("dynamic_family_router.py", "dynamic_time_series_planner.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        for forbidden in ("import requests", "import socket", "urllib.request", "subprocess.", "pickle.loads", "import mcp"):
            assert forbidden not in text, (name, forbidden)
        tree = ast.parse(text)
        direct = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert {"eval", "exec", "compile"}.isdisjoint(direct), name


def _assert_fail_closed_router() -> None:
    invalid = {
        "task_id": "dynamic-time-series-invalid-family",
        "operation": "pattern_discovery",
        "inputs": {"data": [float(index) for index in range(12)]},
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "must fail closed",
            "upstream_refs": [],
        },
    }
    try:
        requirement_files_for_ticket(invalid)
    except DynamicFamilyRoutingError:
        return
    raise AssertionError("unadmitted dynamic operation did not fail closed before dependency install")


def main() -> int:
    _assert_source_safety()
    _assert_fail_closed_router()
    ticket = _ticket()
    assert resolve_dynamic_family(ticket) == "time-series"
    family = family_runtime_metadata(ticket)
    assert family == {
        "family": "time-series",
        "entry_contract": "time_series_forecast",
        "policy_file": "dynamic-time-series-policy.json",
        "graph_file": "dynamic-time-series-capability-graph.json",
        "extra_requirements": [],
    }
    requirements = requirement_files_for_ticket(ticket)
    assert len(requirements) == 1
    assert Path(requirements[0]).name == "requirements-ortools.txt"
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["dynamic_family"] == "time-series"
    assert runtime["dynamic_entry_contract"] == "time_series_forecast"
    assert runtime["dynamic_extra_requirements"] == []
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"
    assert runtime["automatic_parallel_execution"] is False

    plan = plan_dynamic_time_series(ticket)
    expected = ["series_statistics", "pattern_diagnostics", "assumption_checks", "forecast"]
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

    root = HERE / "dynamic-time-series-production-sim"
    ticket_path = HERE / "dynamic-time-series-production-ticket.json"
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
        assert result["operation"] == "time_series_forecast"
        assert result["results"]["dynamic_family"] == "time-series"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["final_stage"] == "forecast"
        assert len(result["results"]["final_result"]["forecast"]) == 5
        assert len(result["results"]["stage_receipts"]) == 4
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert state["status"] == "PASS"
        assert state["family"] == "time-series"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["status"] == "PASS"
        assert audit["dynamic_family"] == "time-series"
        assert audit["solver_status"] == "OPTIMAL"
        assert audit["global_optimal_proven"] is True
        assert audit["network_used"] is False
        assert audit["model_calls"] == 0
        assert result["quality"]["observed_evidence_maturity"] == "controlled-preview"
        print(json.dumps({
            "status": "PASS",
            "dynamic_family": "time-series",
            "stage_order": expected,
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "exhaustive_best_objective": cross["best_objective"],
            "global_optimal_proven": optimization["global_optimal_proven"],
            "forecast_points": len(result["results"]["final_result"]["forecast"]),
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
