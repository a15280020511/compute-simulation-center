#!/usr/bin/env python3
"""Acceptance validator for the sample-based dynamic reliability family."""
from __future__ import annotations

import ast
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import compute_dispatch
from dynamic_family_router import DynamicFamilyRoutingError, family_runtime_metadata, resolve_dynamic_family
from dynamic_reliability_planner import plan_dynamic_reliability
from tool_registry import managed_runtime_plan, requirement_files_for_ticket

HERE = Path(__file__).resolve().parent


def _data() -> list[float]:
    return [value for _ in range(8) for value in (8.0, 9.0, 10.0, 11.0, 12.0)]


def _ticket() -> dict[str, Any]:
    return {
        "task_id": "dynamic-reliability-production-sim-20260808",
        "objective": "Validate sample-based reliability orchestration from structured inputs only.",
        "operation": "descriptive_statistics",
        "inputs": {
            "data": _data(),
            "reliability_context": {
                "threshold": 8.0,
                "tail": "lower",
                "monte_carlo_crosscheck": True,
                "monte_carlo_iterations": 50000,
                "monte_carlo_seed": 7,
                "mc_agreement_tolerance": 0.02,
                "external_failure_probability": 0.079,
                "external_benchmark_tolerance": 0.02
            }
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "Sample statistics feed analytic and Monte Carlo reliability branches before agreement and external benchmark validation.",
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
            "probabilistic_claim": True,
            "benchmark_ids": [],
            "method_ids": [],
            "sample_ids": [],
            "rule_ids": [],
            "independent_cross_check_passed": False,
            "publication_policy": "status_only"
        }
    }


def _assert_source_safety() -> None:
    for name in ("dynamic_family_router.py", "dynamic_reliability_adapters.py", "dynamic_reliability_planner.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        for forbidden in ("import requests", "import socket", "urllib.request", "subprocess.", "pickle.loads", "import mcp"):
            assert forbidden not in text, (name, forbidden)
        tree = ast.parse(text)
        direct = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert {"eval", "exec", "compile"}.isdisjoint(direct), name


def _assert_fail_closed_router() -> None:
    invalid = _ticket()
    invalid["task_id"] = "dynamic-reliability-invalid"
    invalid["inputs"].pop("reliability_context")
    try:
        requirement_files_for_ticket(invalid)
    except DynamicFamilyRoutingError:
        return
    raise AssertionError("reliability request without structured context did not fail closed")


def main() -> int:
    _assert_source_safety()
    _assert_fail_closed_router()
    ticket = _ticket()
    assert resolve_dynamic_family(ticket) == "reliability"
    family = family_runtime_metadata(ticket)
    assert family == {
        "family": "reliability",
        "entry_contract": "descriptive_statistics:sample-normal-reliability",
        "policy_file": "dynamic-reliability-policy.json",
        "graph_file": "dynamic-reliability-capability-graph.json",
        "python_version": "3.12",
        "requirements": ["requirements-global-openturns.txt"],
    }
    requirements = [Path(item).name for item in requirement_files_for_ticket(ticket)]
    assert requirements == ["requirements-ortools.txt", "requirements-global-openturns.txt"]
    runtime = managed_runtime_plan(ticket)
    assert runtime["capability_pack"] == "dynamic-orchestration"
    assert runtime["dynamic_family"] == "reliability"
    assert runtime["dynamic_entry_contract"] == "descriptive_statistics:sample-normal-reliability"
    assert runtime["python_version"] == "3.12"
    assert runtime["network_policy"] == "deny"
    assert runtime["selection_engine"] == "ortools-cp-sat"
    assert runtime["graph_engine"] == "networkx"
    assert runtime["automatic_parallel_execution"] is False
    assert platform.python_version_tuple()[:2] == ("3", "12")

    plan = plan_dynamic_reliability(ticket)
    expected = ["sample_statistics", "analytic_reliability", "monte_carlo_validation", "analytic_mc_agreement", "external_benchmark"]
    assert plan["stage_order"] == expected
    assert plan["objective_text_used"] is False
    assert plan["stage_map"]["sample_statistics"]["depends_on"] == []
    assert plan["stage_map"]["analytic_reliability"]["depends_on"] == ["sample_statistics"]
    assert plan["stage_map"]["monte_carlo_validation"]["depends_on"] == ["sample_statistics"]
    assert plan["stage_map"]["analytic_mc_agreement"]["depends_on"] == ["analytic_reliability", "monte_carlo_validation"]
    assert plan["stage_map"]["external_benchmark"]["depends_on"] == ["analytic_reliability"]
    optimization = plan["optimization"]
    assert optimization["solver_status"] == "OPTIMAL"
    assert optimization["global_optimal_proven"] is True
    assert optimization["selected_nodes"]["monte_carlo_validation"] is True
    assert optimization["selected_nodes"]["analytic_mc_agreement"] is True
    assert optimization["selected_nodes"]["external_benchmark"] is True
    cross = optimization["exhaustive_cross_check"]
    assert cross["performed"] is True and cross["passed"] is True and cross["unique_optimum"] is True
    assert optimization["objective_value"] == cross["best_objective"]

    root = HERE / "dynamic-reliability-production-sim"
    ticket_path = HERE / "dynamic-reliability-production-ticket.json"
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
        assert result["operation"] == "descriptive_statistics"
        assert result["results"]["dynamic_family"] == "reliability"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["stage_dependencies"]["analytic_mc_agreement"] == ["analytic_reliability", "monte_carlo_validation"]
        analytic = result["results"]["final_result"]
        self_check = result["results"]["validation_results"]["analytic_mc_agreement"]
        external = result["results"]["validation_results"]["external_benchmark"]
        assert analytic["mode"] == "openturns_reliability_probability"
        assert analytic["distribution"] == "normal"
        assert 0 <= analytic["failure_probability"] <= 1
        assert self_check["status"] == "PASS"
        assert external["status"] == "PASS"
        assert len(result["results"]["stage_receipts"]) == 5
        assert all(row["status"] == "PASS" for row in result["results"]["stage_receipts"])
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        assert state["status"] == "PASS" and state["family"] == "reliability"
        assert [row["stage_id"] for row in state["stages"]] == expected
        assert all(row["status"] == "PASS" for row in state["stages"])
        assert all("input_sha256" in row and "output_sha256" in row for row in state["stages"])
        assert audit["status"] == "PASS"
        assert audit["dynamic_family"] == "reliability"
        assert audit["solver_status"] == "OPTIMAL"
        assert audit["global_optimal_proven"] is True
        assert audit["graph_contains_branching"] is True
        assert audit["distribution_assumption"] == "normal-from-sample-mean-and-population-standard-deviation"
        assert audit["network_used"] is False and audit["model_calls"] == 0
        print(json.dumps({
            "status": "PASS",
            "dynamic_family": "reliability",
            "python_version": runtime["python_version"],
            "stage_order": expected,
            "stage_dependencies": result["results"]["stage_dependencies"],
            "solver_status": optimization["solver_status"],
            "objective_value": optimization["objective_value"],
            "exhaustive_best_objective": cross["best_objective"],
            "global_optimal_proven": optimization["global_optimal_proven"],
            "analytic_failure_probability": analytic["failure_probability"],
            "analytic_mc_agreement": self_check["status"],
            "external_benchmark": external["status"],
            "graph_contains_branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"]
        }, ensure_ascii=False, indent=2))
    finally:
        ticket_path.unlink(missing_ok=True)
        if root.exists():
            shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
