#!/usr/bin/env python3
"""Unified dynamic-family adapter for governed indirect intelligence fusion.

The adapter does not create a second stage optimizer. It validates the repository
capability graph, delegates the actual bounded stage selection and serial execution
to `indirect_intelligence_analysis`, then normalizes that engine's proven plan into
the common dynamic-family receipt/state contract.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import networkx as nx

from dynamic_family_router import resolve_dynamic_family
from indirect_intelligence_operations import ANALYSIS_CLASSES, MODE, STAGE_ORDER

HERE = Path(__file__).resolve().parent
GRAPH_PATH = HERE / "dynamic-indirect-intelligence-capability-graph.json"
FAMILY = "indirect-intelligence"
DECLARED_OPERATION = "finance_decision_analysis"


class DynamicIndirectIntelligenceError(ValueError):
    """Raised when the indirect-intelligence dynamic family contract is violated."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_graph() -> dict[str, Any]:
    value = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicIndirectIntelligenceError("indirect intelligence capability graph root must be an object")
    if value.get("schema_version") != "dynamic-indirect-intelligence-capability-graph-v1":
        raise DynamicIndirectIntelligenceError("indirect intelligence capability graph schema mismatch")
    if value.get("family") != FAMILY:
        raise DynamicIndirectIntelligenceError("indirect intelligence capability graph family mismatch")
    if value.get("entry_contract") != f"{DECLARED_OPERATION}:{MODE}":
        raise DynamicIndirectIntelligenceError("indirect intelligence capability graph entry contract mismatch")
    if value.get("network_policy") != "deny" or value.get("automatic_parallel_execution") is not False:
        raise DynamicIndirectIntelligenceError("indirect intelligence capability graph violates isolation policy")
    order = value.get("stage_order")
    if order != list(STAGE_ORDER):
        raise DynamicIndirectIntelligenceError("capability graph stage order disagrees with fusion engine")
    expected_edges = [list(edge) for edge in zip(STAGE_ORDER, STAGE_ORDER[1:], strict=False)]
    if value.get("edges") != expected_edges:
        raise DynamicIndirectIntelligenceError("capability graph edges disagree with fusion engine")
    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    graph.add_edges_from(tuple(edge) for edge in expected_edges)
    if not nx.is_directed_acyclic_graph(graph) or list(nx.topological_sort(graph)) != list(STAGE_ORDER):
        raise DynamicIndirectIntelligenceError("capability graph is not the declared deterministic DAG")
    result_contract = value.get("result_contract")
    if not isinstance(result_contract, Mapping):
        raise DynamicIndirectIntelligenceError("indirect intelligence result contract is missing")
    if set(result_contract.get("analysis_classes") or []) != ANALYSIS_CLASSES:
        raise DynamicIndirectIntelligenceError("indirect intelligence analysis class contract mismatch")
    if result_contract.get("inference_may_be_promoted_to_fact") is not False:
        raise DynamicIndirectIntelligenceError("inference-to-fact promotion must remain forbidden")
    if result_contract.get("scope_extrapolation_allowed") is not False:
        raise DynamicIndirectIntelligenceError("scope extrapolation must remain forbidden")
    return value


def _validate_engine_result(result: Mapping[str, Any], graph: Mapping[str, Any]) -> list[str]:
    if result.get("mode") != MODE:
        raise DynamicIndirectIntelligenceError("fusion engine returned the wrong mode")
    plan = result.get("stage_plan")
    if not isinstance(plan, Mapping):
        raise DynamicIndirectIntelligenceError("fusion engine omitted its stage plan")
    if plan.get("selection_engine") != "ortools-cp-sat" or plan.get("graph_engine") != "networkx":
        raise DynamicIndirectIntelligenceError("fusion engine selection/graph engines changed")
    if plan.get("solver_status") != "OPTIMAL":
        raise DynamicIndirectIntelligenceError("fusion engine must prove an OPTIMAL stage plan")
    if plan.get("serial_execution") is not True or plan.get("automatic_parallel_execution") is not False:
        raise DynamicIndirectIntelligenceError("fusion engine must remain strict serial")
    selected = plan.get("selected_stages")
    if not isinstance(selected, list) or not selected:
        raise DynamicIndirectIntelligenceError("fusion engine selected no stages")
    allowed_order = list(graph["stage_order"])
    if len(selected) > len(allowed_order) or any(stage not in allowed_order for stage in selected):
        raise DynamicIndirectIntelligenceError("fusion engine selected an undeclared stage")
    if selected != [stage for stage in allowed_order if stage in set(selected)]:
        raise DynamicIndirectIntelligenceError("fusion engine selected stages outside declared order")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(selected)
    runtime_graph.add_edges_from(zip(selected, selected[1:], strict=False))
    if not nx.is_directed_acyclic_graph(runtime_graph) or list(nx.topological_sort(runtime_graph)) != selected:
        raise DynamicIndirectIntelligenceError("fusion engine runtime stage order is not a deterministic DAG")
    analysis_class = str(result.get("analysis_class") or "")
    if analysis_class not in ANALYSIS_CLASSES:
        raise DynamicIndirectIntelligenceError("fusion engine returned an invalid analysis class")
    if analysis_class in {"LINKED", "INFERRED", "CONTRADICTED"} and result.get("inference_not_fact") is not True:
        raise DynamicIndirectIntelligenceError("non-direct output was not explicitly preserved as inference")
    if result.get("network_used") is not False or int(result.get("external_data_fetches") or 0) != 0:
        raise DynamicIndirectIntelligenceError("indirect intelligence dynamic execution attempted external data access")
    if int(result.get("model_calls") or 0) != 0:
        raise DynamicIndirectIntelligenceError("indirect intelligence compute family must not call models")
    if result.get("automatic_parallel_execution") is not False:
        raise DynamicIndirectIntelligenceError("indirect intelligence dynamic execution enabled parallel stages")
    if result.get("scope_extrapolation_allowed") is not False:
        raise DynamicIndirectIntelligenceError("indirect intelligence dynamic execution enabled scope extrapolation")
    if result.get("expert_semantic_synthesis_required_for_publication") is not True:
        raise DynamicIndirectIntelligenceError("expert semantic synthesis publication gate is missing")
    if result.get("governance_release_gate_required") is not True:
        raise DynamicIndirectIntelligenceError("governance release gate is missing")
    return [str(stage) for stage in selected]


def run_dynamic_indirect_intelligence_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicIndirectIntelligenceError("ticket is not an admitted indirect-intelligence dynamic request")
    handler = operations.get(DECLARED_OPERATION)
    if handler is None:
        raise DynamicIndirectIntelligenceError("finance_decision_analysis handler is unavailable")
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicIndirectIntelligenceError("ticket inputs must be an object")
    graph = _load_graph()
    output_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": "dynamic-auto-v1",
        "family": FAMILY,
        "status": "RUNNING",
        "planning_mode": "delegated-structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "automatic_parallel_execution": False,
        "network_used": False,
        "model_calls": 0,
        "delegate_operation": DECLARED_OPERATION,
        "delegate_mode": MODE,
        "stages": [],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    started = time.perf_counter()
    try:
        result = handler(inputs)
        if not isinstance(result, Mapping):
            raise DynamicIndirectIntelligenceError("fusion engine returned a non-object result")
        final_result = dict(result)
        selected = _validate_engine_result(final_result, graph)
    except Exception:
        state["status"] = "FAILED"
        _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise
    elapsed = time.perf_counter() - started
    stage_results = final_result.get("stage_results")
    if not isinstance(stage_results, Mapping):
        raise DynamicIndirectIntelligenceError("fusion engine omitted stage results")
    envelope_sha = _canonical_sha(inputs)
    receipts: list[dict[str, Any]] = []
    for stage in selected:
        stage_output = stage_results.get(stage)
        if not isinstance(stage_output, Mapping):
            raise DynamicIndirectIntelligenceError(f"fusion engine omitted selected stage output: {stage}")
        receipts.append(
            {
                "stage_id": stage,
                "status": "PASS",
                "delegate_operation": DECLARED_OPERATION,
                "delegate_mode": MODE,
                "delegated_input_envelope_sha256": envelope_sha,
                "output_sha256": _canonical_sha(stage_output),
            }
        )
    state.update(
        {
            "status": "PASS",
            "plan_sha256": _canonical_sha(final_result["stage_plan"]),
            "pipeline_sha256": _canonical_sha(receipts),
            "stages": receipts,
        }
    )
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)

    result_data = {
        "pipeline_id": "dynamic-auto-v1",
        "dynamic_family": FAMILY,
        "pipeline_maturity": "controlled-preview",
        "planning_mode": "delegated-structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "automatic_parallel_execution": False,
        "stage_order": selected,
        "stage_receipts": receipts,
        "final_stage": "fusion_result",
        "final_result": final_result,
        "inference_id": final_result.get("inference_id"),
        "analysis_class": final_result.get("analysis_class"),
        "prior_probability": final_result.get("prior_probability"),
        "posterior_probability": final_result.get("posterior_probability"),
        "confidence": final_result.get("confidence"),
        "inference_not_fact": final_result.get("inference_not_fact"),
        "scope_extrapolation_allowed": False,
    }
    transfer: dict[str, Any] = {
        "schema_version": "compute-result-v1",
        "task_id": str(ticket.get("task_id") or ""),
        "status": "success",
        "operation": str(ticket.get("operation") or ""),
        "objective": ticket.get("objective"),
        "input_sha256": _canonical_sha(ticket),
        "assumptions": ticket.get("assumptions", []),
        "evidence": ticket.get("evidence", []),
        "limitations": ticket.get("limitations", []),
        "results": result_data,
        "maturity_assessment": {
            "engineering_maturity": "controlled-preview",
            "evidence_maturity": "controlled-preview",
        },
        "software": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "ortools": _package_version("ortools"),
            "splink": _package_version("splink"),
            "pgmpy": _package_version("pgmpy"),
            "problog": _package_version("problog"),
            "pm4py": _package_version("pm4py"),
        },
        "execution": {
            "elapsed_seconds": round(elapsed, 6),
            "network_used": False,
            "external_data_fetches": 0,
            "model_calls": 0,
            "reproducible": True,
            "automatic_parallel_execution": False,
        },
    }
    transfer["result_sha256"] = _canonical_sha(
        {
            "schema_version": transfer["schema_version"],
            "task_id": transfer["task_id"],
            "operation": transfer["operation"],
            "input_sha256": transfer["input_sha256"],
            "assumptions": transfer["assumptions"],
            "limitations": transfer["limitations"],
            "results": transfer["results"],
            "maturity_assessment": transfer["maturity_assessment"],
            "software": transfer["software"],
            "execution": transfer["execution"],
        }
    )
    return transfer
