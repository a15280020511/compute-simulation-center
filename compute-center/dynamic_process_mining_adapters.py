#!/usr/bin/env python3
"""Deterministic adapters for the dynamic process-mining family."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pipeline_adapters import ADAPTERS, PipelineAdapterError


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineAdapterError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineAdapterError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineAdapterError(f"{name} must be an integer")
    if value < minimum:
        raise PipelineAdapterError(f"{name} must be at least {minimum}")
    return value


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("process_context")
    return {} if raw is None else _mapping(raw, "inputs.process_context")


def process_ticket_to_pm4py(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "pm4py_directly_follows":
        raise PipelineAdapterError("process-mining family requires pm4py_directly_follows entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "process_context"}


def dfg_to_graph_summary(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del initial_inputs, stage
    primary = _mapping(
        stage_results.get("directly_follows_discovery"),
        "stage_results.directly_follows_discovery",
    )
    raw_edges = _sequence(primary.get("directly_follows_edges"), "directly_follows_edges")
    start_activities = _mapping(primary.get("start_activities"), "start_activities")
    end_activities = _mapping(primary.get("end_activities"), "end_activities")

    nodes = {str(name) for name in start_activities} | {str(name) for name in end_activities}
    edges: list[list[str]] = []
    for index, raw_edge in enumerate(raw_edges):
        edge = _mapping(raw_edge, f"directly_follows_edges[{index}]")
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not source or not target:
            raise PipelineAdapterError("directly-follows edges require non-empty source and target")
        count = _integer(edge.get("count"), f"directly_follows_edges[{index}].count", 1)
        del count
        nodes.update({source, target})
        edges.append([source, target])

    if not nodes:
        raise PipelineAdapterError("process discovery must emit at least one activity")
    return {
        "mode": "large_graph_summary",
        "nodes": sorted(nodes),
        "edges": sorted(edges, key=lambda pair: (pair[0], pair[1])),
        "directed": True,
        "pagerank_iterations": 20,
        "top_k": min(len(nodes), 100),
    }


def process_graph_to_consistency_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del initial_inputs, stage
    primary = _mapping(
        stage_results.get("directly_follows_discovery"),
        "stage_results.directly_follows_discovery",
    )
    graph = _mapping(stage_results.get("workflow_graph_summary"), "stage_results.workflow_graph_summary")
    dfg_edges = _sequence(primary.get("directly_follows_edges"), "directly_follows_edges")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "process-activity-node-count-consistency",
                "observed": _integer(graph.get("node_count"), "workflow_graph_summary.node_count", 0),
                "benchmark": _integer(primary.get("activity_count"), "directly_follows_discovery.activity_count", 0),
                "tolerance": 0.0,
                "direction": "absolute",
            },
            {
                "name": "process-dfg-graph-edge-count-consistency",
                "observed": _integer(graph.get("edge_count"), "workflow_graph_summary.edge_count", 0),
                "benchmark": len(dfg_edges),
                "tolerance": 0.0,
                "direction": "absolute",
            },
        ],
    }


def process_to_target_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(
        stage_results.get("directly_follows_discovery"),
        "stage_results.directly_follows_discovery",
    )
    context = _context(initial_inputs)
    dfg_edges = _sequence(primary.get("directly_follows_edges"), "directly_follows_edges")
    observed_values = {
        "case_count": _integer(primary.get("case_count"), "directly_follows_discovery.case_count", 0),
        "event_count": _integer(primary.get("event_count"), "directly_follows_discovery.event_count", 0),
        "activity_count": _integer(primary.get("activity_count"), "directly_follows_discovery.activity_count", 0),
        "dfg_edge_count": len(dfg_edges),
    }
    specs = (
        ("case_count", "expected_case_count", "case_count_tolerance"),
        ("event_count", "expected_event_count", "event_count_tolerance"),
        ("activity_count", "expected_activity_count", "activity_count_tolerance"),
        ("dfg_edge_count", "expected_dfg_edge_count", "dfg_edge_count_tolerance"),
    )
    candidates = []
    for observed_name, target_name, tolerance_name in specs:
        if target_name not in context:
            continue
        target = _integer(context.get(target_name), f"process_context.{target_name}", 0)
        tolerance = _integer(context.get(tolerance_name, 0), f"process_context.{tolerance_name}", 0)
        candidates.append(
            {
                "name": f"process-target-{observed_name}",
                "observed": observed_values[observed_name],
                "benchmark": target,
                "tolerance": float(tolerance),
                "direction": "absolute",
            }
        )
    if not candidates:
        raise PipelineAdapterError("process_target_audit requires at least one explicit process target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_process_mining_adapters() -> None:
    adapters = {
        "process_ticket_to_pm4py": process_ticket_to_pm4py,
        "dfg_to_graph_summary": dfg_to_graph_summary,
        "process_graph_to_consistency_audit": process_graph_to_consistency_audit,
        "process_to_target_audit": process_to_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"process-mining adapter name collision: {name}")
        ADAPTERS[name] = handler
