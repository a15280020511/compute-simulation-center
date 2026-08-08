#!/usr/bin/env python3
"""Deterministic adapters and independent NetworkX checks for transport routing."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx

from pipeline_adapters import ADAPTERS, PipelineAdapterError

PRIMARY_STAGE = "aequilibrae_shortest_path"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineAdapterError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineAdapterError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineAdapterError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PipelineAdapterError(f"{name} must be an integer >= {minimum}")
    return value


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("transport_routing_context")
    return {} if raw is None else _mapping(raw, "inputs.transport_routing_context")


def _links(initial_inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_links = _sequence(initial_inputs.get("links"), "inputs.links")
    if not 1 <= len(raw_links) <= 5000:
        raise PipelineAdapterError("inputs.links must contain 1 to 5000 directed links")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_links, start=1):
        link = _mapping(raw, f"inputs.links[{index - 1}]")
        a_node = _integer(link.get("a_node"), f"inputs.links[{index - 1}].a_node", 1)
        b_node = _integer(link.get("b_node"), f"inputs.links[{index - 1}].b_node", 1)
        cost = _finite(link.get("cost"), f"inputs.links[{index - 1}].cost")
        if a_node == b_node or cost <= 0:
            raise PipelineAdapterError("transport links must be positive-cost directed non-self loops")
        rows.append({"link_id": index, "a_node": a_node, "b_node": b_node, "cost": cost})
    return rows


def _primary(stage_results: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    return _mapping(stage_results.get(PRIMARY_STAGE), f"stage_results.{PRIMARY_STAGE}")


def _independent_route_metrics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float]:
    links = _links(initial_inputs)
    primary = _primary(stage_results)
    origin = _integer(initial_inputs.get("origin"), "inputs.origin", 1)
    destination = _integer(initial_inputs.get("destination"), "inputs.destination", 1)
    path_nodes = [_integer(value, "path_nodes[]", 1) for value in _sequence(primary.get("path_nodes"), "primary.path_nodes")]
    path_links = [_integer(value, "path_links[]", 1) for value in _sequence(primary.get("path_links"), "primary.path_links")]
    if len(path_nodes) < 2 or path_nodes[0] != origin or path_nodes[-1] != destination:
        raise PipelineAdapterError("AequilibraE path endpoints do not match the ticket")
    if len(path_links) != len(path_nodes) - 1:
        raise PipelineAdapterError("AequilibraE path link/node cardinality is inconsistent")

    reconstructed_cost = 0.0
    for position, link_id in enumerate(path_links):
        if link_id > len(links):
            raise PipelineAdapterError("AequilibraE returned an unknown link id")
        link = links[link_id - 1]
        if link["a_node"] != path_nodes[position] or link["b_node"] != path_nodes[position + 1]:
            raise PipelineAdapterError("AequilibraE path link does not match consecutive path nodes")
        reconstructed_cost += float(link["cost"])

    graph = nx.DiGraph()
    for link in links:
        left = int(link["a_node"])
        right = int(link["b_node"])
        cost = float(link["cost"])
        existing = graph.get_edge_data(left, right)
        if existing is None or cost < float(existing["weight"]):
            graph.add_edge(left, right, weight=cost)
    try:
        networkx_cost = float(nx.shortest_path_length(graph, source=origin, target=destination, weight="weight", method="dijkstra"))
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise PipelineAdapterError("NetworkX found no path for a route accepted by AequilibraE") from exc
    return networkx_cost, reconstructed_cost


def transport_ticket_to_aequilibrae(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "aequilibrae_shortest_path":
        raise PipelineAdapterError("transport-routing family requires aequilibrae_shortest_path entry mode")
    _links(initial_inputs)
    return {key: value for key, value in initial_inputs.items() if key != "transport_routing_context"}


def transport_to_networkx_exact_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _primary(stage_results)
    networkx_cost, reconstructed_cost = _independent_route_metrics(initial_inputs, stage_results)
    observed_cost = _finite(primary.get("total_cost"), "primary.total_cost")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("cost_consistency_tolerance", 1e-9), "transport_routing_context.cost_consistency_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("cost_consistency_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "aequilibrae-versus-networkx-shortest-cost",
                "observed": observed_cost,
                "benchmark": networkx_cost,
                "tolerance": tolerance,
                "direction": "absolute",
            },
            {
                "name": "reported-versus-reconstructed-path-cost",
                "observed": observed_cost,
                "benchmark": reconstructed_cost,
                "tolerance": tolerance,
                "direction": "absolute",
            },
        ],
    }


def transport_to_cost_target_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _primary(stage_results)
    context = _context(initial_inputs)
    if "maximum_total_cost" not in context:
        raise PipelineAdapterError("route_cost_target_audit requires maximum_total_cost")
    target = _finite(context["maximum_total_cost"], "transport_routing_context.maximum_total_cost")
    tolerance = _finite(context.get("cost_target_tolerance", 0.0), "transport_routing_context.cost_target_tolerance")
    if target < 0 or tolerance < 0:
        raise PipelineAdapterError("route cost target and tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "route-maximum-total-cost",
            "observed": _finite(primary.get("total_cost"), "primary.total_cost"),
            "benchmark": target,
            "tolerance": tolerance,
            "direction": "maximum",
        }],
    }


def transport_to_hop_target_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _primary(stage_results)
    context = _context(initial_inputs)
    if "maximum_link_count" not in context:
        raise PipelineAdapterError("route_hop_target_audit requires maximum_link_count")
    target = _integer(context["maximum_link_count"], "transport_routing_context.maximum_link_count", 1)
    tolerance = _integer(context.get("link_count_tolerance", 0), "transport_routing_context.link_count_tolerance", 0)
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "route-maximum-link-count",
            "observed": _integer(primary.get("link_count"), "primary.link_count", 1),
            "benchmark": target,
            "tolerance": tolerance,
            "direction": "maximum",
        }],
    }


def install_transport_routing_adapters() -> None:
    adapters = {
        "transport_ticket_to_aequilibrae": transport_ticket_to_aequilibrae,
        "transport_to_networkx_exact_audit": transport_to_networkx_exact_audit,
        "transport_to_cost_target_audit": transport_to_cost_target_audit,
        "transport_to_hop_target_audit": transport_to_hop_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"transport-routing adapter name collision: {name}")
        ADAPTERS[name] = handler
