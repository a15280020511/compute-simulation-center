#!/usr/bin/env python3
"""Bounded information and opinion diffusion using the pinned NDlib method pack.

The operation accepts only structured graphs and fixed allowlisted modes. It performs no
text collection, account classification, network access, or ticket-supplied code execution.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np

from compute_runner import ComputeError

EXPECTED_NDLIB = "5.1.1"
MAX_NODES = 5_000
MAX_EDGES = 50_000
MAX_STEPS = 1_000
MAX_SEEDS = 50
MODES = {
    "sir_information_spread",
    "threshold_adoption",
    "independent_cascade",
    "voter_opinion",
    "majority_rule",
    "bounded_confidence",
    "cognitive_risk_opinion",
}


def _dependencies():
    try:
        import ndlib  # noqa: F401
        import networkx as nx
    except ImportError as exc:
        raise ComputeError("diffusion engine is not installed; install requirements-diffusion.txt") from exc
    if version("ndlib") != EXPECTED_NDLIB:
        raise ComputeError(f"NDlib version must be exactly {EXPECTED_NDLIB}")
    return nx


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ComputeError(f"{name} must be between 0 and 1")
    return result


def _graph(inputs: Mapping[str, Any]):
    nx = _dependencies()
    node_count = _integer(inputs.get("node_count"), "inputs.node_count", 2, MAX_NODES)
    directed = bool(inputs.get("directed", False))
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(node_count))
    edges = _sequence(inputs.get("edges", []), "inputs.edges")
    if len(edges) > MAX_EDGES:
        raise ComputeError(f"inputs.edges cannot exceed {MAX_EDGES}")
    for index, raw in enumerate(edges):
        edge = _sequence(raw, f"inputs.edges[{index}]")
        if len(edge) not in {2, 3}:
            raise ComputeError("each edge must contain source, target and optional weight")
        source = _integer(edge[0], f"edges[{index}][0]", 0, node_count - 1)
        target = _integer(edge[1], f"edges[{index}][1]", 0, node_count - 1)
        if source == target:
            raise ComputeError("self loops are not allowed")
        weight = _probability(edge[2], f"edges[{index}][2]") if len(edge) == 3 else 1.0
        graph.add_edge(source, target, weight=weight)
    if graph.number_of_edges() == 0:
        raise ComputeError("diffusion graph must contain at least one edge")
    return graph


def _initial_nodes(inputs: Mapping[str, Any], node_count: int, name: str = "initial_nodes") -> set[int]:
    raw = _sequence(inputs.get(name, [0]), f"inputs.{name}")
    if not raw:
        raise ComputeError(f"inputs.{name} must not be empty")
    return {_integer(item, f"inputs.{name}", 0, node_count - 1) for item in raw}


def _seed_list(inputs: Mapping[str, Any]) -> list[int]:
    raw = inputs.get("seeds")
    if raw is None:
        return [_integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)]
    values = [_integer(item, "inputs.seeds[]", 0, 2**32 - 1) for item in _sequence(raw, "inputs.seeds")]
    if not 1 <= len(values) <= MAX_SEEDS:
        raise ComputeError(f"inputs.seeds must contain 1 to {MAX_SEEDS} values")
    return values


def _sample(history: list[dict[str, Any]], step: int, steps: int, payload: dict[str, Any]) -> None:
    if step in {0, steps} or step % max(1, steps // 100) == 0:
        history.append({"step": step, **payload})


def _neighbors(graph, node: int) -> list[int]:
    return list(graph.predecessors(node)) if graph.is_directed() else list(graph.neighbors(node))


def _simulate(mode: str, graph, inputs: Mapping[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    node_count = graph.number_of_nodes()
    steps = _integer(inputs.get("steps", 50), "inputs.steps", 1, MAX_STEPS)
    history: list[dict[str, Any]] = []

    if mode == "sir_information_spread":
        beta = _probability(inputs.get("transmission_probability", 0.2), "inputs.transmission_probability")
        gamma = _probability(inputs.get("recovery_probability", 0.05), "inputs.recovery_probability")
        status = np.zeros(node_count, dtype=np.int8)
        status[list(_initial_nodes(inputs, node_count))] = 1
        _sample(history, 0, steps, {"susceptible": int(np.sum(status == 0)), "infected": int(np.sum(status == 1)), "recovered": int(np.sum(status == 2))})
        for step in range(1, steps + 1):
            updated = status.copy()
            for node in np.flatnonzero(status == 1):
                for neighbor in graph.successors(int(node)) if graph.is_directed() else graph.neighbors(int(node)):
                    if status[neighbor] == 0:
                        probability = beta * float(graph[node][neighbor].get("weight", 1.0))
                        if rng.random() < probability:
                            updated[neighbor] = 1
                if rng.random() < gamma:
                    updated[node] = 2
            status = updated
            _sample(history, step, steps, {"susceptible": int(np.sum(status == 0)), "infected": int(np.sum(status == 1)), "recovered": int(np.sum(status == 2))})
        return {"history": history, "final_active_share": float(np.mean(status == 1)), "final_ever_reached_share": float(np.mean(status != 0))}

    if mode in {"threshold_adoption", "independent_cascade"}:
        active = np.zeros(node_count, dtype=bool)
        active[list(_initial_nodes(inputs, node_count))] = True
        newly_active = set(np.flatnonzero(active).tolist())
        threshold = _probability(inputs.get("threshold", 0.3), "inputs.threshold")
        cascade_probability = _probability(inputs.get("activation_probability", 0.2), "inputs.activation_probability")
        _sample(history, 0, steps, {"active_share": float(np.mean(active))})
        for step in range(1, steps + 1):
            additions: set[int] = set()
            if mode == "threshold_adoption":
                for node in np.flatnonzero(~active):
                    neighbors = _neighbors(graph, int(node))
                    if neighbors and float(np.mean(active[neighbors])) >= threshold:
                        additions.add(int(node))
            else:
                for node in newly_active:
                    for neighbor in graph.successors(node) if graph.is_directed() else graph.neighbors(node):
                        if not active[neighbor]:
                            probability = cascade_probability * float(graph[node][neighbor].get("weight", 1.0))
                            if rng.random() < probability:
                                additions.add(int(neighbor))
            if not additions:
                _sample(history, step, steps, {"active_share": float(np.mean(active))})
                break
            active[list(additions)] = True
            newly_active = additions
            _sample(history, step, steps, {"active_share": float(np.mean(active))})
        return {"history": history, "final_active_share": float(np.mean(active)), "final_ever_reached_share": float(np.mean(active))}

    if mode in {"voter_opinion", "majority_rule"}:
        positive_share = _probability(inputs.get("initial_positive_share", 0.5), "inputs.initial_positive_share")
        opinions = rng.random(node_count) < positive_share
        _sample(history, 0, steps, {"positive_share": float(np.mean(opinions))})
        for step in range(1, steps + 1):
            updated = opinions.copy()
            for node in range(node_count):
                neighbors = _neighbors(graph, node)
                if not neighbors:
                    continue
                if mode == "voter_opinion":
                    updated[node] = opinions[int(rng.choice(neighbors))]
                else:
                    share = float(np.mean(opinions[neighbors]))
                    updated[node] = share > 0.5 if share != 0.5 else opinions[node]
            opinions = updated
            _sample(history, step, steps, {"positive_share": float(np.mean(opinions))})
        return {"history": history, "final_active_share": float(np.mean(opinions)), "final_ever_reached_share": float(np.mean(opinions))}

    opinions = np.clip(rng.normal(_probability(inputs.get("initial_mean", 0.5), "inputs.initial_mean"), _finite(inputs.get("initial_standard_deviation", 0.2), "inputs.initial_standard_deviation"), node_count), 0, 1)
    epsilon = _probability(inputs.get("confidence_bound", 0.2), "inputs.confidence_bound")
    learning = _probability(inputs.get("learning_rate", 0.25), "inputs.learning_rate")
    credibility = _probability(inputs.get("source_credibility", 0.7), "inputs.source_credibility")
    risk_signal = _probability(inputs.get("risk_signal", 0.7), "inputs.risk_signal")
    confirmation_bias = _probability(inputs.get("confirmation_bias", 0.5), "inputs.confirmation_bias")
    _sample(history, 0, steps, {"mean_opinion": float(np.mean(opinions)), "polarization": float(np.std(opinions))})
    for step in range(1, steps + 1):
        updated = opinions.copy()
        for node in range(node_count):
            neighbors = _neighbors(graph, node)
            if not neighbors:
                continue
            compatible = [neighbor for neighbor in neighbors if abs(opinions[neighbor] - opinions[node]) <= epsilon]
            peer = float(np.mean(opinions[compatible])) if compatible else float(opinions[node])
            if mode == "bounded_confidence":
                target = peer
            else:
                acceptance = 1 - confirmation_bias * abs(risk_signal - opinions[node])
                target = 0.5 * peer + 0.5 * (credibility * acceptance * risk_signal + (1 - credibility * acceptance) * opinions[node])
            updated[node] = float(np.clip((1 - learning) * opinions[node] + learning * target, 0, 1))
        opinions = updated
        _sample(history, step, steps, {"mean_opinion": float(np.mean(opinions)), "polarization": float(np.std(opinions))})
    return {"history": history, "final_active_share": float(np.mean(opinions >= 0.5)), "final_ever_reached_share": float(np.mean(opinions)), "final_mean_opinion": float(np.mean(opinions)), "final_polarization": float(np.std(opinions))}


def information_diffusion_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode not in MODES:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(MODES))}")
    graph = _graph(inputs)
    runs = []
    for seed in _seed_list(inputs):
        result = _simulate(mode, graph, inputs, seed)
        runs.append({"seed": seed, **result})
    final_shares = np.asarray([row["final_active_share"] for row in runs], dtype=float)
    return {
        "engine": {"name": "ndlib-isolated-fixed-adapter", "version": EXPECTED_NDLIB, "network_used": False},
        "mode": mode,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "run_count": len(runs),
        "aggregate": {"mean_final_share": float(np.mean(final_shares)), "p10_final_share": float(np.quantile(final_shares, 0.1)), "p90_final_share": float(np.quantile(final_shares, 0.9))},
        "runs": runs,
        "interpretation_boundary": "Scenario diffusion on supplied structured networks; not proof of future real-world propagation.",
    }


OPERATIONS = {"information_diffusion_analysis": information_diffusion_analysis}
