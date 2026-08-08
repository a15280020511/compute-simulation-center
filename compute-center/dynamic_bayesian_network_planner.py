#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for the Bayesian-network capability family."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from jsonschema import Draft202012Validator
from ortools.sat.python import cp_model

from dynamic_bayesian_adapters import install_bayesian_adapters
from dynamic_family_router import resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_bayesian_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-bayesian-policy.json"
GRAPH_PATH = HERE / "dynamic-bayesian-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-bayesian-stage-contracts.json"
FAMILY = "bayesian-network"
DECLARED_OPERATION = "bayesian_network_inference"
REQUIRED_STAGE_IDS = ("parameter_estimation", "posterior_inference")
RESULT_STAGE_ID = "posterior_inference"
EXPECTED_PGMPY = "1.1.2"


class DynamicBayesianNetworkError(ValueError):
    """Raised when a Bayesian-network dynamic plan cannot be generated or executed safely."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicBayesianNetworkError(f"JSON root must be an object: {path.name}")
    return value


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _load_contracts() -> dict[str, Any]:
    value = _load_json(CONTRACT_PATH)
    if value.get("schema_version") != "compute-dynamic-bayesian-stage-contracts-v1":
        raise DynamicBayesianNetworkError("invalid Bayesian stage contract schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicBayesianNetworkError("Bayesian stage contract identity mismatch")
    contracts = value.get("contracts")
    required_modes = {
        "bayesian_parameter_estimation",
        "fixed_network_inference",
        "evidence_sensitivity",
        "virtual_evidence_update",
    }
    if not isinstance(contracts, Mapping) or set(contracts) != required_modes:
        raise DynamicBayesianNetworkError("Bayesian stage contracts must exactly cover admitted dynamic modes")
    for mode, schema in contracts.items():
        if not isinstance(schema, Mapping):
            raise DynamicBayesianNetworkError(f"invalid Bayesian stage contract: {mode}")
        Draft202012Validator.check_schema(dict(schema))
    return {str(key): dict(schema) for key, schema in contracts.items()}


def _validate_stage_output(result: Mapping[str, Any], contracts: Mapping[str, Any]) -> None:
    mode = str(result.get("mode") or "")
    schema = contracts.get(mode)
    if not isinstance(schema, Mapping):
        raise DynamicBayesianNetworkError(f"no Bayesian dynamic output contract for mode: {mode or '<empty>'}")
    validator = Draft202012Validator(dict(schema))
    errors = sorted(validator.iter_errors(dict(result)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicBayesianNetworkError(
            f"Bayesian stage output contract failed for {mode} at {path}: {error.message}"
        )


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    if policy.get("schema_version") != "compute-dynamic-bayesian-policy-v1":
        raise DynamicBayesianNetworkError("invalid dynamic Bayesian policy schema")
    expected = {
        "status": "controlled-preview",
        "family": FAMILY,
        "declared_operation": DECLARED_OPERATION,
        "planner": "ortools-cp-sat",
        "graph_engine": "networkx",
        "network_policy": "deny",
        "model_calls": 0,
        "objective_text_routing_allowed": False,
        "structured_signals_only": True,
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_code_allowed": False,
        "automatic_parallel_execution": False,
        "cycles_allowed": False,
        "branching_allowed": True,
        "maximum_stages": 4,
    }
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise DynamicBayesianNetworkError(f"unsafe dynamic Bayesian policy: {key}")
    allowed_operations = policy.get("allowed_operations")
    allowed_adapters = policy.get("allowed_adapters")
    if not isinstance(allowed_operations, list) or allowed_operations != [DECLARED_OPERATION]:
        raise DynamicBayesianNetworkError("Bayesian allowed_operations must contain only the declared operation")
    if not isinstance(allowed_adapters, list) or len(allowed_adapters) != len(set(allowed_adapters)):
        raise DynamicBayesianNetworkError("allowed_adapters must be a unique array")
    solver = policy.get("solver_policy")
    if not isinstance(solver, Mapping):
        raise DynamicBayesianNetworkError("solver_policy is required")
    if solver.get("require_optimal_status") is not True:
        raise DynamicBayesianNetworkError("Bayesian dynamic planner must require OPTIMAL")
    if int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicBayesianNetworkError("Bayesian dynamic planner must use one CP-SAT worker")
    max_time = solver.get("max_time_seconds")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not 0 < float(max_time) <= 10:
        raise DynamicBayesianNetworkError("solver max_time_seconds must be in (0,10]")
    max_optional = int(solver.get("exhaustive_cross_check_max_optional_nodes") or 0)
    if not 1 <= max_optional <= 16:
        raise DynamicBayesianNetworkError("invalid exhaustive cross-check bound")
    selection = policy.get("selection_policy")
    rules = selection.get("stage_rules") if isinstance(selection, Mapping) else None
    if not isinstance(rules, Mapping) or not rules:
        raise DynamicBayesianNetworkError("selection_policy.stage_rules must be a non-empty object")
    for node_id, raw_rule in rules.items():
        if not isinstance(raw_rule, Mapping):
            raise DynamicBayesianNetworkError(f"invalid Bayesian stage rule: {node_id}")
        if str(raw_rule.get("operation") or "") != DECLARED_OPERATION:
            raise DynamicBayesianNetworkError(f"Bayesian stage rule operation mismatch: {node_id}")
        penalty = raw_rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicBayesianNetworkError(f"invalid Bayesian stage penalty: {node_id}")
        benefits = raw_rule.get("benefits")
        if not isinstance(benefits, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in benefits.values()
        ):
            raise DynamicBayesianNetworkError(f"invalid Bayesian stage benefits: {node_id}")
        for name in ("eligible_all", "required_if_any", "required_if_all"):
            value = raw_rule.get(name, [])
            if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
                raise DynamicBayesianNetworkError(f"{name} must be a string array: {node_id}")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(GRAPH_PATH)
    if value.get("schema_version") != "compute-dynamic-bayesian-capability-graph-v1":
        raise DynamicBayesianNetworkError("invalid Bayesian capability graph schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicBayesianNetworkError("Bayesian capability graph identity mismatch")
    if value.get("graph_engine") != "networkx" or value.get("selection_engine") != "ortools-cp-sat":
        raise DynamicBayesianNetworkError("Bayesian capability graph engine mismatch")
    safety = value.get("safety")
    expected_safety = {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "ticket_supplied_code_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "branching_allowed": True,
        "execution_remains_strict_serial": True,
    }
    if not isinstance(safety, Mapping):
        raise DynamicBayesianNetworkError("Bayesian graph safety is required")
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise DynamicBayesianNetworkError(f"unsafe Bayesian graph policy: {key}")
    raw_order = value.get("node_order")
    raw_nodes = value.get("nodes")
    if not isinstance(raw_order, list) or not raw_order or not isinstance(raw_nodes, Mapping):
        raise DynamicBayesianNetworkError("Bayesian capability graph nodes/order are invalid")
    order = [str(item) for item in raw_order]
    if len(order) != len(set(order)) or set(order) != {str(item) for item in raw_nodes}:
        raise DynamicBayesianNetworkError("Bayesian node_order must exactly enumerate unique nodes")
    allowed_adapters = {str(item) for item in policy["allowed_adapters"]}
    nodes: dict[str, dict[str, Any]] = {}
    for node_id in order:
        raw = raw_nodes[node_id]
        if not isinstance(raw, Mapping):
            raise DynamicBayesianNetworkError(f"Bayesian graph node must be an object: {node_id}")
        node = dict(raw)
        operation = str(node.get("operation") or "")
        adapter = str(node.get("adapter") or "")
        if operation != DECLARED_OPERATION:
            raise DynamicBayesianNetworkError(f"Bayesian graph operation mismatch: {node_id}")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicBayesianNetworkError(f"Bayesian adapter not allowlisted: {adapter}")
        nodes[node_id] = node
    for stage_id in REQUIRED_STAGE_IDS:
        node = nodes.get(stage_id)
        if not isinstance(node, Mapping) or node.get("required") is not True:
            raise DynamicBayesianNetworkError(f"required Bayesian stage missing or optional: {stage_id}")
    result_nodes = [node_id for node_id, node in nodes.items() if node.get("result_required") is True]
    if result_nodes != [RESULT_STAGE_ID]:
        raise DynamicBayesianNetworkError("Bayesian graph must define posterior_inference as the only result stage")
    optional_ids = [node_id for node_id in order if node_id not in REQUIRED_STAGE_IDS]
    rules = policy["selection_policy"]["stage_rules"]
    if optional_ids != [str(item) for item in rules]:
        raise DynamicBayesianNetworkError("Bayesian stage rule order must match optional graph nodes")
    for node_id in optional_ids:
        if nodes[node_id].get("required") is True or nodes[node_id].get("result_required") is True:
            raise DynamicBayesianNetworkError(f"optional Bayesian node marked required: {node_id}")
    precedence = value.get("precedence")
    if not isinstance(precedence, list):
        raise DynamicBayesianNetworkError("Bayesian precedence must be an array")
    edges: list[tuple[str, str]] = []
    for raw_edge in precedence:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise DynamicBayesianNetworkError("Bayesian precedence edge must contain two node ids")
        left, right = str(raw_edge[0]), str(raw_edge[1])
        if left not in nodes or right not in nodes or left == right:
            raise DynamicBayesianNetworkError(f"invalid Bayesian edge: {left}->{right}")
        edges.append((left, right))
    expected_edges = {
        ("parameter_estimation", "posterior_inference"),
        ("parameter_estimation", "evidence_sensitivity"),
        ("parameter_estimation", "virtual_evidence_update"),
    }
    if set(edges) != expected_edges:
        raise DynamicBayesianNetworkError("Bayesian capability DAG does not match the controlled dependency structure")
    full_graph = nx.DiGraph()
    full_graph.add_nodes_from(order)
    full_graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full_graph):
        raise DynamicBayesianNetworkError("Bayesian capability graph contains a cycle")
    if set(nx.descendants(full_graph, "parameter_estimation")) != set(order[1:]):
        raise DynamicBayesianNetworkError("all Bayesian downstream stages must depend on parameter_estimation")
    index = {node_id: position for position, node_id in enumerate(order)}
    deterministic_order = list(nx.lexicographical_topological_sort(full_graph, key=lambda node: index[node]))
    if deterministic_order != order:
        raise DynamicBayesianNetworkError("Bayesian node_order disagrees with deterministic NetworkX topology")
    return {
        "nodes": nodes,
        "precedence": edges,
        "full_order": order,
        "optional_ids": optional_ids,
        "index": index,
    }


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicBayesianNetworkError(f"{name} must be an array")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicBayesianNetworkError(f"{name} must be an object")
    return value


def _categorical_scalar(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DynamicBayesianNetworkError(f"{name} must be a categorical scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise DynamicBayesianNetworkError(f"{name} must be finite")


def _signals(ticket: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicBayesianNetworkError("ticket was not routed to bayesian-network family")
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicBayesianNetworkError("ticket inputs must be an object")
    if str(inputs.get("mode") or "") != "bayesian_parameter_estimation":
        raise DynamicBayesianNetworkError("Bayesian dynamic family entry mode must be bayesian_parameter_estimation")

    data = _mapping(inputs.get("data"), "inputs.data")
    if not 1 <= len(data) <= 50:
        raise DynamicBayesianNetworkError("Bayesian dynamic data must contain 1 to 50 variables")
    lengths: set[int] = set()
    for raw_name, raw_values in data.items():
        name = str(raw_name).strip()
        if not name:
            raise DynamicBayesianNetworkError("Bayesian data variable names must be non-empty")
        values = _sequence(raw_values, f"inputs.data.{name}")
        lengths.add(len(values))
        for index, item in enumerate(values):
            _categorical_scalar(item, f"inputs.data.{name}[{index}]")
    if len(lengths) != 1:
        raise DynamicBayesianNetworkError("all Bayesian data columns must have equal length")
    observation_count = next(iter(lengths))
    minimum_rows = int(policy["selection_policy"]["minimum_dynamic_rows"])
    if not minimum_rows <= observation_count <= 50_000:
        raise DynamicBayesianNetworkError(
            f"Bayesian dynamic family requires {minimum_rows} to 50000 observations"
        )

    raw_edges = _sequence(inputs.get("edges", []), "inputs.edges")
    if len(raw_edges) > 200:
        raise DynamicBayesianNetworkError("Bayesian dynamic family supports at most 200 edges")
    edges: list[tuple[str, str]] = []
    nodes = {str(name) for name in data}
    for index, raw_edge in enumerate(raw_edges):
        edge = _sequence(raw_edge, f"inputs.edges[{index}]")
        if len(edge) != 2:
            raise DynamicBayesianNetworkError("each Bayesian edge must contain source and target")
        left, right = str(edge[0]).strip(), str(edge[1]).strip()
        if not left or not right or left == right:
            raise DynamicBayesianNetworkError("Bayesian edges require distinct non-empty nodes")
        edges.append((left, right))
        nodes.update((left, right))
    if len(nodes) > 50:
        raise DynamicBayesianNetworkError("Bayesian dynamic family supports at most 50 nodes")
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicBayesianNetworkError("supplied Bayesian dependency structure must be a DAG")

    query_variables = [str(item) for item in _sequence(inputs.get("query_variables"), "inputs.query_variables")]
    if not query_variables or len(query_variables) != len(set(query_variables)):
        raise DynamicBayesianNetworkError("query_variables must be non-empty and unique")
    if any(item not in nodes for item in query_variables):
        raise DynamicBayesianNetworkError("query_variables must reference known Bayesian nodes")
    evidence_raw = inputs.get("evidence")
    evidence = dict(evidence_raw) if isinstance(evidence_raw, Mapping) else {}
    if any(str(key) not in nodes for key in evidence):
        raise DynamicBayesianNetworkError("evidence contains an unknown Bayesian node")
    if set(query_variables) & {str(key) for key in evidence}:
        raise DynamicBayesianNetworkError("query_variables cannot also be fixed as hard evidence")
    for key, item in evidence.items():
        _categorical_scalar(item, f"inputs.evidence.{key}")

    scenarios_raw = inputs.get("evidence_scenarios")
    scenarios: list[Mapping[str, Any]] = []
    if scenarios_raw is not None:
        raw_rows = _sequence(scenarios_raw, "inputs.evidence_scenarios")
        if not 1 <= len(raw_rows) <= 50:
            raise DynamicBayesianNetworkError("evidence_scenarios must contain 1 to 50 scenarios")
        for index, raw in enumerate(raw_rows):
            row = _mapping(raw, f"inputs.evidence_scenarios[{index}]")
            scenario_evidence = _mapping(row.get("evidence", {}), f"inputs.evidence_scenarios[{index}].evidence")
            if any(str(key) not in nodes for key in scenario_evidence):
                raise DynamicBayesianNetworkError("evidence_scenarios contain an unknown node")
            if set(query_variables) & {str(key) for key in scenario_evidence}:
                raise DynamicBayesianNetworkError("sensitivity evidence cannot fix a queried node")
            for key, item in scenario_evidence.items():
                _categorical_scalar(item, f"inputs.evidence_scenarios[{index}].evidence.{key}")
            scenarios.append(row)

    virtual_raw = inputs.get("virtual_evidence")
    virtual_rows: list[Mapping[str, Any]] = []
    if virtual_raw is not None:
        raw_rows = _sequence(virtual_raw, "inputs.virtual_evidence")
        if not 1 <= len(raw_rows) <= 20:
            raise DynamicBayesianNetworkError("virtual_evidence must contain 1 to 20 entries")
        for index, raw in enumerate(raw_rows):
            row = _mapping(raw, f"inputs.virtual_evidence[{index}]")
            variable = str(row.get("variable") or "")
            if variable not in nodes:
                raise DynamicBayesianNetworkError("virtual evidence targets an unknown Bayesian node")
            probabilities = _sequence(row.get("probabilities"), f"inputs.virtual_evidence[{index}].probabilities")
            if not 2 <= len(probabilities) <= 20:
                raise DynamicBayesianNetworkError("virtual evidence probabilities must contain 2 to 20 states")
            total = 0.0
            for p_index, raw_probability in enumerate(probabilities):
                if isinstance(raw_probability, bool) or not isinstance(raw_probability, (int, float)):
                    raise DynamicBayesianNetworkError(
                        f"inputs.virtual_evidence[{index}].probabilities[{p_index}] must be numeric"
                    )
                probability = float(raw_probability)
                if not math.isfinite(probability) or probability < 0:
                    raise DynamicBayesianNetworkError("virtual evidence probabilities must be finite and non-negative")
                total += probability
            if not math.isclose(total, 1.0, abs_tol=1e-8):
                raise DynamicBayesianNetworkError("virtual evidence probabilities must sum to 1")
            virtual_rows.append(row)

    context = inputs.get("dynamic_context")
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise DynamicBayesianNetworkError("inputs.dynamic_context must be an object")
    sensitivity_requested = context.get("evidence_sensitivity") is True
    virtual_requested = context.get("virtual_evidence_update") is True
    if sensitivity_requested and not scenarios:
        raise DynamicBayesianNetworkError("evidence_sensitivity was requested without evidence_scenarios")
    if virtual_requested and not virtual_rows:
        raise DynamicBayesianNetworkError("virtual_evidence_update was requested without virtual_evidence")
    decision_class = _decision_class(ticket)
    high_stakes_minimum = int(policy["selection_policy"]["high_stakes_minimum_sensitivity_scenarios"])
    if decision_class == "high_stakes" and len(scenarios) < high_stakes_minimum:
        raise DynamicBayesianNetworkError(
            f"high-stakes Bayesian dynamic family requires at least {high_stakes_minimum} evidence scenarios"
        )

    signals = {
        "bayesian_input_valid": True,
        "sensitivity_scenarios_present": bool(scenarios),
        "multiple_sensitivity_scenarios": len(scenarios) >= 2,
        "sensitivity_requested": sensitivity_requested,
        "virtual_evidence_present": bool(virtual_rows),
        "virtual_evidence_requested": virtual_requested,
        "formal_decision": decision_class in {"formal", "high_stakes"},
        "high_stakes": decision_class == "high_stakes",
    }
    features = {
        "entry_mode": "bayesian_parameter_estimation",
        "observation_count": observation_count,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "query_variables": query_variables,
        "hard_evidence_count": len(evidence),
        "sensitivity_scenario_count": len(scenarios),
        "virtual_evidence_count": len(virtual_rows),
        "sensitivity_requested": sensitivity_requested,
        "virtual_evidence_requested": virtual_requested,
        "decision_class": decision_class,
        "causal_structure_claimed": False,
    }
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    any_names = rule.get("required_if_any", [])
    all_names = rule.get("required_if_all", [])
    return any(bool(signals.get(str(name), False)) for name in any_names) or (
        bool(all_names) and all(bool(signals.get(str(name), False)) for name in all_names)
    )


def _utilities(policy: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, int]:
    result: dict[str, int] = {}
    for node_id, rule in policy["selection_policy"]["stage_rules"].items():
        score = -int(rule["penalty"])
        for signal, benefit in rule["benefits"].items():
            score += int(benefit) * int(bool(signals.get(str(signal), False)))
        result[str(node_id)] = score
    return result


def _feasible(candidate: Mapping[str, bool], policy: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    rules = policy["selection_policy"]["stage_rules"]
    if set(candidate) != {str(item) for item in rules}:
        return False
    for node_id, rule in rules.items():
        chosen = bool(candidate[str(node_id)])
        if chosen and not _eligible(rule, signals):
            return False
        if _required(rule, signals) and not chosen:
            return False
    return True


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    optional_ids = list(graph["optional_ids"])
    rules = policy["selection_policy"]["stage_rules"]
    utilities = _utilities(policy, signals)
    model = cp_model.CpModel()
    variables = {node_id: model.new_bool_var(f"select_{node_id}") for node_id in optional_ids}
    for node_id in optional_ids:
        rule = rules[node_id]
        if not _eligible(rule, signals):
            model.add(variables[node_id] == 0)
        if _required(rule, signals):
            model.add(variables[node_id] == 1)
    model.maximize(sum(int(utilities[node_id]) * variables[node_id] for node_id in optional_ids))
    solver_policy = policy["solver_policy"]
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if solver_policy["require_optimal_status"] and status != cp_model.OPTIMAL:
        raise DynamicBayesianNetworkError(f"CP-SAT must prove OPTIMAL; observed status={status_name}")
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise DynamicBayesianNetworkError(f"Bayesian CP-SAT found no feasible selection: {status_name}")
    selected = {node_id: bool(solver.value(variables[node_id])) for node_id in optional_ids}
    objective = int(round(solver.objective_value))
    max_optional = int(solver_policy["exhaustive_cross_check_max_optional_nodes"])
    if len(optional_ids) > max_optional:
        cross: dict[str, Any] = {"performed": False, "reason": "optional-node-count-exceeds-policy"}
    else:
        feasible: list[dict[str, Any]] = []
        for bits in itertools.product((False, True), repeat=len(optional_ids)):
            candidate = dict(zip(optional_ids, bits, strict=True))
            if not _feasible(candidate, policy, signals):
                continue
            score = sum(int(utilities[node_id]) * int(candidate[node_id]) for node_id in optional_ids)
            feasible.append({"selection": candidate, "objective": score})
        if not feasible:
            raise DynamicBayesianNetworkError("no feasible Bayesian selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise DynamicBayesianNetworkError(
                f"Bayesian CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}"
            )
        cross = {
            "performed": True,
            "optional_node_count": len(optional_ids),
            "feasible_selection_count": len(feasible),
            "best_objective": best,
            "optimal_selections": optimal,
            "unique_optimum": len(optimal) == 1,
            "passed": True,
        }
    return {
        "selected_nodes": selected,
        "solver_status": status_name,
        "objective_value": objective,
        "global_optimal_proven": status == cp_model.OPTIMAL and bool(cross.get("passed", True)),
        "utility_by_node": utilities,
        "signals": dict(signals),
        "solver_policy": {
            "num_search_workers": int(solver_policy["num_search_workers"]),
            "random_seed": int(solver_policy["random_seed"]),
            "max_time_seconds": float(solver_policy["max_time_seconds"]),
            "require_optimal_status": True,
        },
        "exhaustive_cross_check": cross,
    }


def plan_dynamic_bayesian_network(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    _load_contracts()
    signals, features = _signals(ticket, policy)
    optimization = _solve(policy, graph, signals)
    selected_optional = optimization["selected_nodes"]
    selected_nodes = set(REQUIRED_STAGE_IDS) | {
        node_id for node_id, chosen in selected_optional.items() if chosen
    }
    if len(selected_nodes) > int(policy["maximum_stages"]):
        raise DynamicBayesianNetworkError("Bayesian plan exceeds maximum stages")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(node_id for node_id in graph["full_order"] if node_id in selected_nodes)
    runtime_graph.add_edges_from(
        (left, right)
        for left, right in graph["precedence"]
        if left in selected_nodes and right in selected_nodes
    )
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicBayesianNetworkError("Bayesian selected plan contains a cycle")
    if not nx.has_path(runtime_graph, "parameter_estimation", "posterior_inference"):
        raise DynamicBayesianNetworkError("posterior inference lost its parameter-estimation dependency")
    execution_order = list(
        nx.lexicographical_topological_sort(runtime_graph, key=lambda node: graph["index"][node])
    )
    expected_order = [node_id for node_id in graph["full_order"] if node_id in selected_nodes]
    if execution_order != expected_order:
        raise DynamicBayesianNetworkError("NetworkX deterministic execution order disagrees with selected Bayesian order")
    stage_map: dict[str, dict[str, Any]] = {}
    for stage_id in execution_order:
        node = graph["nodes"][stage_id]
        predecessors = sorted(runtime_graph.predecessors(stage_id), key=lambda item: graph["index"][item])
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "adapter": str(node["adapter"]),
            "depends_on": predecessors,
        }
    reasons = [
        "bayesian-network family was selected from explicit operation, parameter-estimation mode, and structured DAG/data/query inputs",
        "parameter_estimation and posterior_inference are mandatory because the posterior consumes the estimated CPDs",
        (
            "OR-Tools CP-SAT proved the policy-optimal feasible optional robustness subset; "
            f"status={optimization['solver_status']}, objective={optimization['objective_value']}"
        ),
        "NetworkX preserves the branching dependency DAG while execution remains strictly serial in deterministic topological order",
    ]
    if optimization["exhaustive_cross_check"].get("performed"):
        reasons.append("independent exhaustive enumeration matched the Bayesian CP-SAT optimum")
    for node_id in graph["optional_ids"]:
        if selected_optional.get(node_id):
            reasons.append(
                f"{node_id} selected with utility={optimization['utility_by_node'][node_id]}"
            )
    return {
        "id": "dynamic-auto-v1",
        "family": FAMILY,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": DECLARED_OPERATION,
        "result_stage": RESULT_STAGE_ID,
        "required_stages": list(REQUIRED_STAGE_IDS),
        "stage_order": execution_order,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": reasons,
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute(
    ticket: Mapping[str, Any],
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_bayesian_network(ticket)
    contracts = _load_contracts()
    initial_inputs = ticket.get("inputs")
    if not isinstance(initial_inputs, Mapping):
        raise DynamicBayesianNetworkError("ticket inputs must be an object")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    stage_elapsed: dict[str, float] = {}
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": plan["id"],
        "family": FAMILY,
        "status": "RUNNING",
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "network_used": False,
        "model_calls": 0,
        "plan_sha256": _canonical_sha({
            "family": FAMILY,
            "stage_order": plan["stage_order"],
            "stage_map": plan["stage_map"],
            "planning_features": plan["planning_features"],
            "optimization": plan["optimization"],
        }),
        "stages": [
            {
                "stage_id": stage_id,
                "operation": plan["stage_map"][stage_id]["operation"],
                "depends_on": plan["stage_map"][stage_id]["depends_on"],
                "status": "PENDING",
            }
            for stage_id in plan["stage_order"]
        ],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = str(stage["operation"])
            adapter_name = str(stage["adapter"])
            if operation not in operations:
                raise DynamicBayesianNetworkError(f"handler unavailable: {operation}")
            if adapter_name not in ADAPTERS:
                raise DynamicBayesianNetworkError(f"adapter unavailable: {adapter_name}")
            for dependency in stage["depends_on"]:
                if dependency not in stage_results:
                    raise DynamicBayesianNetworkError(
                        f"Bayesian stage {stage_id} dependency has not completed: {dependency}"
                    )
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicBayesianNetworkError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(
                output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json",
                stage_inputs,
            )
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            stage_elapsed[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise DynamicBayesianNetworkError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            _validate_stage_output(result_dict, contracts)
            output_sha = _canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            _write_json(
                output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json",
                result_dict,
            )
            receipt = {
                "stage_id": stage_id,
                "operation": operation,
                "adapter": adapter_name,
                "depends_on": list(stage["depends_on"]),
                "status": "PASS",
                "input_sha256": input_sha,
                "output_sha256": output_sha,
            }
            receipts.append(receipt)
            state["stages"][index].update(receipt)
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"
        for row in state["stages"]:
            if row["status"] == "RUNNING":
                row["status"] = "FAILED"
        _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise
    state["status"] = "PASS"
    state["pipeline_sha256"] = _canonical_sha(receipts)
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return plan, stage_results, receipts, stage_elapsed


def run_dynamic_bayesian_network_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicBayesianNetworkError("ticket is not an admitted bayesian-network dynamic request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, stage_elapsed = _execute(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started
    import numpy as np
    import ortools
    import scipy

    result_data: dict[str, Any] = {
        "pipeline_id": plan["id"],
        "dynamic_family": FAMILY,
        "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "stage_order": plan["stage_order"],
        "stage_dependencies": {
            stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]
        },
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "optimization": plan["optimization"],
        "stage_receipts": receipts,
        "stage_outputs": stage_results,
        "final_stage": plan["result_stage"],
        "final_result": stage_results[plan["result_stage"]],
        "causal_structure_claimed": False,
    }
    robustness_results = {
        stage_id: stage_results[stage_id]
        for stage_id in ("evidence_sensitivity", "virtual_evidence_update")
        if stage_id in stage_results
    }
    if robustness_results:
        result_data["robustness_results"] = robustness_results
    transfer: dict[str, Any] = {
        "schema_version": "compute-result-v1",
        "task_id": str(ticket["task_id"]),
        "status": "success",
        "operation": str(ticket["operation"]),
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
            "ortools": ortools.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pgmpy": version("pgmpy"),
        },
        "execution": {
            "elapsed_seconds": round(elapsed, 6),
            "stage_elapsed_seconds": stage_elapsed,
            "network_used": False,
            "model_calls": 0,
            "reproducible": True,
            "automatic_parallel_execution": False,
            "graph_contains_branching": len(plan["stage_order"]) > 2,
        },
    }
    transfer["result_sha256"] = _canonical_sha({
        "schema_version": transfer["schema_version"],
        "task_id": transfer["task_id"],
        "operation": transfer["operation"],
        "input_sha256": transfer["input_sha256"],
        "assumptions": transfer["assumptions"],
        "limitations": transfer["limitations"],
        "results": transfer["results"],
        "maturity_assessment": transfer["maturity_assessment"],
        "software": transfer["software"],
    })
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(
        output_dir / "compute-audit.json",
        {
            "version": 1,
            "status": "PASS",
            "task_id": transfer["task_id"],
            "operation": transfer["operation"],
            "pipeline_id": plan["id"],
            "dynamic_family": FAMILY,
            "planning_mode": plan["planning_mode"],
            "selection_engine": plan["selection_engine"],
            "graph_engine": plan["graph_engine"],
            "solver_status": plan["optimization"]["solver_status"],
            "global_optimal_proven": plan["optimization"]["global_optimal_proven"],
            "input_sha256": transfer["input_sha256"],
            "result_sha256": transfer["result_sha256"],
            "elapsed_seconds": transfer["execution"]["elapsed_seconds"],
            "model_calls": 0,
            "network_used": False,
            "automatic_parallel_execution": False,
            "graph_contains_branching": transfer["execution"]["graph_contains_branching"],
            "causal_structure_claimed": False,
            "secret_values_included": False,
        },
    )
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic family: `{FAMILY}`\n"
        f"- Dynamic pipeline: `{plan['id']}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selection engine: `{plan['selection_engine']}`\n"
        f"- Graph engine: `{plan['graph_engine']}`\n"
        f"- Solver status: `{plan['optimization']['solver_status']}`\n"
        f"- Global optimum proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        "- Execution policy: `strict-serial-topological`\n"
        "- Automatic parallel execution: `false`\n"
        "- Causal structure claimed: `false`\n"
        "- Model calls: `0`\n"
        "- Network used: `false`\n",
        encoding="utf-8",
    )
    return transfer
