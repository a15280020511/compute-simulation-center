#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for bounded system-dynamics simulations."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from jsonschema import Draft202012Validator
from ortools.sat.python import cp_model

from dynamic_family_router import resolve_dynamic_family
from dynamic_system_dynamics_adapters import ROBUSTNESS_PARAMETERS, install_system_dynamics_adapters
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_system_dynamics_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-system-dynamics-policy.json"
GRAPH_PATH = HERE / "dynamic-system-dynamics-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-system-dynamics-stage-contracts.json"
FAMILY = "system-dynamics"
DECLARED_OPERATION = "system_dynamics_simulation"
REQUIRED_STAGE_IDS = ("primary_simulation",)
RESULT_STAGE_ID = "primary_simulation"


class DynamicSystemDynamicsError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicSystemDynamicsError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicSystemDynamicsError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicSystemDynamicsError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicSystemDynamicsError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicSystemDynamicsError(f"{name} must be finite")
    return result


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _load_contracts() -> dict[str, Any]:
    value = _load_json(CONTRACT_PATH)
    if value.get("schema_version") != "compute-dynamic-system-dynamics-stage-contracts-v1":
        raise DynamicSystemDynamicsError("invalid system-dynamics stage contract schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicSystemDynamicsError("system-dynamics stage contract identity mismatch")
    contracts = value.get("contracts")
    expected = {"primary_simulation", "trajectory_statistics", "robustness_simulation", "robustness_audit", "external_final_benchmark"}
    if not isinstance(contracts, Mapping) or set(contracts) != expected:
        raise DynamicSystemDynamicsError("system-dynamics contracts must exactly cover admitted stages")
    normalized: dict[str, Any] = {}
    for stage_id, schema in contracts.items():
        if not isinstance(schema, Mapping):
            raise DynamicSystemDynamicsError(f"invalid system-dynamics contract: {stage_id}")
        Draft202012Validator.check_schema(dict(schema))
        normalized[str(stage_id)] = dict(schema)
    return normalized


def _validate_stage_output(stage_id: str, result: Mapping[str, Any], contracts: Mapping[str, Any]) -> None:
    schema = contracts.get(stage_id)
    if not isinstance(schema, Mapping):
        raise DynamicSystemDynamicsError(f"no output contract for stage: {stage_id}")
    errors = sorted(Draft202012Validator(dict(schema)).iter_errors(dict(result)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicSystemDynamicsError(f"system-dynamics output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-system-dynamics-policy-v1",
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
        "maximum_stages": 5,
    }
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise DynamicSystemDynamicsError(f"unsafe system-dynamics policy: {key}")
    modes = policy.get("allowed_modes")
    if not isinstance(modes, list) or set(modes) != set(ROBUSTNESS_PARAMETERS):
        raise DynamicSystemDynamicsError("system-dynamics allowed_modes mismatch")
    if policy.get("allowed_operations") != ["system_dynamics_simulation", "descriptive_statistics", "finance_decision_analysis"]:
        raise DynamicSystemDynamicsError("system-dynamics allowed_operations mismatch")
    adapters = policy.get("allowed_adapters")
    if not isinstance(adapters, list) or len(adapters) != len(set(adapters)):
        raise DynamicSystemDynamicsError("system-dynamics allowed_adapters must be unique")
    solver = _mapping(policy.get("solver_policy"), "solver_policy")
    if solver.get("require_optimal_status") is not True or int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicSystemDynamicsError("system-dynamics selector must require OPTIMAL with one worker")
    if not 0 < float(solver.get("max_time_seconds") or 0) <= 10:
        raise DynamicSystemDynamicsError("system-dynamics selector time bound is invalid")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    expected_rules = ["trajectory_statistics", "robustness_simulation", "robustness_audit", "external_final_benchmark"]
    if list(rules) != expected_rules:
        raise DynamicSystemDynamicsError("system-dynamics optional rule order is fixed")
    for stage_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"stage_rules.{stage_id}")
        penalty = rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicSystemDynamicsError(f"invalid system-dynamics penalty: {stage_id}")
        benefits = _mapping(rule.get("benefits"), f"stage_rules.{stage_id}.benefits")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in benefits.values()):
            raise DynamicSystemDynamicsError(f"invalid system-dynamics benefits: {stage_id}")
        for name in ("eligible_all", "required_if_any"):
            values = rule.get(name, [])
            if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                raise DynamicSystemDynamicsError(f"{name} must be a string array: {stage_id}")
        coupled = rule.get("coupled_equal_to")
        if coupled is not None and coupled not in expected_rules:
            raise DynamicSystemDynamicsError(f"unknown system-dynamics coupling target: {coupled}")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_json(GRAPH_PATH)
    if graph.get("schema_version") != "compute-dynamic-system-dynamics-capability-graph-v1":
        raise DynamicSystemDynamicsError("invalid system-dynamics graph schema")
    if graph.get("status") != "controlled-preview" or graph.get("family") != FAMILY:
        raise DynamicSystemDynamicsError("system-dynamics graph identity mismatch")
    if graph.get("selection_engine") != "ortools-cp-sat" or graph.get("graph_engine") != "networkx":
        raise DynamicSystemDynamicsError("system-dynamics graph engine mismatch")
    safety = _mapping(graph.get("safety"), "graph.safety")
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
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise DynamicSystemDynamicsError(f"unsafe system-dynamics graph policy: {key}")
    order = [str(item) for item in _sequence(graph.get("node_order"), "graph.node_order")]
    expected_order = ["primary_simulation", "trajectory_statistics", "robustness_simulation", "robustness_audit", "external_final_benchmark"]
    raw_nodes = _mapping(graph.get("nodes"), "graph.nodes")
    if order != expected_order or set(raw_nodes) != set(expected_order):
        raise DynamicSystemDynamicsError("system-dynamics node order is fixed")
    allowed_operations = set(policy["allowed_operations"])
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(raw_nodes[stage_id], f"graph.nodes.{stage_id}"))
        if node.get("operation") not in allowed_operations:
            raise DynamicSystemDynamicsError(f"operation not allowlisted: {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicSystemDynamicsError(f"adapter not allowlisted: {adapter}")
        nodes[stage_id] = node
    if nodes["primary_simulation"].get("required") is not True or nodes["primary_simulation"].get("result_required") is not True:
        raise DynamicSystemDynamicsError("primary simulation must be required result stage")
    for stage_id in order[1:]:
        if nodes[stage_id].get("required") is True or nodes[stage_id].get("result_required") is True:
            raise DynamicSystemDynamicsError(f"optional system-dynamics node marked required: {stage_id}")
    edges: list[tuple[str, str]] = []
    for raw_edge in _sequence(graph.get("precedence"), "graph.precedence"):
        edge = _sequence(raw_edge, "graph.precedence[]")
        if len(edge) != 2:
            raise DynamicSystemDynamicsError("system-dynamics precedence edge requires two nodes")
        edges.append((str(edge[0]), str(edge[1])))
    expected_edges = {
        ("primary_simulation", "trajectory_statistics"),
        ("primary_simulation", "robustness_simulation"),
        ("primary_simulation", "robustness_audit"),
        ("robustness_simulation", "robustness_audit"),
        ("primary_simulation", "external_final_benchmark"),
    }
    if set(edges) != expected_edges:
        raise DynamicSystemDynamicsError("system-dynamics DAG does not match controlled branch-and-join structure")
    full = nx.DiGraph()
    full.add_nodes_from(order)
    full.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full) or max(dict(full.out_degree()).values()) < 2:
        raise DynamicSystemDynamicsError("system-dynamics graph must be an acyclic branching DAG")
    index = {stage_id: position for position, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full, key=lambda node: index[node])) != order:
        raise DynamicSystemDynamicsError("system-dynamics policy order disagrees with NetworkX topology")
    return {"nodes": nodes, "precedence": edges, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicSystemDynamicsError("ticket was not routed to system-dynamics family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    mode = str(inputs.get("mode") or "")
    if mode not in policy["allowed_modes"]:
        raise DynamicSystemDynamicsError("system-dynamics mode is not admitted")
    steps = inputs.get("steps", 100)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10_000:
        raise DynamicSystemDynamicsError("system-dynamics steps must be an integer from 1 to 10000")
    dt = _finite(inputs.get("dt", 1.0), "inputs.dt")
    if not 0 < dt <= 1000:
        raise DynamicSystemDynamicsError("system-dynamics dt must be in (0,1000]")
    context_raw = inputs.get("system_dynamics_context")
    context = {} if context_raw is None else dict(_mapping(context_raw, "inputs.system_dynamics_context"))
    allowed_context = {
        "trajectory_summary_requested",
        "target_metric",
        "target_stock_name",
        "robustness_parameter",
        "perturbation_fraction",
        "max_absolute_deviation",
        "external_final_value",
        "external_final_tolerance",
    }
    unexpected = sorted(set(context) - allowed_context)
    if unexpected:
        raise DynamicSystemDynamicsError(f"system_dynamics_context contains unsupported fields: {unexpected}")
    trajectory_requested = context.get("trajectory_summary_requested", False)
    if not isinstance(trajectory_requested, bool):
        raise DynamicSystemDynamicsError("trajectory_summary_requested must be boolean")

    robustness_keys = {"robustness_parameter", "perturbation_fraction", "max_absolute_deviation"}
    present_robustness = {key for key in robustness_keys if key in context}
    if present_robustness and present_robustness != robustness_keys:
        raise DynamicSystemDynamicsError("robustness configuration requires parameter, perturbation_fraction, and max_absolute_deviation")
    robustness_configured = present_robustness == robustness_keys
    robustness_compatible = mode != "stock_flow"
    robustness_parameter = None
    if robustness_configured:
        if not robustness_compatible:
            raise DynamicSystemDynamicsError("stock_flow robustness perturbation is not admitted in v1")
        robustness_parameter = str(context["robustness_parameter"])
        if robustness_parameter not in ROBUSTNESS_PARAMETERS[mode]:
            raise DynamicSystemDynamicsError(f"robustness_parameter is not admitted for mode {mode}")
        fraction = _finite(context["perturbation_fraction"], "system_dynamics_context.perturbation_fraction")
        if not -0.5 <= fraction <= 1.0 or abs(fraction) < 1e-12:
            raise DynamicSystemDynamicsError("perturbation_fraction must be in [-0.5,1.0] and non-zero")
        tolerance = _finite(context["max_absolute_deviation"], "system_dynamics_context.max_absolute_deviation")
        if tolerance < 0:
            raise DynamicSystemDynamicsError("max_absolute_deviation must be non-negative")
        base = _finite(inputs.get(robustness_parameter), f"inputs.{robustness_parameter}")
        if abs(base) < 1e-15:
            raise DynamicSystemDynamicsError("robustness perturbation requires a non-zero baseline parameter")

    external_keys = {"external_final_value", "external_final_tolerance"}
    present_external = {key for key in external_keys if key in context}
    if present_external and present_external != external_keys:
        raise DynamicSystemDynamicsError("external benchmark requires final value and tolerance")
    external_available = present_external == external_keys
    if external_available:
        _finite(context["external_final_value"], "system_dynamics_context.external_final_value")
        tolerance = _finite(context["external_final_tolerance"], "system_dynamics_context.external_final_tolerance")
        if tolerance < 0:
            raise DynamicSystemDynamicsError("external_final_tolerance must be non-negative")

    decision_class = _decision_class(ticket)
    signals = {
        "trajectory_summary_requested": trajectory_requested,
        "formal_or_high_stakes": decision_class in {"formal", "high_stakes"},
        "long_horizon": steps >= 100,
        "robustness_configured": robustness_configured,
        "robustness_compatible": robustness_compatible,
        "external_benchmark_available": external_available,
    }
    features = {
        "mode": mode,
        "steps": steps,
        "dt": dt,
        "decision_class": decision_class,
        "trajectory_summary_requested": trajectory_requested,
        "robustness_configured": robustness_configured,
        "robustness_parameter": robustness_parameter,
        "external_benchmark_available": external_available,
    }
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", []))


def _selection_feasible(selected: Mapping[str, bool], rules: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    for stage_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"rules.{stage_id}")
        chosen = bool(selected[stage_id])
        if chosen and not _eligible(rule, signals):
            return False
        if _required(rule, signals) and not chosen:
            return False
        coupled = rule.get("coupled_equal_to")
        if coupled is not None and chosen != bool(selected[str(coupled)]):
            return False
    return True


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules")
    optional_ids = list(graph["optional_ids"])
    utilities: dict[str, int] = {}
    eligibility: dict[str, bool] = {}
    required: dict[str, bool] = {}
    for stage_id in optional_ids:
        rule = _mapping(rules[stage_id], f"rules.{stage_id}")
        score = -int(rule["penalty"])
        for signal_name, benefit in _mapping(rule["benefits"], f"rules.{stage_id}.benefits").items():
            score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utilities[stage_id] = score
        eligibility[stage_id] = _eligible(rule, signals)
        required[stage_id] = _required(rule, signals)

    model = cp_model.CpModel()
    variables = {stage_id: model.new_bool_var(f"select_{stage_id}") for stage_id in optional_ids}
    for stage_id in optional_ids:
        if not eligibility[stage_id]:
            model.add(variables[stage_id] == 0)
        if required[stage_id]:
            model.add(variables[stage_id] == 1)
        coupled = rules[stage_id].get("coupled_equal_to")
        if coupled is not None:
            model.add(variables[stage_id] == variables[str(coupled)])
    model.maximize(sum(utilities[stage_id] * variables[stage_id] for stage_id in optional_ids))
    solver = cp_model.CpSolver()
    solver_policy = _mapping(policy["solver_policy"], "solver_policy")
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise DynamicSystemDynamicsError(f"system-dynamics stage selection must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected = {stage_id: bool(solver.value(variables[stage_id])) for stage_id in optional_ids}
    objective = int(round(solver.objective_value))
    feasible: list[dict[str, Any]] = []
    if len(optional_ids) <= int(solver_policy["exhaustive_cross_check_max_optional_nodes"]):
        for bits in itertools.product((False, True), repeat=len(optional_ids)):
            candidate = dict(zip(optional_ids, bits, strict=True))
            if not _selection_feasible(candidate, rules, signals):
                continue
            score = sum(utilities[stage_id] * int(candidate[stage_id]) for stage_id in optional_ids)
            feasible.append({"selection": candidate, "objective": score})
        if not feasible:
            raise DynamicSystemDynamicsError("no feasible system-dynamics selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise DynamicSystemDynamicsError(f"system-dynamics CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}")
        cross = {"performed": True, "optional_node_count": len(optional_ids), "feasible_selection_count": len(feasible), "best_objective": best, "optimal_selections": optimal, "unique_optimum": len(optimal) == 1, "passed": True}
    else:
        cross = {"performed": False, "passed": True}
    return {
        "selected_nodes": selected,
        "solver_status": solver.StatusName(status),
        "objective_value": objective,
        "global_optimal_proven": True,
        "utility_by_node": utilities,
        "eligibility_by_node": eligibility,
        "required_by_node": required,
        "signals": dict(signals),
        "solver_policy": {"num_search_workers": int(solver_policy["num_search_workers"]), "random_seed": int(solver_policy["random_seed"]), "max_time_seconds": float(solver_policy["max_time_seconds"]), "require_optimal_status": True},
        "exhaustive_cross_check": cross,
    }


def plan_dynamic_system_dynamics(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    _load_contracts()
    signals, features = _signals(ticket, policy)
    optimization = _solve(policy, graph, signals)
    selected_nodes = set(REQUIRED_STAGE_IDS) | {stage_id for stage_id, selected in optimization["selected_nodes"].items() if selected}
    if len(selected_nodes) > int(policy["maximum_stages"]):
        raise DynamicSystemDynamicsError("system-dynamics plan exceeds maximum stages")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(stage_id for stage_id in graph["full_order"] if stage_id in selected_nodes)
    runtime_graph.add_edges_from((left, right) for left, right in graph["precedence"] if left in selected_nodes and right in selected_nodes)
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicSystemDynamicsError("selected system-dynamics plan contains a cycle")
    execution_order = list(nx.lexicographical_topological_sort(runtime_graph, key=lambda node: graph["index"][node]))
    expected_order = [stage_id for stage_id in graph["full_order"] if stage_id in selected_nodes]
    if execution_order != expected_order:
        raise DynamicSystemDynamicsError("NetworkX deterministic order disagrees with system-dynamics policy order")
    initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    entry_mode = str(initial_inputs.get("mode") or "")
    stage_map: dict[str, dict[str, Any]] = {}
    for stage_id in execution_order:
        node = graph["nodes"][stage_id]
        operation = str(node["operation"])
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": operation,
            "mode": str(node.get("mode") or (entry_mode if operation == "system_dynamics_simulation" else "")),
            "adapter": str(node["adapter"]),
            "depends_on": sorted(runtime_graph.predecessors(stage_id), key=lambda item: graph["index"][item]),
        }
    return {
        "id": "dynamic-auto-v1",
        "family": FAMILY,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": DECLARED_OPERATION,
        "declared_mode": entry_mode,
        "result_stage": RESULT_STAGE_ID,
        "required_stages": list(REQUIRED_STAGE_IDS),
        "stage_order": execution_order,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": [
            "system-dynamics family is selected only from the explicit system_dynamics_simulation operation and admitted structured mode",
            f"OR-Tools CP-SAT proved the policy-optimal optional validation subset; status={optimization['solver_status']}, objective={optimization['objective_value']}",
            "primary and robustness simulations reuse the fixed offline NumPy/SciPy system-dynamics operation; no ticket-supplied equations or code are executed",
            "trajectory statistics consume deterministic primary history through a fixed adapter",
            "robustness and external-final audits use the fixed benchmark_comparison operation with explicit numeric tolerances",
            "NetworkX preserves the branching DAG while execution remains strict serial deterministic topological order",
            "independent exhaustive enumeration cross-checks the bounded optional-stage optimum",
        ],
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_system_dynamics(ticket)
    contracts = _load_contracts()
    initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    elapsed_by_stage: dict[str, float] = {}
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
        "plan_sha256": _canonical_sha({"family": FAMILY, "stage_order": plan["stage_order"], "stage_map": plan["stage_map"], "planning_features": plan["planning_features"], "optimization": plan["optimization"]}),
        "stages": [{"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "mode": plan["stage_map"][stage_id]["mode"], "depends_on": plan["stage_map"][stage_id]["depends_on"], "status": "PENDING"} for stage_id in plan["stage_order"]],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = stage["operation"]
            adapter_name = stage["adapter"]
            if operation not in operations or adapter_name not in ADAPTERS:
                raise DynamicSystemDynamicsError(f"handler or adapter unavailable at {stage_id}")
            for dependency in stage["depends_on"]:
                if dependency not in stage_results:
                    raise DynamicSystemDynamicsError(f"dependency has not completed: {dependency}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicSystemDynamicsError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            elapsed_by_stage[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise DynamicSystemDynamicsError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            _validate_stage_output(stage_id, result_dict, contracts)
            if stage_id in {"robustness_audit", "external_final_benchmark"} and result_dict.get("status") != "PASS":
                raise DynamicSystemDynamicsError(f"system-dynamics validation failed at {stage_id}")
            output_sha = _canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", result_dict)
            receipt = {"stage_id": stage_id, "operation": operation, "mode": stage["mode"], "adapter": adapter_name, "depends_on": list(stage["depends_on"]), "status": "PASS", "input_sha256": input_sha, "output_sha256": output_sha}
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
    return plan, stage_results, receipts, elapsed_by_stage


def run_dynamic_system_dynamics_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicSystemDynamicsError("ticket is not an admitted system-dynamics dynamic request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, elapsed_by_stage = _execute(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started
    import numpy as np
    import ortools
    import scipy

    validation_results = {stage_id: stage_results[stage_id] for stage_id in ("trajectory_statistics", "robustness_simulation", "robustness_audit", "external_final_benchmark") if stage_id in stage_results}
    result_data: dict[str, Any] = {
        "pipeline_id": plan["id"],
        "dynamic_family": FAMILY,
        "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "stage_order": plan["stage_order"],
        "stage_dependencies": {stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]},
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "optimization": plan["optimization"],
        "stage_receipts": receipts,
        "stage_outputs": stage_results,
        "final_stage": RESULT_STAGE_ID,
        "final_result": stage_results[RESULT_STAGE_ID],
    }
    if validation_results:
        result_data["validation_results"] = validation_results
    software = {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(plan["stage_order"])
    for stage_id in plan["stage_order"]:
        for dependency in plan["stage_map"][stage_id]["depends_on"]:
            runtime_graph.add_edge(dependency, stage_id)
    graph_contains_branching = bool(runtime_graph and max(dict(runtime_graph.out_degree()).values(), default=0) > 1)
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
        "maturity_assessment": {"engineering_maturity": "controlled-preview", "evidence_maturity": "controlled-preview"},
        "software": software,
        "execution": {"elapsed_seconds": round(elapsed, 6), "stage_elapsed_seconds": elapsed_by_stage, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": graph_contains_branching},
    }
    transfer["result_sha256"] = _canonical_sha({"schema_version": transfer["schema_version"], "task_id": transfer["task_id"], "operation": transfer["operation"], "input_sha256": transfer["input_sha256"], "assumptions": transfer["assumptions"], "limitations": transfer["limitations"], "results": transfer["results"], "maturity_assessment": transfer["maturity_assessment"], "software": transfer["software"]})
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(output_dir / "compute-audit.json", {
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
        "graph_contains_branching": graph_contains_branching,
        "primary_engine": "numpy-system-dynamics",
        "ticket_supplied_equations_executed": False,
        "secret_values_included": False
    })
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic family: `{FAMILY}`\n"
        f"- Mode: `{plan['declared_mode']}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selector: `{plan['optimization']['solver_status']}`\n"
        f"- Selector global optimal proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        "- Network used: `false`\n"
        "- Model calls: `0`\n",
        encoding="utf-8",
    )
    return transfer
