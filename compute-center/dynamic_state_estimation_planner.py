#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for the bounded state-estimation family."""
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
from dynamic_state_estimation_adapters import install_state_estimation_adapters
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_state_estimation_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-state-estimation-policy.json"
GRAPH_PATH = HERE / "dynamic-state-estimation-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-state-estimation-stage-contracts.json"
FAMILY = "state-estimation"
DECLARED_OPERATION = "finance_decision_analysis"
ENTRY_MODE = "bounded_linear_kalman_filter"
REQUIRED_STAGE_IDS = ("state_estimation",)
RESULT_STAGE_ID = "state_estimation"


class DynamicStateEstimationError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicStateEstimationError(f"JSON root must be an object: {path.name}")
    return value


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicStateEstimationError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicStateEstimationError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicStateEstimationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicStateEstimationError(f"{name} must be finite")
    return result


def _matrix(value: Any, name: str, maximum_rows: int = 10000, maximum_columns: int = 20) -> list[list[float]]:
    rows = _sequence(value, name)
    if not 1 <= len(rows) <= maximum_rows:
        raise DynamicStateEstimationError(f"{name} row count is outside the governed range")
    converted: list[list[float]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"{name}[{row_index}]")
        if not 1 <= len(row) <= maximum_columns:
            raise DynamicStateEstimationError(f"{name}[{row_index}] width is outside the governed range")
        values = [_finite(item, f"{name}[{row_index}][]") for item in row]
        width = len(values) if width is None else width
        if len(values) != width:
            raise DynamicStateEstimationError(f"{name} rows must have equal width")
        converted.append(values)
    return converted


def _load_contracts() -> dict[str, Any]:
    value = _load_json(CONTRACT_PATH)
    if value.get("schema_version") != "compute-dynamic-state-estimation-stage-contracts-v1":
        raise DynamicStateEstimationError("invalid state-estimation stage contract schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicStateEstimationError("state-estimation stage contract identity mismatch")
    contracts = value.get("contracts")
    required_modes = {"bounded_linear_kalman_filter", "realized_outcome_feedback", "benchmark_comparison"}
    if not isinstance(contracts, Mapping) or set(contracts) != required_modes:
        raise DynamicStateEstimationError("state-estimation contracts must exactly cover admitted modes")
    for mode, schema in contracts.items():
        if not isinstance(schema, Mapping):
            raise DynamicStateEstimationError(f"invalid state-estimation contract: {mode}")
        Draft202012Validator.check_schema(dict(schema))
    return {str(key): dict(schema) for key, schema in contracts.items()}


def _validate_stage_output(result: Mapping[str, Any], contracts: Mapping[str, Any]) -> None:
    mode = str(result.get("mode") or "")
    schema = contracts.get(mode)
    if not isinstance(schema, Mapping):
        raise DynamicStateEstimationError(f"no dynamic state-estimation output contract for mode: {mode or '<empty>'}")
    validator = Draft202012Validator(dict(schema))
    errors = sorted(validator.iter_errors(dict(result)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicStateEstimationError(f"state-estimation output contract failed for {mode} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-state-estimation-policy-v1",
        "status": "controlled-preview",
        "family": FAMILY,
        "declared_operation": DECLARED_OPERATION,
        "entry_mode": ENTRY_MODE,
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
        "maximum_stages": 3,
    }
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise DynamicStateEstimationError(f"unsafe state-estimation policy: {key}")
    if policy.get("allowed_operations") != [DECLARED_OPERATION]:
        raise DynamicStateEstimationError("state-estimation allowed_operations must contain only finance_decision_analysis")
    adapters = policy.get("allowed_adapters")
    if not isinstance(adapters, list) or len(adapters) != len(set(adapters)):
        raise DynamicStateEstimationError("state-estimation allowed_adapters must be unique")
    solver = _mapping(policy.get("solver_policy"), "solver_policy")
    if solver.get("require_optimal_status") is not True or int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicStateEstimationError("state-estimation solver must require OPTIMAL with one worker")
    max_time = solver.get("max_time_seconds")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not 0 < float(max_time) <= 10:
        raise DynamicStateEstimationError("state-estimation solver time bound is invalid")
    if not 1 <= int(solver.get("exhaustive_cross_check_max_optional_nodes") or 0) <= 16:
        raise DynamicStateEstimationError("state-estimation exhaustive cross-check bound is invalid")
    selection = _mapping(policy.get("selection_policy"), "selection_policy")
    rules = _mapping(selection.get("stage_rules"), "selection_policy.stage_rules")
    if list(rules) != ["realized_feedback", "benchmark_check"]:
        raise DynamicStateEstimationError("state-estimation optional rule order is fixed")
    for node_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"stage_rules.{node_id}")
        if rule.get("operation") != DECLARED_OPERATION:
            raise DynamicStateEstimationError(f"state-estimation stage operation mismatch: {node_id}")
        penalty = rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicStateEstimationError(f"invalid state-estimation penalty: {node_id}")
        benefits = _mapping(rule.get("benefits"), f"stage_rules.{node_id}.benefits")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in benefits.values()):
            raise DynamicStateEstimationError(f"invalid state-estimation benefits: {node_id}")
        for name in ("eligible_all", "required_if_any", "required_if_all"):
            values = rule.get(name, [])
            if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                raise DynamicStateEstimationError(f"{name} must be a string array: {node_id}")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(GRAPH_PATH)
    if value.get("schema_version") != "compute-dynamic-state-estimation-capability-graph-v1":
        raise DynamicStateEstimationError("invalid state-estimation graph schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicStateEstimationError("state-estimation graph identity mismatch")
    if value.get("graph_engine") != "networkx" or value.get("selection_engine") != "ortools-cp-sat":
        raise DynamicStateEstimationError("state-estimation graph engine mismatch")
    safety = _mapping(value.get("safety"), "graph.safety")
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
            raise DynamicStateEstimationError(f"unsafe state-estimation graph policy: {key}")
    order = [str(item) for item in _sequence(value.get("node_order"), "graph.node_order")]
    raw_nodes = _mapping(value.get("nodes"), "graph.nodes")
    if order != ["state_estimation", "realized_feedback", "benchmark_check"] or set(order) != set(raw_nodes):
        raise DynamicStateEstimationError("state-estimation node order is fixed")
    allowed_adapters = {str(item) for item in policy["allowed_adapters"]}
    nodes: dict[str, dict[str, Any]] = {}
    for node_id in order:
        node = dict(_mapping(raw_nodes[node_id], f"graph.nodes.{node_id}"))
        if node.get("operation") != DECLARED_OPERATION:
            raise DynamicStateEstimationError(f"state-estimation operation mismatch: {node_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicStateEstimationError(f"state-estimation adapter not allowlisted: {adapter}")
        nodes[node_id] = node
    if nodes["state_estimation"].get("required") is not True or nodes["state_estimation"].get("result_required") is not True:
        raise DynamicStateEstimationError("state_estimation must be the required result stage")
    for node_id in ("realized_feedback", "benchmark_check"):
        if nodes[node_id].get("required") is True or nodes[node_id].get("result_required") is True:
            raise DynamicStateEstimationError(f"optional state-estimation node marked required: {node_id}")
    edges = []
    for raw_edge in _sequence(value.get("precedence"), "graph.precedence"):
        edge = _sequence(raw_edge, "graph.precedence[]")
        if len(edge) != 2:
            raise DynamicStateEstimationError("state-estimation precedence edges require two nodes")
        edges.append((str(edge[0]), str(edge[1])))
    expected_edges = {("state_estimation", "realized_feedback"), ("state_estimation", "benchmark_check")}
    if set(edges) != expected_edges:
        raise DynamicStateEstimationError("state-estimation DAG does not match the controlled branching structure")
    full_graph = nx.DiGraph()
    full_graph.add_nodes_from(order)
    full_graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full_graph):
        raise DynamicStateEstimationError("state-estimation graph contains a cycle")
    index = {node_id: position for position, node_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full_graph, key=lambda node: index[node])) != order:
        raise DynamicStateEstimationError("state-estimation node order disagrees with NetworkX topology")
    return {"nodes": nodes, "precedence": edges, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicStateEstimationError("ticket was not routed to state-estimation family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != ENTRY_MODE:
        raise DynamicStateEstimationError("state-estimation entry mode mismatch")
    transition = _matrix(inputs.get("transition_matrix"), "inputs.transition_matrix", maximum_rows=20)
    state_dimension = len(transition)
    if any(len(row) != state_dimension for row in transition):
        raise DynamicStateEstimationError("transition_matrix must be square")
    observation_matrix = _matrix(inputs.get("observation_matrix"), "inputs.observation_matrix", maximum_rows=20)
    observation_dimension = len(observation_matrix)
    if any(len(row) != state_dimension for row in observation_matrix):
        raise DynamicStateEstimationError("observation_matrix width must match state dimension")
    observations = _matrix(inputs.get("observations"), "inputs.observations", maximum_rows=10000)
    if any(len(row) != observation_dimension for row in observations):
        raise DynamicStateEstimationError("observation rows must match observation dimension")
    context_raw = inputs.get("dynamic_context", {})
    context = _mapping(context_raw, "inputs.dynamic_context") if context_raw is not None else {}
    for key in ("realized_feedback", "benchmark_check"):
        if key in context and not isinstance(context[key], bool):
            raise DynamicStateEstimationError(f"dynamic_context.{key} must be boolean")
    benchmark_available = False
    if "benchmark_state" in inputs or "benchmark_tolerance" in inputs:
        benchmark = [_finite(item, "inputs.benchmark_state[]") for item in _sequence(inputs.get("benchmark_state"), "inputs.benchmark_state")]
        if len(benchmark) != state_dimension:
            raise DynamicStateEstimationError("benchmark_state must match state dimension")
        raw_tolerance = inputs.get("benchmark_tolerance")
        if isinstance(raw_tolerance, Sequence) and not isinstance(raw_tolerance, (str, bytes)):
            tolerance = [_finite(item, "inputs.benchmark_tolerance[]") for item in raw_tolerance]
            if len(tolerance) != state_dimension:
                raise DynamicStateEstimationError("benchmark_tolerance array must match state dimension")
        else:
            tolerance = [_finite(raw_tolerance, "inputs.benchmark_tolerance")]
        if any(item < 0 for item in tolerance):
            raise DynamicStateEstimationError("benchmark_tolerance must be non-negative")
        benchmark_available = True
    scalar_observation = observation_dimension == 1
    long_enough = len(observations) >= 4
    decision_class = _decision_class(ticket)
    signals = {
        "realized_feedback_requested": context.get("realized_feedback") is True,
        "benchmark_requested": context.get("benchmark_check") is True,
        "formal_or_high_stakes": decision_class in {"formal", "high_stakes"},
        "long_enough_feedback_series": long_enough,
        "scalar_observation": scalar_observation,
        "benchmark_available": benchmark_available,
    }
    if signals["realized_feedback_requested"] and not (scalar_observation and long_enough):
        raise DynamicStateEstimationError("requested realized feedback requires at least four scalar observations")
    if signals["benchmark_requested"] and not benchmark_available:
        raise DynamicStateEstimationError("requested benchmark check requires benchmark_state and benchmark_tolerance")
    features = {
        "decision_class": decision_class,
        "state_dimension": state_dimension,
        "observation_dimension": observation_dimension,
        "observation_count": len(observations),
        "benchmark_available": benchmark_available,
        "structured_dynamic_context": {key: bool(context.get(key, False)) for key in ("realized_feedback", "benchmark_check")},
    }
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(name, False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    any_names = rule.get("required_if_any", [])
    all_names = rule.get("required_if_all", [])
    return any(bool(signals.get(name, False)) for name in any_names) or (
        bool(all_names) and all(bool(signals.get(name, False)) for name in all_names)
    )


def _utility(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> int:
    benefits = _mapping(rule.get("benefits"), "rule.benefits")
    return sum(int(value) for name, value in benefits.items() if signals.get(str(name), False)) - int(rule.get("penalty", 0))


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules")
    optional_ids = list(graph["optional_ids"])
    solver_policy = _mapping(policy["solver_policy"], "solver_policy")
    model = cp_model.CpModel()
    variables = {node_id: model.NewBoolVar(node_id) for node_id in optional_ids}
    utilities: dict[str, int] = {}
    eligibility: dict[str, bool] = {}
    required: dict[str, bool] = {}
    for node_id in optional_ids:
        rule = _mapping(rules[node_id], f"stage_rules.{node_id}")
        eligibility[node_id] = _eligible(rule, signals)
        required[node_id] = _required(rule, signals)
        utilities[node_id] = _utility(rule, signals)
        if not eligibility[node_id]:
            model.Add(variables[node_id] == 0)
        if required[node_id]:
            if not eligibility[node_id]:
                raise DynamicStateEstimationError(f"required state-estimation stage is ineligible: {node_id}")
            model.Add(variables[node_id] == 1)
    model.Maximize(sum(utilities[node_id] * variables[node_id] for node_id in optional_ids))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status != cp_model.OPTIMAL:
        raise DynamicStateEstimationError(f"state-estimation CP-SAT must prove OPTIMAL, got {status_name}")
    selected = {node_id: bool(solver.Value(variables[node_id])) for node_id in optional_ids}
    objective = int(round(solver.ObjectiveValue()))
    cross: dict[str, Any] = {"performed": False, "passed": True}
    if len(optional_ids) <= int(solver_policy["exhaustive_cross_check_max_optional_nodes"]):
        feasible = []
        for values in itertools.product((False, True), repeat=len(optional_ids)):
            selection = dict(zip(optional_ids, values, strict=True))
            if any(selection[node_id] and not eligibility[node_id] for node_id in optional_ids):
                continue
            if any(required[node_id] and not selection[node_id] for node_id in optional_ids):
                continue
            score = sum(utilities[node_id] for node_id in optional_ids if selection[node_id])
            feasible.append({"selection": selection, "objective": score})
        if not feasible:
            raise DynamicStateEstimationError("no feasible state-estimation selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise DynamicStateEstimationError(f"state-estimation CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}")
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
        "global_optimal_proven": True,
        "utility_by_node": utilities,
        "eligibility_by_node": eligibility,
        "required_by_node": required,
        "signals": dict(signals),
        "solver_policy": {
            "num_search_workers": int(solver_policy["num_search_workers"]),
            "random_seed": int(solver_policy["random_seed"]),
            "max_time_seconds": float(solver_policy["max_time_seconds"]),
            "require_optimal_status": True,
        },
        "exhaustive_cross_check": cross,
    }


def plan_dynamic_state_estimation(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    _load_contracts()
    signals, features = _signals(ticket)
    optimization = _solve(policy, graph, signals)
    selected_nodes = {RESULT_STAGE_ID} | {node_id for node_id, chosen in optimization["selected_nodes"].items() if chosen}
    if len(selected_nodes) > int(policy["maximum_stages"]):
        raise DynamicStateEstimationError("state-estimation plan exceeds maximum stages")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(node_id for node_id in graph["full_order"] if node_id in selected_nodes)
    runtime_graph.add_edges_from((left, right) for left, right in graph["precedence"] if left in selected_nodes and right in selected_nodes)
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicStateEstimationError("selected state-estimation plan contains a cycle")
    execution_order = list(nx.lexicographical_topological_sort(runtime_graph, key=lambda node: graph["index"][node]))
    expected_order = [node_id for node_id in graph["full_order"] if node_id in selected_nodes]
    if execution_order != expected_order:
        raise DynamicStateEstimationError("NetworkX deterministic order disagrees with state-estimation policy order")
    stage_map: dict[str, dict[str, Any]] = {}
    for stage_id in execution_order:
        node = graph["nodes"][stage_id]
        predecessors = sorted(runtime_graph.predecessors(stage_id), key=lambda item: graph["index"][item])
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "mode": str(node["mode"]),
            "adapter": str(node["adapter"]),
            "depends_on": predecessors,
        }
    reasons = [
        "state-estimation family was selected only from the exact finance_decision_analysis/bounded_linear_kalman_filter operation-mode pair and structured matrices",
        "bounded linear Kalman estimation is mandatory; optional validation branches consume its deterministic filtered-state output",
        f"OR-Tools CP-SAT proved the policy-optimal optional validation subset; status={optimization['solver_status']}, objective={optimization['objective_value']}",
        "NetworkX preserves a real branching validation DAG while execution remains strict serial deterministic topological order",
    ]
    if optimization["exhaustive_cross_check"].get("performed"):
        reasons.append("independent exhaustive enumeration matched the state-estimation CP-SAT optimum")
    return {
        "id": "dynamic-auto-v1",
        "family": FAMILY,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": DECLARED_OPERATION,
        "entry_mode": ENTRY_MODE,
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


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_state_estimation(ticket)
    contracts = _load_contracts()
    initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
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
        "plan_sha256": _canonical_sha({"family": FAMILY, "stage_order": plan["stage_order"], "stage_map": plan["stage_map"], "planning_features": plan["planning_features"], "optimization": plan["optimization"]}),
        "stages": [{"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "mode": plan["stage_map"][stage_id]["mode"], "depends_on": plan["stage_map"][stage_id]["depends_on"], "status": "PENDING"} for stage_id in plan["stage_order"]],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = str(stage["operation"])
            adapter_name = str(stage["adapter"])
            if operation not in operations or adapter_name not in ADAPTERS:
                raise DynamicStateEstimationError(f"state-estimation handler or adapter unavailable at {stage_id}")
            for dependency in stage["depends_on"]:
                if dependency not in stage_results:
                    raise DynamicStateEstimationError(f"state-estimation dependency has not completed: {dependency}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicStateEstimationError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            stage_elapsed[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise DynamicStateEstimationError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            _validate_stage_output(result_dict, contracts)
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
    return plan, stage_results, receipts, stage_elapsed


def run_dynamic_state_estimation_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicStateEstimationError("ticket is not an admitted state-estimation dynamic request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, stage_elapsed = _execute(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started
    import numpy as np
    import ortools
    import scipy

    validation_results = {stage_id: stage_results[stage_id] for stage_id in ("realized_feedback", "benchmark_check") if stage_id in stage_results}
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
        "final_stage": plan["result_stage"],
        "final_result": stage_results[plan["result_stage"]],
    }
    if validation_results:
        result_data["validation_results"] = validation_results
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
        "software": {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "execution": {"elapsed_seconds": round(elapsed, 6), "stage_elapsed_seconds": stage_elapsed, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": len(plan["stage_order"]) > 2},
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
        "graph_contains_branching": transfer["execution"]["graph_contains_branching"],
        "fixed_offline_generic_state_estimation": True,
        "live_feed_used": False,
        "individual_or_target_tracking_allowed": False,
        "secret_values_included": False
    })
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic family: `{FAMILY}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Solver: `{plan['optimization']['solver_status']}`\n"
        f"- Global optimal proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        "- Network used: `false`\n"
        "- Model calls: `0`\n"
        "- Live feed used: `false`\n",
        encoding="utf-8",
    )
    return transfer
