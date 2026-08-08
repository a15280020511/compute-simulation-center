#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for controlled matrix-game analysis."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from jsonschema import Draft202012Validator
from ortools.sat.python import cp_model

from dynamic_family_router import resolve_dynamic_family
from dynamic_game_theory_adapters import ALLOWED_GAMES, install_game_theory_adapters
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_game_theory_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-game-theory-policy.json"
GRAPH_PATH = HERE / "dynamic-game-theory-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-game-theory-stage-contracts.json"
FAMILY = "game-theory"
DECLARED_OPERATION = "finance_decision_analysis"
DECLARED_MODE = "open_spiel_policy_evaluation"
REQUIRED_STAGE_IDS = ("policy_evaluation",)
RESULT_STAGE_ID = "policy_evaluation"


class DynamicGameTheoryError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicGameTheoryError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicGameTheoryError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicGameTheoryError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicGameTheoryError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicGameTheoryError(f"{name} must be finite")
    return result


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_contracts() -> dict[str, Any]:
    value = _load_json(CONTRACT_PATH)
    if value.get("schema_version") != "compute-dynamic-game-theory-stage-contracts-v1":
        raise DynamicGameTheoryError("invalid game-theory stage contract schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicGameTheoryError("game-theory stage contract identity mismatch")
    contracts = value.get("contracts")
    expected = {"policy_evaluation", "pure_equilibria", "equilibrium_count_audit", "expected_utility_audit"}
    if not isinstance(contracts, Mapping) or set(contracts) != expected:
        raise DynamicGameTheoryError("game-theory contracts must exactly cover admitted stages")
    normalized: dict[str, Any] = {}
    for stage_id, schema in contracts.items():
        if not isinstance(schema, Mapping):
            raise DynamicGameTheoryError(f"invalid game-theory contract: {stage_id}")
        Draft202012Validator.check_schema(dict(schema))
        normalized[str(stage_id)] = dict(schema)
    return normalized


def _validate_stage_output(stage_id: str, result: Mapping[str, Any], contracts: Mapping[str, Any]) -> None:
    schema = contracts.get(stage_id)
    if not isinstance(schema, Mapping):
        raise DynamicGameTheoryError(f"no game-theory output contract for stage: {stage_id}")
    errors = sorted(Draft202012Validator(dict(schema)).iter_errors(dict(result)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicGameTheoryError(f"game-theory output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-game-theory-policy-v1",
        "status": "controlled-preview",
        "family": FAMILY,
        "declared_operation": DECLARED_OPERATION,
        "declared_mode": DECLARED_MODE,
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
            raise DynamicGameTheoryError(f"unsafe game-theory policy: {key}")
    if set(policy.get("allowed_games") or []) != ALLOWED_GAMES:
        raise DynamicGameTheoryError("game-theory allowed_games mismatch")
    if policy.get("allowed_operations") != [DECLARED_OPERATION]:
        raise DynamicGameTheoryError("game-theory allowed_operations mismatch")
    adapters = policy.get("allowed_adapters")
    if not isinstance(adapters, list) or len(adapters) != len(set(adapters)):
        raise DynamicGameTheoryError("game-theory allowed_adapters must be unique")
    solver = _mapping(policy.get("solver_policy"), "solver_policy")
    if solver.get("require_optimal_status") is not True or int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicGameTheoryError("game-theory selector must require OPTIMAL with one worker")
    if not 0 < float(solver.get("max_time_seconds") or 0) <= 10:
        raise DynamicGameTheoryError("game-theory selector time bound is invalid")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    expected_rules = ["pure_equilibria", "equilibrium_count_audit", "expected_utility_audit"]
    if list(rules) != expected_rules:
        raise DynamicGameTheoryError("game-theory optional rule order is fixed")
    for stage_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"stage_rules.{stage_id}")
        penalty = rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicGameTheoryError(f"invalid game-theory penalty: {stage_id}")
        benefits = _mapping(rule.get("benefits"), f"stage_rules.{stage_id}.benefits")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in benefits.values()):
            raise DynamicGameTheoryError(f"invalid game-theory benefits: {stage_id}")
        for name in ("eligible_all", "required_if_any", "requires_selected"):
            values = rule.get(name, [])
            if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                raise DynamicGameTheoryError(f"{name} must be a string array: {stage_id}")
            if name == "requires_selected" and any(item not in expected_rules for item in values):
                raise DynamicGameTheoryError(f"unknown game-theory dependency in {stage_id}")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_json(GRAPH_PATH)
    if graph.get("schema_version") != "compute-dynamic-game-theory-capability-graph-v1":
        raise DynamicGameTheoryError("invalid game-theory graph schema")
    if graph.get("status") != "controlled-preview" or graph.get("family") != FAMILY:
        raise DynamicGameTheoryError("game-theory graph identity mismatch")
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
            raise DynamicGameTheoryError(f"unsafe game-theory graph policy: {key}")
    order = [str(item) for item in _sequence(graph.get("node_order"), "graph.node_order")]
    expected_order = ["policy_evaluation", "pure_equilibria", "equilibrium_count_audit", "expected_utility_audit"]
    raw_nodes = _mapping(graph.get("nodes"), "graph.nodes")
    if order != expected_order or set(raw_nodes) != set(expected_order):
        raise DynamicGameTheoryError("game-theory node order is fixed")
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(raw_nodes[stage_id], f"graph.nodes.{stage_id}"))
        if node.get("operation") != DECLARED_OPERATION:
            raise DynamicGameTheoryError(f"game-theory operation not allowlisted: {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicGameTheoryError(f"game-theory adapter not allowlisted: {adapter}")
        nodes[stage_id] = node
    if nodes[RESULT_STAGE_ID].get("required") is not True or nodes[RESULT_STAGE_ID].get("result_required") is not True:
        raise DynamicGameTheoryError("policy_evaluation must be the required result stage")
    edges: list[tuple[str, str]] = []
    for raw_edge in _sequence(graph.get("precedence"), "graph.precedence"):
        edge = _sequence(raw_edge, "graph.precedence[]")
        if len(edge) != 2:
            raise DynamicGameTheoryError("game-theory precedence edge requires two nodes")
        edges.append((str(edge[0]), str(edge[1])))
    expected_edges = {
        ("policy_evaluation", "pure_equilibria"),
        ("pure_equilibria", "equilibrium_count_audit"),
        ("policy_evaluation", "expected_utility_audit"),
    }
    if set(edges) != expected_edges:
        raise DynamicGameTheoryError("game-theory DAG does not match controlled structure")
    full = nx.DiGraph()
    full.add_nodes_from(order)
    full.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full) or max(dict(full.out_degree()).values()) < 2:
        raise DynamicGameTheoryError("game-theory graph must be an acyclic branching DAG")
    index = {stage_id: position for position, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full, key=lambda node: index[node])) != order:
        raise DynamicGameTheoryError("game-theory policy order disagrees with NetworkX topology")
    return {"nodes": nodes, "precedence": edges, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicGameTheoryError("ticket was not routed to game-theory family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != DECLARED_MODE:
        raise DynamicGameTheoryError("game-theory entry mode mismatch")
    game_id = str(inputs.get("game_id") or "matrix_rps")
    if game_id not in ALLOWED_GAMES:
        raise DynamicGameTheoryError("game_id is outside the controlled allowlist")
    context_raw = inputs.get("game_context")
    context = {} if context_raw is None else dict(_mapping(context_raw, "inputs.game_context"))
    allowed_context = {"equilibrium_analysis_requested", "expected_pure_equilibrium_count", "expected_policy_utility", "utility_tolerance"}
    unexpected = sorted(set(context) - allowed_context)
    if unexpected:
        raise DynamicGameTheoryError(f"game_context contains unsupported fields: {unexpected}")
    requested = context.get("equilibrium_analysis_requested", False)
    if not isinstance(requested, bool):
        raise DynamicGameTheoryError("equilibrium_analysis_requested must be boolean")
    count_available = "expected_pure_equilibrium_count" in context
    if count_available:
        expected_count = context["expected_pure_equilibrium_count"]
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or not 0 <= expected_count <= 900:
            raise DynamicGameTheoryError("expected_pure_equilibrium_count must be an integer from 0 to 900")
    utility_keys = {"expected_policy_utility", "utility_tolerance"}
    present_utility = {key for key in utility_keys if key in context}
    if present_utility and present_utility != utility_keys:
        raise DynamicGameTheoryError("expected utility audit requires expected_policy_utility and utility_tolerance")
    utility_available = present_utility == utility_keys
    if utility_available:
        values = _sequence(context["expected_policy_utility"], "game_context.expected_policy_utility")
        if len(values) != 2:
            raise DynamicGameTheoryError("expected_policy_utility must contain two values")
        _finite(values[0], "game_context.expected_policy_utility[0]")
        _finite(values[1], "game_context.expected_policy_utility[1]")
        tolerance = _finite(context["utility_tolerance"], "game_context.utility_tolerance")
        if tolerance < 0:
            raise DynamicGameTheoryError("utility_tolerance must be non-negative")
    decision_class = _decision_class(ticket)
    signals = {
        "equilibrium_analysis_requested": requested,
        "expected_equilibrium_count_available": count_available,
        "expected_policy_utility_available": utility_available,
        "formal_or_high_stakes": decision_class in {"formal", "high_stakes"},
    }
    return signals, {
        "game_id": game_id,
        "decision_class": decision_class,
        "equilibrium_analysis_requested": requested,
        "expected_equilibrium_count_available": count_available,
        "expected_policy_utility_available": utility_available,
    }


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
        if chosen and any(not bool(selected[name]) for name in rule.get("requires_selected", [])):
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
        for dependency in rules[stage_id].get("requires_selected", []):
            model.add(variables[stage_id] <= variables[str(dependency)])
    model.maximize(sum(utilities[stage_id] * variables[stage_id] for stage_id in optional_ids))
    solver = cp_model.CpSolver()
    solver_policy = _mapping(policy["solver_policy"], "solver_policy")
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise DynamicGameTheoryError(f"game-theory stage selection must prove OPTIMAL; observed {solver.StatusName(status)}")
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
            raise DynamicGameTheoryError("no feasible game-theory selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise DynamicGameTheoryError(f"game-theory CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}")
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


def plan_dynamic_game_theory(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    _load_contracts()
    signals, features = _signals(ticket)
    optimization = _solve(policy, graph, signals)
    selected_nodes = set(REQUIRED_STAGE_IDS) | {stage_id for stage_id, chosen in optimization["selected_nodes"].items() if chosen}
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(stage_id for stage_id in graph["full_order"] if stage_id in selected_nodes)
    runtime_graph.add_edges_from((left, right) for left, right in graph["precedence"] if left in selected_nodes and right in selected_nodes)
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicGameTheoryError("selected game-theory plan contains a cycle")
    execution_order = list(nx.lexicographical_topological_sort(runtime_graph, key=lambda node: graph["index"][node]))
    expected_order = [stage_id for stage_id in graph["full_order"] if stage_id in selected_nodes]
    if execution_order != expected_order:
        raise DynamicGameTheoryError("NetworkX deterministic order disagrees with game-theory policy order")
    stage_map: dict[str, dict[str, Any]] = {}
    for stage_id in execution_order:
        node = graph["nodes"][stage_id]
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "mode": str(node.get("mode") or ""),
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
        "declared_mode": DECLARED_MODE,
        "result_stage": RESULT_STAGE_ID,
        "required_stages": list(REQUIRED_STAGE_IDS),
        "stage_order": execution_order,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": [
            "game-theory family is selected only from finance_decision_analysis:open_spiel_policy_evaluation and the fixed matrix_rps/matrix_pd allowlist",
            f"OR-Tools CP-SAT proved the policy-optimal optional validation subset; status={optimization['solver_status']}, objective={optimization['objective_value']}",
            "OpenSpiel evaluates the governed matrix game and produces the payoff tensor; user-defined game code is forbidden",
            "a fixed adapter converts the same payoff tensor into PyGambit row/column payoff matrices for independent pure-equilibrium analysis",
            "benchmark_comparison validates explicit equilibrium-count and expected-policy-utility claims without natural-language inference",
            "NetworkX preserves the branching DAG while execution remains strict serial deterministic topological order",
            "independent exhaustive enumeration cross-checks the bounded optional-stage optimum",
        ],
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_game_theory(ticket)
    contracts = _load_contracts()
    initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    elapsed_by_stage: dict[str, float] = {}
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": plan["id"], "family": FAMILY, "status": "RUNNING",
        "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False, "network_used": False, "model_calls": 0,
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
                raise DynamicGameTheoryError(f"handler or adapter unavailable at {stage_id}")
            for dependency in stage["depends_on"]:
                if dependency not in stage_results:
                    raise DynamicGameTheoryError(f"dependency has not completed: {dependency}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicGameTheoryError(f"adapter failed at {stage_id}: {exc}") from exc
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
                raise DynamicGameTheoryError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            _validate_stage_output(stage_id, result_dict, contracts)
            if stage_id in {"equilibrium_count_audit", "expected_utility_audit"} and result_dict.get("status") != "PASS":
                raise DynamicGameTheoryError(f"game-theory validation failed at {stage_id}")
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
            if row["status"] == "RUNNING": row["status"] = "FAILED"
        _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise
    state["status"] = "PASS"
    state["pipeline_sha256"] = _canonical_sha(receipts)
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return plan, stage_results, receipts, elapsed_by_stage


def run_dynamic_game_theory_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicGameTheoryError("ticket is not an admitted game-theory dynamic request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, elapsed_by_stage = _execute(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started
    import numpy as np
    import ortools
    import scipy
    validation_results = {stage_id: stage_results[stage_id] for stage_id in ("pure_equilibria", "equilibrium_count_audit", "expected_utility_audit") if stage_id in stage_results}
    result_data: dict[str, Any] = {
        "pipeline_id": plan["id"], "dynamic_family": FAMILY, "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False, "stage_order": plan["stage_order"],
        "stage_dependencies": {stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]},
        "planning_features": plan["planning_features"], "planning_reasons": plan["planning_reasons"], "optimization": plan["optimization"],
        "stage_receipts": receipts, "stage_outputs": stage_results, "final_stage": RESULT_STAGE_ID, "final_result": stage_results[RESULT_STAGE_ID],
    }
    if validation_results: result_data["validation_results"] = validation_results
    software: dict[str, Any] = {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    if "policy_evaluation" in stage_results: software["open-spiel"] = _package_version("open-spiel")
    if "pure_equilibria" in stage_results: software["pygambit"] = _package_version("pygambit")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(plan["stage_order"])
    for stage_id in plan["stage_order"]:
        for dependency in plan["stage_map"][stage_id]["depends_on"]: runtime_graph.add_edge(dependency, stage_id)
    graph_contains_branching = bool(runtime_graph and max(dict(runtime_graph.out_degree()).values(), default=0) > 1)
    transfer: dict[str, Any] = {
        "schema_version": "compute-result-v1", "task_id": str(ticket["task_id"]), "status": "success", "operation": str(ticket["operation"]),
        "objective": ticket.get("objective"), "input_sha256": _canonical_sha(ticket), "assumptions": ticket.get("assumptions", []), "evidence": ticket.get("evidence", []), "limitations": ticket.get("limitations", []),
        "results": result_data, "maturity_assessment": {"engineering_maturity": "controlled-preview", "evidence_maturity": "controlled-preview"}, "software": software,
        "execution": {"elapsed_seconds": round(elapsed, 6), "stage_elapsed_seconds": elapsed_by_stage, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": graph_contains_branching},
    }
    transfer["result_sha256"] = _canonical_sha({"schema_version": transfer["schema_version"], "task_id": transfer["task_id"], "operation": transfer["operation"], "input_sha256": transfer["input_sha256"], "assumptions": transfer["assumptions"], "limitations": transfer["limitations"], "results": transfer["results"], "maturity_assessment": transfer["maturity_assessment"], "software": transfer["software"]})
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(output_dir / "compute-audit.json", {
        "version": 1, "status": "PASS", "task_id": transfer["task_id"], "operation": transfer["operation"], "pipeline_id": plan["id"], "dynamic_family": FAMILY,
        "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"], "solver_status": plan["optimization"]["solver_status"], "global_optimal_proven": plan["optimization"]["global_optimal_proven"],
        "input_sha256": transfer["input_sha256"], "result_sha256": transfer["result_sha256"], "elapsed_seconds": transfer["execution"]["elapsed_seconds"], "model_calls": 0, "network_used": False, "automatic_parallel_execution": False,
        "graph_contains_branching": graph_contains_branching, "primary_engine": "open-spiel", "cross_check_engine": "pygambit" if "pure_equilibria" in stage_results else None,
        "user_defined_game_code_allowed": False, "ticket_supplied_code_executed": False, "secret_values_included": False,
    })
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n- Operation: `{transfer['operation']}`\n- Dynamic family: `{FAMILY}`\n"
        f"- Game: `{plan['planning_features']['game_id']}`\n- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selector: `{plan['optimization']['solver_status']}`\n- Selector global optimal proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        "- Network used: `false`\n- Model calls: `0`\n", encoding="utf-8")
    return transfer
