#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for the isolated compute center.

OR-Tools CP-SAT selects the globally optimal feasible subset of allowlisted optional
stages under an explicit integer policy objective. NetworkX validates the repository-
controlled capability graph and provides deterministic serial precedence. Free-form
objective text never routes tools; only structured ticket signals do. Execution remains
serial, offline, deterministic, contract checked, and bounded by the existing compute
center governance controls.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from ortools.sat.python import cp_model

from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError
from pipeline_engine import PipelineEngineError, _validate_output, load_contracts

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-orchestration-policy.json"
GRAPH_PATH = HERE / "dynamic-capability-graph.json"
DYNAMIC_PIPELINE_ID = "dynamic-auto-v1"
DYNAMIC_STAGE_ID = "dynamic"
ENTRY_OPERATION = "scenario_compare"
ENTRY_STAGE_ID = "scenarios"


class DynamicPlanningError(ValueError):
    """Raised when a dynamic plan cannot be generated or executed safely."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicPlanningError(f"JSON root must be an object: {path.name}")
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def is_dynamic_pipeline_ticket(ticket: Mapping[str, Any]) -> bool:
    pipeline = ticket.get("pipeline")
    return bool(
        isinstance(pipeline, Mapping)
        and str(pipeline.get("pipeline_id") or "") == DYNAMIC_PIPELINE_ID
        and str(pipeline.get("stage_id") or "") == DYNAMIC_STAGE_ID
        and str(ticket.get("operation") or "") == ENTRY_OPERATION
    )


def _load_policy() -> dict[str, Any]:
    value = _load_json(POLICY_PATH)
    if value.get("schema_version") != "compute-dynamic-orchestration-policy-v5":
        raise DynamicPlanningError("invalid dynamic orchestration policy schema")
    expected = {
        "status": "controlled-preview",
        "planner": "ortools-cp-sat",
        "graph_engine": "networkx",
        "network_policy": "deny",
        "model_calls": 0,
        "objective_text_routing_allowed": False,
        "structured_signals_only": True,
        "dynamic_stage_selection_allowed": True,
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_code_allowed": False,
        "automatic_parallel_execution": False,
        "cycles_allowed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise DynamicPlanningError(f"unsafe dynamic orchestration policy: {key}")
    if int(value.get("maximum_stages") or 0) != 8:
        raise DynamicPlanningError("maximum_stages must equal 8")

    activation = value.get("production_activation")
    if not isinstance(activation, Mapping):
        raise DynamicPlanningError("production_activation is required")
    if (
        activation.get("pipeline_id") != DYNAMIC_PIPELINE_ID
        or activation.get("stage_id") != DYNAMIC_STAGE_ID
        or activation.get("entry_operation") != ENTRY_OPERATION
    ):
        raise DynamicPlanningError("dynamic production activation contract mismatch")

    allowed_operations = value.get("allowed_operations")
    allowed_adapters = value.get("allowed_adapters")
    if not isinstance(allowed_operations, list) or not allowed_operations:
        raise DynamicPlanningError("allowed_operations must be a non-empty array")
    if not isinstance(allowed_adapters, list) or not allowed_adapters:
        raise DynamicPlanningError("allowed_adapters must be a non-empty array")
    if len(allowed_operations) != len(set(str(item) for item in allowed_operations)):
        raise DynamicPlanningError("allowed_operations contains duplicates")
    if len(allowed_adapters) != len(set(str(item) for item in allowed_adapters)):
        raise DynamicPlanningError("allowed_adapters contains duplicates")

    solver_policy = value.get("solver_policy")
    if not isinstance(solver_policy, Mapping):
        raise DynamicPlanningError("solver_policy is required")
    if solver_policy.get("require_optimal_status") is not True:
        raise DynamicPlanningError("dynamic orchestration must require OPTIMAL solver status")
    if int(solver_policy.get("num_search_workers") or 0) != 1:
        raise DynamicPlanningError("dynamic orchestration must use one deterministic CP-SAT worker")
    max_time = solver_policy.get("max_time_seconds")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not 0 < float(max_time) <= 10:
        raise DynamicPlanningError("solver max_time_seconds must be in (0,10]")
    max_optional = int(solver_policy.get("exhaustive_cross_check_max_optional_nodes") or 0)
    if not 1 <= max_optional <= 16:
        raise DynamicPlanningError("exhaustive cross-check optional-node limit must be in [1,16]")

    selection = value.get("selection_policy")
    if not isinstance(selection, Mapping):
        raise DynamicPlanningError("selection_policy is required")
    rules = selection.get("stage_rules")
    if not isinstance(rules, Mapping) or not rules:
        raise DynamicPlanningError("selection_policy.stage_rules must be a non-empty object")
    for node_id, raw_rule in rules.items():
        if not isinstance(raw_rule, Mapping):
            raise DynamicPlanningError(f"invalid stage rule: {node_id}")
        if not str(raw_rule.get("operation") or ""):
            raise DynamicPlanningError(f"stage rule operation missing: {node_id}")
        penalty = raw_rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicPlanningError(f"stage rule penalty must be a non-negative integer: {node_id}")
        benefits = raw_rule.get("benefits")
        if not isinstance(benefits, Mapping):
            raise DynamicPlanningError(f"stage rule benefits must be an object: {node_id}")
        for signal, value_raw in benefits.items():
            if not str(signal) or isinstance(value_raw, bool) or not isinstance(value_raw, int):
                raise DynamicPlanningError(f"invalid benefit in stage rule: {node_id}")
        for field in ("eligible_all", "required_if_any"):
            raw = raw_rule.get(field, [])
            if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw):
                raise DynamicPlanningError(f"{field} must be a string array: {node_id}")
        required_all = raw_rule.get("required_if_all", [])
        if not isinstance(required_all, list):
            raise DynamicPlanningError(f"required_if_all must be an array: {node_id}")
        for group in required_all:
            if not isinstance(group, list) or not group or any(not isinstance(item, str) or not item for item in group):
                raise DynamicPlanningError(f"required_if_all groups must contain signal names: {node_id}")
    return value


def _load_capability_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(GRAPH_PATH)
    if value.get("schema_version") != "compute-dynamic-capability-graph-v2":
        raise DynamicPlanningError("invalid dynamic capability graph schema")
    if value.get("status") != "controlled-preview":
        raise DynamicPlanningError("dynamic capability graph must remain controlled-preview")
    if value.get("graph_engine") != "networkx" or value.get("selection_engine") != "ortools-cp-sat":
        raise DynamicPlanningError("dynamic capability graph engine mismatch")

    safety = value.get("safety")
    if not isinstance(safety, Mapping):
        raise DynamicPlanningError("dynamic capability graph safety policy missing")
    expected_safety = {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "full_graph_must_be_single_serial_chain": True,
    }
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise DynamicPlanningError(f"unsafe capability graph policy: {key}")

    raw_nodes = value.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise DynamicPlanningError("dynamic capability graph has no nodes")
    nodes: dict[str, dict[str, Any]] = {}
    allowed_operations = {str(item) for item in policy["allowed_operations"]}
    allowed_adapters = {str(item) for item in policy["allowed_adapters"]}
    contracts = load_contracts()
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise DynamicPlanningError("capability graph node must be an object")
        node = dict(raw)
        node_id = str(node.get("id") or "")
        operation = str(node.get("operation") or "")
        adapter = str(node.get("adapter") or "")
        if not node_id or node_id in nodes:
            raise DynamicPlanningError(f"invalid or duplicate capability node: {node_id!r}")
        if operation not in allowed_operations or operation not in contracts:
            raise DynamicPlanningError(f"capability node operation is not contract-allowlisted: {operation}")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicPlanningError(f"capability node adapter not allowlisted: {adapter}")
        nodes[node_id] = node

    if ENTRY_STAGE_ID not in nodes:
        raise DynamicPlanningError("entry capability node is missing")
    entry = nodes[ENTRY_STAGE_ID]
    if entry.get("role") != "entry" or entry.get("operation") != ENTRY_OPERATION or entry.get("adapter") != "ticket_inputs":
        raise DynamicPlanningError("entry capability node contract mismatch")
    entry_nodes = [node_id for node_id, node in nodes.items() if node.get("role") == "entry"]
    if entry_nodes != [ENTRY_STAGE_ID]:
        raise DynamicPlanningError("dynamic capability graph must contain exactly one entry node")

    rules = policy["selection_policy"]["stage_rules"]
    optional_ids = set(nodes) - {ENTRY_STAGE_ID}
    if optional_ids != {str(item) for item in rules}:
        raise DynamicPlanningError("capability graph optional nodes must exactly match selection stage rules")
    for node_id in optional_ids:
        if str(rules[node_id]["operation"]) != str(nodes[node_id]["operation"]):
            raise DynamicPlanningError(f"stage rule operation mismatch: {node_id}")

    precedence = value.get("precedence")
    if not isinstance(precedence, list):
        raise DynamicPlanningError("capability graph precedence must be an array")
    edges: list[tuple[str, str]] = []
    for raw in precedence:
        if not isinstance(raw, list) or len(raw) != 2:
            raise DynamicPlanningError("capability graph precedence edge must contain two node ids")
        left, right = str(raw[0]), str(raw[1])
        if left not in nodes or right not in nodes or left == right:
            raise DynamicPlanningError(f"invalid capability graph edge: {left}->{right}")
        edges.append((left, right))
    if len(edges) != len(set(edges)):
        raise DynamicPlanningError("duplicate capability graph edge")

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicPlanningError("capability graph contains a cycle")
    full_order = list(nx.topological_sort(graph))
    if not full_order or full_order[0] != ENTRY_STAGE_ID:
        raise DynamicPlanningError("entry stage must be first in capability precedence")
    expected_edges = set(zip(full_order, full_order[1:], strict=False))
    if set(edges) != expected_edges:
        raise DynamicPlanningError("full capability graph must be one explicit serial chain")
    for index, node_id in enumerate(full_order):
        expected_in = 0 if index == 0 else 1
        expected_out = 0 if index == len(full_order) - 1 else 1
        if graph.in_degree(node_id) != expected_in or graph.out_degree(node_id) != expected_out:
            raise DynamicPlanningError("capability graph branching or disconnected nodes are forbidden")
    return {"nodes": nodes, "precedence": edges, "full_order": full_order}


def _uncertainty_count(ticket: Mapping[str, Any]) -> int:
    count = 0
    data_context = ticket.get("data_context")
    variables = data_context.get("variables") if isinstance(data_context, Mapping) else []
    if isinstance(variables, list):
        for row in variables:
            if not isinstance(row, Mapping):
                continue
            confidence = str(row.get("confidence") or "")
            source_type = str(row.get("source_type") or "")
            if confidence in {"low", "medium"} or source_type in {
                "proxy",
                "gpts_assumption",
                "expert_hypothesis",
            }:
                count += 1
    assumptions = ticket.get("assumptions")
    if isinstance(assumptions, list):
        for row in assumptions:
            if isinstance(row, Mapping) and str(row.get("confidence") or "") in {"low", "medium"}:
                count += 1
    return count


def _scenario_features(ticket: Mapping[str, Any]) -> tuple[int, int, dict[str, set[float]]]:
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicPlanningError("ticket inputs must be an object")
    scenarios = inputs.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise DynamicPlanningError("dynamic scenario orchestration requires inputs.scenarios")
    values_by_name: dict[str, set[float]] = {}
    for row in scenarios:
        if not isinstance(row, Mapping) or not isinstance(row.get("values"), Mapping):
            raise DynamicPlanningError("each scenario must contain values")
        for name, raw in row["values"].items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise DynamicPlanningError(f"scenario value must be numeric: {name}")
            values_by_name.setdefault(str(name), set()).add(float(raw))
    varied = sum(len(values) > 1 for values in values_by_name.values())
    return len(scenarios), varied, values_by_name


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _probabilistic_claim(ticket: Mapping[str, Any]) -> bool:
    profile = ticket.get("quality_profile")
    return bool(profile.get("probabilistic_claim", False)) if isinstance(profile, Mapping) else False


def _continuous_decision_signal(ticket: Mapping[str, Any], values_by_name: Mapping[str, set[float]]) -> bool:
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    context = inputs.get("dynamic_context")
    if context is None:
        return False
    if not isinstance(context, Mapping):
        raise DynamicPlanningError("inputs.dynamic_context must be an object")
    requested = context.get("continuous_decision_optimization") is True
    if not requested:
        return False
    if context.get("allow_continuous_interpolation") is not True:
        raise DynamicPlanningError("continuous optimization requires allow_continuous_interpolation=true")
    raw_names = context.get("controllable_variables")
    if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
        raise DynamicPlanningError("continuous optimization requires controllable_variables array")
    names = [str(item) for item in raw_names]
    if not names or len(names) != len(set(names)):
        raise DynamicPlanningError("controllable_variables must be non-empty and unique")
    model = inputs.get("model")
    coefficients = model.get("coefficients") if isinstance(model, Mapping) else None
    if not isinstance(coefficients, Mapping) or not coefficients:
        raise DynamicPlanningError("continuous optimization requires model coefficients")
    model_names = {str(name) for name in coefficients}
    if set(names) != model_names:
        raise DynamicPlanningError("every model variable must be explicitly declared controllable")
    for name in names:
        values = values_by_name.get(name)
        if not values or len(values) < 2:
            raise DynamicPlanningError(
                f"continuous optimization requires at least two observed scenario values for controllable variable {name}"
            )
    return True


def _deterministic_seed(ticket: Mapping[str, Any]) -> int:
    payload = {
        "task_id": str(ticket.get("task_id") or "dynamic-plan"),
        "operation": str(ticket.get("operation") or ""),
        "inputs": ticket.get("inputs"),
        "quality_profile": ticket.get("quality_profile"),
        "data_context": ticket.get("data_context"),
        "assumptions": ticket.get("assumptions"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _selection_signals(
    *,
    policy: Mapping[str, Any],
    scenario_count: int,
    varied_variable_count: int,
    uncertainty_count: int,
    probabilistic: bool,
    decision_class: str,
    continuous_decision: bool,
) -> dict[str, bool]:
    selection = policy["selection_policy"]
    variable_variation = varied_variable_count > 0
    return {
        "variable_variation": variable_variation,
        "scenario_summary_eligible": scenario_count >= int(selection["statistics_min_scenarios"]),
        "scenario_structure": (
            scenario_count >= int(selection["sensitivity_min_scenarios"])
            and variable_variation
        ),
        "uncertainty_trigger": uncertainty_count >= int(selection["uncertainty_trigger_count"]),
        "formal_decision": decision_class in {"formal", "high_stakes"},
        "high_stakes_uncertainty": decision_class == "high_stakes" and uncertainty_count > 0,
        "probabilistic_claim": probabilistic,
        "continuous_decision_optimization": continuous_decision,
    }


def _rule_eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    eligible_all = [str(item) for item in rule.get("eligible_all", [])]
    return all(bool(signals.get(name, False)) for name in eligible_all)


def _rule_required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    if any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", [])):
        return True
    for group in rule.get("required_if_all", []):
        if all(bool(signals.get(str(name), False)) for name in group):
            return True
    return False


def _stage_utilities(policy: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, int]:
    rules = policy["selection_policy"]["stage_rules"]
    utilities: dict[str, int] = {}
    for node_id, rule in rules.items():
        score = -int(rule["penalty"])
        for signal_name, benefit in rule["benefits"].items():
            score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utilities[str(node_id)] = score
    return utilities


def _selection_feasible(
    selected: Mapping[str, bool],
    *,
    policy: Mapping[str, Any],
    signals: Mapping[str, bool],
) -> bool:
    rules = policy["selection_policy"]["stage_rules"]
    if set(selected) != {str(item) for item in rules}:
        return False
    for node_id, rule in rules.items():
        chosen = bool(selected[str(node_id)])
        if chosen and not _rule_eligible(rule, signals):
            return False
        if _rule_required(rule, signals) and not chosen:
            return False
    return True


def _exhaustive_cross_check(
    *,
    policy: Mapping[str, Any],
    signals: Mapping[str, bool],
    optional_ids: list[str],
    utilities: Mapping[str, int],
    selected: Mapping[str, bool],
    solver_objective: int,
) -> dict[str, Any]:
    max_nodes = int(policy["solver_policy"]["exhaustive_cross_check_max_optional_nodes"])
    if len(optional_ids) > max_nodes:
        return {"performed": False, "reason": "optional-node-count-exceeds-policy"}
    feasible: list[dict[str, Any]] = []
    for bits in itertools.product((False, True), repeat=len(optional_ids)):
        candidate = dict(zip(optional_ids, bits, strict=True))
        if not _selection_feasible(candidate, policy=policy, signals=signals):
            continue
        score = sum(int(utilities[node_id]) * int(candidate[node_id]) for node_id in optional_ids)
        feasible.append({"selection": candidate, "objective": score})
    if not feasible:
        raise DynamicPlanningError("no feasible selections exist during exhaustive cross-check")
    best = max(row["objective"] for row in feasible)
    optimal = [row["selection"] for row in feasible if row["objective"] == best]
    passed = solver_objective == best and dict(selected) in optimal
    if not passed:
        raise DynamicPlanningError(
            f"CP-SAT optimum disagrees with exhaustive cross-check: solver={solver_objective}, exhaustive={best}"
        )
    return {
        "performed": True,
        "optional_node_count": len(optional_ids),
        "feasible_selection_count": len(feasible),
        "best_objective": best,
        "optimal_selections": optimal,
        "passed": True,
    }


def _solve_stage_selection(
    *,
    policy: Mapping[str, Any],
    capability_graph: Mapping[str, Any],
    signals: Mapping[str, bool],
) -> dict[str, Any]:
    rules = policy["selection_policy"]["stage_rules"]
    optional_ids = [node_id for node_id in capability_graph["full_order"] if node_id != ENTRY_STAGE_ID]
    if optional_ids != [str(item) for item in rules]:
        raise DynamicPlanningError("stage rule order must match capability graph precedence")
    utilities = _stage_utilities(policy, signals)

    model = cp_model.CpModel()
    variables = {
        node_id: model.new_bool_var(f"select_{node_id}")
        for node_id in optional_ids
    }
    for node_id in optional_ids:
        rule = rules[node_id]
        if not _rule_eligible(rule, signals):
            model.add(variables[node_id] == 0)
        if _rule_required(rule, signals):
            model.add(variables[node_id] == 1)
    model.maximize(sum(int(utilities[node_id]) * variables[node_id] for node_id in optional_ids))

    solver = cp_model.CpSolver()
    solver_policy = policy["solver_policy"]
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if solver_policy["require_optimal_status"] and status != cp_model.OPTIMAL:
        raise DynamicPlanningError(f"CP-SAT must prove OPTIMAL; observed status={status_name}")
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise DynamicPlanningError(f"OR-Tools could not find a feasible selection: {status_name}")

    selected = {node_id: bool(solver.value(variables[node_id])) for node_id in optional_ids}
    objective = int(round(solver.objective_value))
    cross_check = _exhaustive_cross_check(
        policy=policy,
        signals=signals,
        optional_ids=optional_ids,
        utilities=utilities,
        selected=selected,
        solver_objective=objective,
    )
    return {
        "selected_nodes": selected,
        "solver_status": status_name,
        "objective_value": objective,
        "global_optimal_proven": status == cp_model.OPTIMAL and bool(cross_check.get("passed", True)),
        "utility_by_node": {node_id: int(utilities[node_id]) for node_id in optional_ids},
        "utility": {
            str(rules[node_id]["operation"]): int(utilities[node_id])
            for node_id in optional_ids
        },
        "signals": dict(signals),
        "solver_policy": {
            "num_search_workers": int(solver_policy["num_search_workers"]),
            "random_seed": int(solver_policy["random_seed"]),
            "max_time_seconds": float(solver_policy["max_time_seconds"]),
            "require_optimal_status": True,
        },
        "exhaustive_cross_check": cross_check,
        "scenario_statistics": bool(selected.get("scenario_statistics", False)),
        "sensitivity": bool(selected.get("sensitivity", False)),
        "monte_carlo": bool(selected.get("risk_simulation", False)),
        "constrained_optimization": bool(selected.get("decision_optimization", False)),
    }


def _build_selected_order(
    solution: Mapping[str, Any],
    capability_graph: Mapping[str, Any],
    maximum_stages: int,
) -> list[str]:
    selected_optional = solution.get("selected_nodes")
    if not isinstance(selected_optional, Mapping):
        raise DynamicPlanningError("selected_nodes missing from solver result")
    ordered = [
        node_id
        for node_id in capability_graph["full_order"]
        if node_id == ENTRY_STAGE_ID or bool(selected_optional.get(node_id, False))
    ]
    if not ordered or ordered[0] != ENTRY_STAGE_ID:
        raise DynamicPlanningError("selected dynamic plan lost its entry stage")
    if len(ordered) > maximum_stages:
        raise DynamicPlanningError("dynamic plan exceeds maximum stages")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(ordered)
    runtime_graph.add_edges_from(zip(ordered, ordered[1:], strict=False))
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicPlanningError("selected dynamic plan contains a cycle")
    for index, stage_id in enumerate(ordered):
        expected_in = 0 if index == 0 else 1
        expected_out = 0 if index == len(ordered) - 1 else 1
        if runtime_graph.in_degree(stage_id) != expected_in or runtime_graph.out_degree(stage_id) != expected_out:
            raise DynamicPlanningError("dynamic branching or disconnected stages are forbidden")
    return ordered


def plan_dynamic_pipeline(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    capability_graph = _load_capability_graph(policy)
    operation = str(ticket.get("operation") or "")
    if operation != ENTRY_OPERATION:
        raise DynamicPlanningError("controlled-preview dynamic planner supports scenario_compare entry tickets only")

    scenario_count, varied_variable_count, values_by_name = _scenario_features(ticket)
    uncertainty_count = _uncertainty_count(ticket)
    probabilistic = _probabilistic_claim(ticket)
    decision_class = _decision_class(ticket)
    continuous_decision = _continuous_decision_signal(ticket, values_by_name)
    signals = _selection_signals(
        policy=policy,
        scenario_count=scenario_count,
        varied_variable_count=varied_variable_count,
        uncertainty_count=uncertainty_count,
        probabilistic=probabilistic,
        decision_class=decision_class,
        continuous_decision=continuous_decision,
    )
    solution = _solve_stage_selection(
        policy=policy,
        capability_graph=capability_graph,
        signals=signals,
    )
    ordered = _build_selected_order(solution, capability_graph, int(policy["maximum_stages"]))

    nodes = capability_graph["nodes"]
    stage_map: dict[str, dict[str, Any]] = {}
    for index, stage_id in enumerate(ordered):
        node = nodes[stage_id]
        stage: dict[str, Any] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "adapter": str(node["adapter"]),
            "depends_on": [] if index == 0 else [ordered[index - 1]],
        }
        if stage_id == "risk_simulation":
            stage["fixed_parameters"] = {
                "iterations": int(policy["selection_policy"]["monte_carlo_iterations"][decision_class]),
                "seed": _deterministic_seed(ticket),
            }
        stage_map[stage_id] = stage

    reasons = [
        "scenario_compare is the declared entry operation",
        (
            "OR-Tools CP-SAT proved the policy-optimal feasible optional-stage subset; "
            f"status={solution['solver_status']}, objective={solution['objective_value']}"
        ),
    ]
    if solution["exhaustive_cross_check"].get("performed"):
        reasons.append("independent exhaustive enumeration matched the CP-SAT global optimum")
    rules = policy["selection_policy"]["stage_rules"]
    for stage_id in ordered:
        if stage_id == ENTRY_STAGE_ID:
            continue
        operation_name = str(rules[stage_id]["operation"])
        reasons.append(
            f"{operation_name} selected with utility={solution['utility_by_node'][stage_id]}"
        )
    if continuous_decision:
        reasons.append(
            "constrained optimization eligibility came only from explicit structured authorization: "
            "continuous_decision_optimization=true, allow_continuous_interpolation=true, and complete controllable_variables"
        )

    return {
        "id": DYNAMIC_PIPELINE_ID,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-dynamic",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "entry_operation": operation,
        "result_stage": ordered[-1],
        "stage_order": ordered,
        "stage_map": stage_map,
        "planning_features": {
            "scenario_count": scenario_count,
            "varied_variable_count": varied_variable_count,
            "uncertainty_signal_count": uncertainty_count,
            "probabilistic_claim": probabilistic,
            "decision_class": decision_class,
            "continuous_decision_optimization": continuous_decision,
        },
        "planning_reasons": reasons,
        "optimization": solution,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute_plan(
    ticket: Mapping[str, Any],
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_pipeline(ticket)
    contracts = load_contracts()
    initial_inputs = ticket.get("inputs")
    if not isinstance(initial_inputs, Mapping):
        raise DynamicPlanningError("ticket inputs must be an object")

    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    stage_elapsed: dict[str, float] = {}
    state: dict[str, Any] | None = None
    if output_dir is not None:
        state = {
            "schema_version": "compute-dynamic-pipeline-state-v2",
            "pipeline_id": plan["id"],
            "status": "RUNNING",
            "planning_mode": plan["planning_mode"],
            "selection_engine": plan["selection_engine"],
            "graph_engine": plan["graph_engine"],
            "automatic_parallel_execution": False,
            "network_used": False,
            "model_calls": 0,
            "plan_sha256": _canonical_sha({
                "stage_order": plan["stage_order"],
                "planning_features": plan["planning_features"],
                "optimization": plan["optimization"],
            }),
            "stages": [
                {
                    "stage_id": stage_id,
                    "operation": plan["stage_map"][stage_id]["operation"],
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
            if operation not in operations:
                raise DynamicPlanningError(f"handler unavailable: {operation}")
            if state is not None:
                state["stages"][index]["status"] = "RUNNING"
                _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[str(stage["adapter"])](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicPlanningError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            if output_dir is not None:
                _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            stage_elapsed[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise DynamicPlanningError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            try:
                _validate_output(operation, result_dict, contracts)
            except PipelineEngineError as exc:
                raise DynamicPlanningError(str(exc)) from exc
            output_sha = _canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            if output_dir is not None:
                _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", result_dict)
            receipt = {
                "stage_id": stage_id,
                "operation": operation,
                "adapter": str(stage["adapter"]),
                "status": "PASS",
                "input_sha256": input_sha,
                "output_sha256": output_sha,
            }
            receipts.append(receipt)
            if state is not None:
                state["stages"][index].update(receipt)
                _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        if state is not None:
            state["status"] = "FAILED"
            for row in state["stages"]:
                if row["status"] == "RUNNING":
                    row["status"] = "FAILED"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise

    if state is not None:
        state["status"] = "PASS"
        state["pipeline_sha256"] = _canonical_sha(receipts)
        _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return plan, stage_results, receipts, stage_elapsed


def execute_dynamic_pipeline(
    ticket: Mapping[str, Any],
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    plan, stage_results, receipts, _ = _execute_plan(ticket, operations)
    return {
        "status": "PASS",
        "planner": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "stage_order": plan["stage_order"],
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "optimization": plan["optimization"],
        "receipts": receipts,
        "final_result": stage_results[plan["result_stage"]],
        "network_used": False,
        "model_calls": 0,
        "automatic_parallel_execution": False,
    }


def run_dynamic_pipeline_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    if not is_dynamic_pipeline_ticket(ticket):
        raise DynamicPlanningError(
            f"dynamic production ticket must use pipeline_id={DYNAMIC_PIPELINE_ID} and stage_id={DYNAMIC_STAGE_ID}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, stage_elapsed = _execute_plan(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started

    import numpy as np
    import ortools
    import scipy

    result_data = {
        "pipeline_id": plan["id"],
        "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "stage_order": plan["stage_order"],
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "optimization": plan["optimization"],
        "stage_receipts": receipts,
        "stage_outputs": stage_results,
        "final_stage": plan["result_stage"],
        "final_result": stage_results[plan["result_stage"]],
    }
    transfer = {
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
        },
        "execution": {
            "elapsed_seconds": round(elapsed, 6),
            "stage_elapsed_seconds": stage_elapsed,
            "network_used": False,
            "model_calls": 0,
            "reproducible": True,
            "automatic_parallel_execution": False,
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
            "secret_values_included": False,
        },
    )
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic pipeline: `{plan['id']}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selection engine: `{plan['selection_engine']}`\n"
        f"- Graph engine: `{plan['graph_engine']}`\n"
        f"- Solver status: `{plan['optimization']['solver_status']}`\n"
        f"- Global optimum proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        "- Execution policy: `strict-serial`\n"
        "- Automatic parallel execution: `false`\n"
        "- Model calls: `0`\n"
        "- Network used: `false`\n",
        encoding="utf-8",
    )
    return transfer


if __name__ == "__main__":
    print(json.dumps(_load_policy(), ensure_ascii=False, indent=2))
