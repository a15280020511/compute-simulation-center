#!/usr/bin/env python3
"""Experimental optimized dynamic orchestration for the isolated compute center.

OR-Tools CP-SAT selects the best feasible subset of allowlisted analytical stages under
an explicit integer objective. NetworkX validates and orders the resulting dependency
graph. Free-form objective text never routes tools; only structured ticket signals do.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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


class DynamicPlanningError(ValueError):
    """Raised when a dynamic plan cannot be generated or executed safely."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicPlanningError(f"JSON root must be an object: {path.name}")
    return value


def _load_policy() -> dict[str, Any]:
    value = _load_json(POLICY_PATH)
    if value.get("schema_version") != "compute-dynamic-orchestration-policy-v2":
        raise DynamicPlanningError("invalid dynamic orchestration policy schema")
    expected = {
        "status": "experimental",
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
    return value


def _load_capability_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(GRAPH_PATH)
    if value.get("schema_version") != "compute-dynamic-capability-graph-v1":
        raise DynamicPlanningError("invalid dynamic capability graph schema")
    if value.get("status") != "experimental":
        raise DynamicPlanningError("dynamic capability graph must remain experimental")
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
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise DynamicPlanningError("capability graph node must be an object")
        node = dict(raw)
        node_id = str(node.get("id") or "")
        operation = str(node.get("operation") or "")
        adapter = str(node.get("adapter") or "")
        if not node_id or node_id in nodes:
            raise DynamicPlanningError(f"invalid or duplicate capability node: {node_id!r}")
        if operation not in allowed_operations:
            raise DynamicPlanningError(f"capability node operation not allowlisted: {operation}")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicPlanningError(f"capability node adapter not allowlisted: {adapter}")
        nodes[node_id] = node

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
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicPlanningError("capability graph contains a cycle")
    return {"nodes": nodes, "precedence": edges}


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
                "proxy", "gpts_assumption", "expert_hypothesis",
            }:
                count += 1
    assumptions = ticket.get("assumptions")
    if isinstance(assumptions, list):
        for row in assumptions:
            if isinstance(row, Mapping) and str(row.get("confidence") or "") in {"low", "medium"}:
                count += 1
    return count


def _scenario_features(ticket: Mapping[str, Any]) -> tuple[int, int]:
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
    return len(scenarios), varied


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _probabilistic_claim(ticket: Mapping[str, Any]) -> bool:
    profile = ticket.get("quality_profile")
    return bool(profile.get("probabilistic_claim", False)) if isinstance(profile, Mapping) else False


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


def _solve_stage_selection(
    *,
    policy: Mapping[str, Any],
    scenario_count: int,
    varied_variable_count: int,
    uncertainty_count: int,
    probabilistic: bool,
    decision_class: str,
) -> dict[str, Any]:
    selection = policy["selection_policy"]
    benefit = selection["benefit"]
    penalty = selection["stage_penalty"]
    sensitivity_structural_signal = (
        scenario_count >= int(selection["sensitivity_min_scenarios"])
        and varied_variable_count > 0
    )
    uncertainty_trigger = uncertainty_count >= int(selection["uncertainty_trigger_count"])
    formal_signal = decision_class in {"formal", "high_stakes"}

    sensitivity_utility = -int(penalty["sensitivity_analysis"])
    sensitivity_utility += int(benefit["sensitivity_from_scenario_structure"]) * int(sensitivity_structural_signal)
    sensitivity_utility += int(benefit["sensitivity_from_uncertainty"]) * int(uncertainty_trigger)
    sensitivity_utility += int(benefit["sensitivity_from_formal_decision"]) * int(formal_signal)

    monte_carlo_utility = -int(penalty["monte_carlo"])
    monte_carlo_utility += int(benefit["monte_carlo_from_probabilistic_claim"]) * int(probabilistic)
    monte_carlo_utility += int(benefit["monte_carlo_from_uncertainty"]) * int(uncertainty_trigger)
    monte_carlo_utility += int(benefit["monte_carlo_from_formal_decision"]) * int(formal_signal)

    model = cp_model.CpModel()
    sensitivity = model.new_bool_var("select_sensitivity")
    monte_carlo = model.new_bool_var("select_monte_carlo")

    if varied_variable_count == 0:
        model.add(sensitivity == 0)
    if probabilistic:
        model.add(monte_carlo == 1)
    if uncertainty_trigger:
        model.add(monte_carlo == 1)
    if decision_class == "high_stakes" and uncertainty_count > 0 and varied_variable_count > 0:
        model.add(sensitivity == 1)

    model.maximize(sensitivity_utility * sensitivity + monte_carlo_utility * monte_carlo)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 1.0
    status = solver.solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise DynamicPlanningError("OR-Tools could not find a feasible dynamic stage selection")

    return {
        "sensitivity": bool(solver.value(sensitivity)),
        "monte_carlo": bool(solver.value(monte_carlo)),
        "solver_status": solver.status_name(status),
        "objective_value": int(round(solver.objective_value)),
        "utility": {
            "sensitivity_analysis": sensitivity_utility,
            "monte_carlo": monte_carlo_utility,
        },
        "signals": {
            "sensitivity_structural_signal": sensitivity_structural_signal,
            "uncertainty_trigger": uncertainty_trigger,
            "formal_signal": formal_signal,
        },
    }


def _build_selected_graph(
    selected_ids: list[str],
    capability_graph: Mapping[str, Any],
) -> list[str]:
    selected = set(selected_ids)
    graph = nx.DiGraph()
    graph.add_nodes_from(selected_ids)
    precedence = list(capability_graph["precedence"])
    for left, right in precedence:
        if left in selected and right in selected:
            # Add only transitive-reduction-compatible precedence. If sensitivity is
            # selected between scenarios and risk, avoid a direct scenarios->risk edge.
            if left == "scenarios" and right == "risk_simulation" and "sensitivity" in selected:
                continue
            graph.add_edge(left, right)
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicPlanningError("selected dynamic plan contains a cycle")
    ordered = list(nx.topological_sort(graph))
    if len(graph.edges) != max(0, len(ordered) - 1):
        raise DynamicPlanningError("selected dynamic plan must be a single serial chain")
    for index, stage_id in enumerate(ordered):
        expected_in = 0 if index == 0 else 1
        expected_out = 0 if index == len(ordered) - 1 else 1
        if graph.in_degree(stage_id) != expected_in or graph.out_degree(stage_id) != expected_out:
            raise DynamicPlanningError("dynamic branching or disconnected stages are forbidden")
    return ordered


def plan_dynamic_pipeline(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    capability_graph = _load_capability_graph(policy)
    operation = str(ticket.get("operation") or "")
    if operation != "scenario_compare":
        raise DynamicPlanningError(
            "experimental optimized planner currently supports scenario_compare entry tickets only"
        )

    scenario_count, varied_variable_count = _scenario_features(ticket)
    uncertainty_count = _uncertainty_count(ticket)
    probabilistic = _probabilistic_claim(ticket)
    decision_class = _decision_class(ticket)
    solution = _solve_stage_selection(
        policy=policy,
        scenario_count=scenario_count,
        varied_variable_count=varied_variable_count,
        uncertainty_count=uncertainty_count,
        probabilistic=probabilistic,
        decision_class=decision_class,
    )

    selected_ids = ["scenarios"]
    if solution["sensitivity"]:
        selected_ids.append("sensitivity")
    if solution["monte_carlo"]:
        selected_ids.append("risk_simulation")
    ordered = _build_selected_graph(selected_ids, capability_graph)
    if len(ordered) > int(policy["maximum_stages"]):
        raise DynamicPlanningError("dynamic plan exceeds maximum stages")

    nodes = capability_graph["nodes"]
    stage_map: dict[str, dict[str, Any]] = {}
    for stage_id in ordered:
        node = nodes[stage_id]
        stage = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "adapter": str(node["adapter"]),
            "depends_on": [] if stage_id == ordered[0] else [ordered[ordered.index(stage_id) - 1]],
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
            "OR-Tools CP-SAT selected the optimal feasible optional-stage subset under "
            f"the explicit policy objective; status={solution['solver_status']}, objective={solution['objective_value']}"
        ),
    ]
    if solution["sensitivity"]:
        reasons.append(
            "sensitivity_analysis selected with utility="
            f"{solution['utility']['sensitivity_analysis']}"
        )
    if solution["monte_carlo"]:
        reasons.append(
            "monte_carlo selected with utility="
            f"{solution['utility']['monte_carlo']}"
        )

    return {
        "id": "dynamic-auto-v2",
        "maturity": "experimental",
        "planning_mode": "structured-signal-optimized-dynamic",
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
        },
        "optimization": solution,
        "planning_reasons": reasons,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def execute_dynamic_pipeline(
    ticket: Mapping[str, Any],
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    plan = plan_dynamic_pipeline(ticket)
    contracts = load_contracts()
    initial_inputs = ticket.get("inputs")
    if not isinstance(initial_inputs, Mapping):
        raise DynamicPlanningError("ticket inputs must be an object")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []

    for stage_id in plan["stage_order"]:
        stage = plan["stage_map"][stage_id]
        operation = str(stage["operation"])
        if operation not in operations:
            raise DynamicPlanningError(f"handler unavailable: {operation}")
        try:
            stage_inputs = ADAPTERS[str(stage["adapter"])](initial_inputs, stage_results, stage)
        except PipelineAdapterError as exc:
            raise DynamicPlanningError(f"adapter failed at {stage_id}: {exc}") from exc
        derived_ticket = dict(ticket)
        derived_ticket["operation"] = operation
        derived_ticket["inputs"] = stage_inputs
        validate_operation_inputs(derived_ticket)
        result = operations[operation](stage_inputs)
        if not isinstance(result, Mapping):
            raise DynamicPlanningError(f"stage returned non-object result: {stage_id}")
        result_dict = dict(result)
        try:
            _validate_output(operation, result_dict, contracts)
        except PipelineEngineError as exc:
            raise DynamicPlanningError(str(exc)) from exc
        stage_results[stage_id] = result_dict
        receipts.append({
            "stage_id": stage_id,
            "operation": operation,
            "adapter": str(stage["adapter"]),
            "status": "PASS",
        })

    return {
        "status": "PASS",
        "planner": "structured-signal-optimized-dynamic",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "stage_order": plan["stage_order"],
        "planning_features": plan["planning_features"],
        "optimization": plan["optimization"],
        "planning_reasons": plan["planning_reasons"],
        "receipts": receipts,
        "final_result": stage_results[plan["result_stage"]],
        "network_used": False,
        "model_calls": 0,
        "automatic_parallel_execution": False,
    }


if __name__ == "__main__":
    print(json.dumps(_load_policy(), ensure_ascii=False, indent=2))
