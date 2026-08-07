#!/usr/bin/env python3
"""Experimental structured-signal dynamic orchestration for the compute center.

This planner does not interpret free-form objective text. It composes only allowlisted
operations from existing structured ticket fields, validates the resulting NetworkX DAG,
and executes stages strictly serially with fixed adapters and output contracts.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import networkx as nx

from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError
from pipeline_engine import PipelineEngineError, _validate_output, load_contracts

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-orchestration-policy.json"


class DynamicPlanningError(ValueError):
    """Raised when a dynamic plan cannot be generated or executed safely."""


def _load_policy() -> dict[str, Any]:
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "compute-dynamic-orchestration-policy-v1":
        raise DynamicPlanningError("invalid dynamic orchestration policy schema")
    expected = {
        "status": "experimental",
        "engine": "networkx",
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


def plan_dynamic_pipeline(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    operation = str(ticket.get("operation") or "")
    if operation != "scenario_compare":
        raise DynamicPlanningError(
            "experimental dynamic planner currently supports scenario_compare entry tickets only"
        )

    scenario_count, varied_variable_count = _scenario_features(ticket)
    uncertainty_count = _uncertainty_count(ticket)
    probabilistic = _probabilistic_claim(ticket)
    decision_class = _decision_class(ticket)
    selection = policy["selection_policy"]

    need_sensitivity = (
        scenario_count >= int(selection["sensitivity_min_scenarios"])
        and varied_variable_count > 0
    ) or uncertainty_count >= int(selection["uncertainty_trigger_count"])
    need_monte_carlo = probabilistic or uncertainty_count >= int(selection["uncertainty_trigger_count"])

    stages: list[dict[str, Any]] = [
        {
            "id": "scenarios",
            "operation": "scenario_compare",
            "depends_on": [],
            "adapter": "ticket_inputs",
        }
    ]
    reasons = ["scenario_compare is the declared entry operation"]

    if need_sensitivity:
        stages.append(
            {
                "id": "sensitivity",
                "operation": "sensitivity_analysis",
                "depends_on": [stages[-1]["id"]],
                "adapter": "scenario_ranking_to_sensitivity",
            }
        )
        reasons.append(
            f"sensitivity selected: scenarios={scenario_count}, varied_variables={varied_variable_count}, uncertainty_signals={uncertainty_count}"
        )

    if need_monte_carlo:
        stages.append(
            {
                "id": "risk_simulation",
                "operation": "monte_carlo",
                "depends_on": [stages[-1]["id"]],
                "adapter": "scenario_ranking_to_monte_carlo",
                "fixed_parameters": {
                    "iterations": int(selection["monte_carlo_iterations"][decision_class]),
                    "seed": _deterministic_seed(ticket),
                },
            }
        )
        reasons.append(
            f"monte_carlo selected: probabilistic_claim={str(probabilistic).lower()}, uncertainty_signals={uncertainty_count}, decision_class={decision_class}"
        )

    if len(stages) > int(policy["maximum_stages"]):
        raise DynamicPlanningError("dynamic plan exceeds maximum stages")

    allowed_operations = set(str(item) for item in policy["allowed_operations"])
    allowed_adapters = set(str(item) for item in policy["allowed_adapters"])
    graph = nx.DiGraph()
    stage_map: dict[str, dict[str, Any]] = {}
    for stage in stages:
        if stage["operation"] not in allowed_operations:
            raise DynamicPlanningError(f"operation is not allowlisted: {stage['operation']}")
        if stage["adapter"] not in allowed_adapters or stage["adapter"] not in ADAPTERS:
            raise DynamicPlanningError(f"adapter is not allowlisted: {stage['adapter']}")
        stage_map[stage["id"]] = stage
        graph.add_node(stage["id"])
    for stage in stages:
        for dependency in stage["depends_on"]:
            if dependency not in stage_map:
                raise DynamicPlanningError(f"unknown dependency: {dependency}")
            graph.add_edge(dependency, stage["id"])
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicPlanningError("dynamic plan contains a cycle")
    ordered = list(nx.topological_sort(graph))
    if ordered != [stage["id"] for stage in stages]:
        raise DynamicPlanningError("dynamic plan order is unstable")
    if len(graph.edges) != max(0, len(stages) - 1):
        raise DynamicPlanningError("dynamic plan must remain a single serial chain")
    for index, stage_id in enumerate(ordered):
        expected_in = 0 if index == 0 else 1
        expected_out = 0 if index == len(ordered) - 1 else 1
        if graph.in_degree(stage_id) != expected_in or graph.out_degree(stage_id) != expected_out:
            raise DynamicPlanningError("dynamic branching or disconnected stages are forbidden")

    return {
        "id": "dynamic-auto-v1",
        "maturity": "experimental",
        "planning_mode": "structured-signal-dynamic",
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
        receipts.append(
            {
                "stage_id": stage_id,
                "operation": operation,
                "adapter": str(stage["adapter"]),
                "status": "PASS",
            }
        )

    return {
        "status": "PASS",
        "planner": "structured-signal-dynamic",
        "engine": "networkx",
        "stage_order": plan["stage_order"],
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "receipts": receipts,
        "final_result": stage_results[plan["result_stage"]],
        "network_used": False,
        "model_calls": 0,
        "automatic_parallel_execution": False,
    }


if __name__ == "__main__":
    print(json.dumps(_load_policy(), ensure_ascii=False, indent=2))
