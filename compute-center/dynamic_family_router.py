#!/usr/bin/env python3
"""Fail-closed router for repository-controlled dynamic capability families.

Routing is based only on the explicit operation and structured input shape. Free-form
objective text is never inspected. The router itself does not import OR-Tools or any
family planner until execution is requested.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

DYNAMIC_PIPELINE_ID = "dynamic-auto-v1"
DYNAMIC_STAGE_ID = "dynamic"

FAMILY_BY_OPERATION = {
    "scenario_compare": "scenario-decision",
    "time_series_forecast": "time-series",
    "causal_policy_evaluation": "causal-policy",
    "bayesian_network_inference": "bayesian-network",
}


class DynamicFamilyRoutingError(ValueError):
    """Raised when a dynamic request does not map to one unambiguous family."""


def is_dynamic_request(ticket: Mapping[str, Any]) -> bool:
    pipeline = ticket.get("pipeline")
    return bool(
        isinstance(pipeline, Mapping)
        and str(pipeline.get("pipeline_id") or "") == DYNAMIC_PIPELINE_ID
        and str(pipeline.get("stage_id") or "") == DYNAMIC_STAGE_ID
    )


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicFamilyRoutingError(f"{name} must be an array")
    return value


def resolve_dynamic_family(ticket: Mapping[str, Any]) -> str:
    if not is_dynamic_request(ticket):
        raise DynamicFamilyRoutingError("ticket does not request the dynamic production contract")
    operation = str(ticket.get("operation") or "")
    family = FAMILY_BY_OPERATION.get(operation)
    if family is None:
        raise DynamicFamilyRoutingError(
            f"dynamic operation is not admitted to any capability family: {operation or '<empty>'}"
        )
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicFamilyRoutingError("dynamic ticket inputs must be an object")

    if family == "scenario-decision":
        scenarios = _sequence(inputs.get("scenarios"), "inputs.scenarios")
        model = inputs.get("model")
        if not scenarios or not isinstance(model, Mapping):
            raise DynamicFamilyRoutingError("scenario-decision family requires model and non-empty scenarios")
        return family

    if family == "time-series":
        data = _sequence(inputs.get("data"), "inputs.data")
        if len(data) < 5:
            raise DynamicFamilyRoutingError("time-series family requires at least five observations")
        return family

    if family == "causal-policy":
        mode = str(inputs.get("mode") or "")
        admitted_modes = {"backdoor_adjustment", "propensity_weighting"}
        if mode not in admitted_modes:
            raise DynamicFamilyRoutingError(
                "causal-policy dynamic family currently admits only "
                "backdoor_adjustment or propensity_weighting"
            )
        treatment = _sequence(inputs.get("treatment"), "inputs.treatment")
        outcome = _sequence(inputs.get("outcome"), "inputs.outcome")
        if len(treatment) < 8 or len(treatment) != len(outcome):
            raise DynamicFamilyRoutingError(
                "causal-policy family requires equal treatment/outcome arrays with at least eight observations"
            )
        confounders = inputs.get("confounders")
        if not isinstance(confounders, Mapping) or not confounders:
            raise DynamicFamilyRoutingError("causal-policy family requires at least one declared confounder")
        return family

    if family == "bayesian-network":
        mode = str(inputs.get("mode") or "")
        if mode != "bayesian_parameter_estimation":
            raise DynamicFamilyRoutingError(
                "bayesian-network dynamic family currently admits only bayesian_parameter_estimation as the entry mode"
            )
        data = inputs.get("data")
        if not isinstance(data, Mapping) or not data:
            raise DynamicFamilyRoutingError("bayesian-network family requires a non-empty data object")
        data_nodes = {str(name) for name in data}
        if any(not name for name in data_nodes):
            raise DynamicFamilyRoutingError("bayesian-network data variable names must be non-empty")
        query_variables = _sequence(inputs.get("query_variables"), "inputs.query_variables")
        if not query_variables:
            raise DynamicFamilyRoutingError("bayesian-network family requires non-empty query_variables")
        if any(str(item) not in data_nodes for item in query_variables):
            raise DynamicFamilyRoutingError("bayesian-network query_variables must have observed data columns")
        edges = _sequence(inputs.get("edges", []), "inputs.edges")
        for index, raw_edge in enumerate(edges):
            edge = _sequence(raw_edge, f"inputs.edges[{index}]")
            if len(edge) != 2:
                raise DynamicFamilyRoutingError("bayesian-network edges must contain source and target")
            left, right = str(edge[0]), str(edge[1])
            if left not in data_nodes or right not in data_nodes:
                raise DynamicFamilyRoutingError(
                    "bayesian-network dependency nodes must have observed data columns for parameter estimation"
                )
        return family

    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")


def family_runtime_metadata(ticket: Mapping[str, Any]) -> dict[str, Any]:
    family = resolve_dynamic_family(ticket)
    if family == "scenario-decision":
        return {
            "family": family,
            "entry_contract": "scenario_compare",
            "policy_file": "dynamic-orchestration-policy.json",
            "graph_file": "dynamic-capability-graph.json",
        }
    if family == "time-series":
        return {
            "family": family,
            "entry_contract": "time_series_forecast",
            "policy_file": "dynamic-time-series-policy.json",
            "graph_file": "dynamic-time-series-capability-graph.json",
        }
    if family == "causal-policy":
        return {
            "family": family,
            "entry_contract": "causal_policy_evaluation",
            "policy_file": "dynamic-causal-policy.json",
            "graph_file": "dynamic-causal-capability-graph.json",
            "python_version": "3.13",
            "requirements": ["requirements-causal.txt"],
        }
    if family == "bayesian-network":
        return {
            "family": family,
            "entry_contract": "bayesian_network_inference",
            "policy_file": "dynamic-bayesian-policy.json",
            "graph_file": "dynamic-bayesian-capability-graph.json",
            "python_version": "3.12",
            "requirements": ["requirements-bayesian-network.txt"],
        }
    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")


def run_dynamic_family_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    family = resolve_dynamic_family(ticket)
    if family == "scenario-decision":
        from dynamic_pipeline_planner import run_dynamic_pipeline_ticket

        return run_dynamic_pipeline_ticket(ticket, output_dir, operations)
    if family == "time-series":
        from dynamic_time_series_planner import run_dynamic_time_series_ticket

        return run_dynamic_time_series_ticket(ticket, output_dir, operations)
    if family == "causal-policy":
        from dynamic_causal_policy_planner import run_dynamic_causal_policy_ticket

        return run_dynamic_causal_policy_ticket(ticket, output_dir, operations)
    if family == "bayesian-network":
        from dynamic_bayesian_network_planner import run_dynamic_bayesian_network_ticket

        return run_dynamic_bayesian_network_ticket(ticket, output_dir, operations)
    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")
