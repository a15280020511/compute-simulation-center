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

from drift_registry import drift_requirements

DYNAMIC_PIPELINE_ID = "dynamic-auto-v1"
DYNAMIC_STAGE_ID = "dynamic"

FAMILY_BY_OPERATION = {
    "scenario_compare": "scenario-decision",
    "time_series_forecast": "time-series",
    "causal_policy_evaluation": "causal-policy",
    "bayesian_network_inference": "bayesian-network",
    "descriptive_statistics": "reliability",
    "system_dynamics_simulation": "system-dynamics",
}
FAMILY_BY_OPERATION_MODE = {
    ("finance_decision_analysis", "indirect_intelligence_analysis"): "indirect-intelligence",
    ("finance_decision_analysis", "bounded_linear_kalman_filter"): "state-estimation",
    ("finance_decision_analysis", "mixed_integer_optimization"): "optimization",
    ("finance_decision_analysis", "open_spiel_policy_evaluation"): "game-theory",
    ("finance_decision_analysis", "evidently_data_drift"): "drift",
}
INDIRECT_INTELLIGENCE_REQUIREMENTS = [
    "requirements-ortools.txt",
    "requirements-intelligence-rapidfuzz.txt",
    "requirements-intelligence-datasketch.txt",
    "requirements-intelligence-splink.txt",
    "requirements-graph-rdflib.txt",
    "requirements-graph-owlready2.txt",
    "requirements-graph-pyshacl.txt",
    "requirements-graph-igraph.txt",
    "requirements-global-pm4py.txt",
    "requirements-bayesian-network.txt",
    "requirements-intelligence-problog.txt",
]
SYSTEM_DYNAMICS_MODES = {
    "stock_flow",
    "feedback_delay",
    "policy_switch",
    "coupled_capacity",
    "resource_depletion",
    "adoption_saturation",
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


def _matrix_dimensions(value: Any, name: str) -> tuple[int, int]:
    rows = _sequence(value, name)
    if not 20 <= len(rows) <= 5000:
        raise DynamicFamilyRoutingError(f"{name} must contain 20 to 5000 rows")
    width: int | None = None
    for index, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"{name}[{index}]")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise DynamicFamilyRoutingError(f"{name} must be rectangular")
    if width is None or not 2 <= width <= 30:
        raise DynamicFamilyRoutingError(f"{name} must contain 2 to 30 columns")
    return len(rows), width


def resolve_dynamic_family(ticket: Mapping[str, Any]) -> str:
    if not is_dynamic_request(ticket):
        raise DynamicFamilyRoutingError("ticket does not request the dynamic production contract")
    operation = str(ticket.get("operation") or "")
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicFamilyRoutingError("dynamic ticket inputs must be an object")
    mode = str(inputs.get("mode") or "")
    family = FAMILY_BY_OPERATION_MODE.get((operation, mode)) or FAMILY_BY_OPERATION.get(operation)
    if family is None:
        raise DynamicFamilyRoutingError(
            f"dynamic operation/mode is not admitted to any capability family: "
            f"{operation or '<empty>'}/{mode or '<none>'}"
        )

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

    if family == "indirect-intelligence":
        if operation != "finance_decision_analysis" or mode != "indirect_intelligence_analysis":
            raise DynamicFamilyRoutingError("indirect-intelligence family requires its exact operation/mode pair")
        hypothesis = str(inputs.get("hypothesis") or "").strip()
        if not hypothesis:
            raise DynamicFamilyRoutingError("indirect-intelligence family requires a non-empty hypothesis")
        evidence = _sequence(inputs.get("evidence"), "inputs.evidence")
        if not evidence:
            raise DynamicFamilyRoutingError("indirect-intelligence family requires non-empty structured evidence")
        return family

    if family == "state-estimation":
        if operation != "finance_decision_analysis" or mode != "bounded_linear_kalman_filter":
            raise DynamicFamilyRoutingError("state-estimation family requires its exact operation/mode pair")
        for name in (
            "transition_matrix",
            "observation_matrix",
            "process_covariance",
            "observation_covariance",
            "initial_covariance",
            "initial_state",
            "observations",
        ):
            value = inputs.get(name)
            if not _sequence(value, f"inputs.{name}"):
                raise DynamicFamilyRoutingError(f"state-estimation family requires non-empty {name}")
        return family

    if family == "reliability":
        if operation != "descriptive_statistics":
            raise DynamicFamilyRoutingError("reliability family requires descriptive_statistics as the exact entry operation")
        data = _sequence(inputs.get("data"), "inputs.data")
        if len(data) < 2:
            raise DynamicFamilyRoutingError("reliability family requires at least two sample observations")
        context = inputs.get("reliability_context")
        if not isinstance(context, Mapping):
            raise DynamicFamilyRoutingError("reliability family requires structured reliability_context")
        if "threshold" not in context:
            raise DynamicFamilyRoutingError("reliability_context.threshold is required")
        tail = str(context.get("tail") or "lower").lower()
        if tail not in {"lower", "upper"}:
            raise DynamicFamilyRoutingError("reliability_context.tail must be lower or upper")
        return family

    if family == "optimization":
        if operation != "finance_decision_analysis" or mode != "mixed_integer_optimization":
            raise DynamicFamilyRoutingError("optimization family requires its exact operation/mode pair")
        variables = _sequence(inputs.get("variables"), "inputs.variables")
        if not 1 <= len(variables) <= 200:
            raise DynamicFamilyRoutingError("optimization family requires 1 to 200 variables")
        constraints = _sequence(inputs.get("constraints", []), "inputs.constraints")
        if len(constraints) > 1000:
            raise DynamicFamilyRoutingError("optimization family admits at most 1000 constraints")
        return family

    if family == "system-dynamics":
        if operation != "system_dynamics_simulation" or mode not in SYSTEM_DYNAMICS_MODES:
            raise DynamicFamilyRoutingError("system-dynamics family requires system_dynamics_simulation and an admitted fixed mode")
        steps = inputs.get("steps", 100)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10_000:
            raise DynamicFamilyRoutingError("system-dynamics family requires steps from 1 to 10000")
        return family

    if family == "game-theory":
        if operation != "finance_decision_analysis" or mode != "open_spiel_policy_evaluation":
            raise DynamicFamilyRoutingError("game-theory family requires finance_decision_analysis:open_spiel_policy_evaluation")
        game_id = str(inputs.get("game_id") or "matrix_rps")
        if game_id not in {"matrix_rps", "matrix_pd"}:
            raise DynamicFamilyRoutingError("game-theory family admits only matrix_rps and matrix_pd")
        return family

    if family == "drift":
        if operation != "finance_decision_analysis" or mode != "evidently_data_drift":
            raise DynamicFamilyRoutingError("drift family requires finance_decision_analysis:evidently_data_drift")
        _, reference_columns = _matrix_dimensions(inputs.get("reference"), "inputs.reference")
        _, current_columns = _matrix_dimensions(inputs.get("current"), "inputs.current")
        if reference_columns != current_columns:
            raise DynamicFamilyRoutingError("drift reference/current column counts must match")
        context = inputs.get("drift_context")
        if context is not None and not isinstance(context, Mapping):
            raise DynamicFamilyRoutingError("drift_context must be an object when supplied")
        return family

    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")


def family_runtime_metadata(ticket: Mapping[str, Any]) -> dict[str, Any]:
    family = resolve_dynamic_family(ticket)
    if family == "scenario-decision":
        return {"family": family, "entry_contract": "scenario_compare", "policy_file": "dynamic-orchestration-policy.json", "graph_file": "dynamic-capability-graph.json"}
    if family == "time-series":
        return {"family": family, "entry_contract": "time_series_forecast", "policy_file": "dynamic-time-series-policy.json", "graph_file": "dynamic-time-series-capability-graph.json"}
    if family == "causal-policy":
        return {"family": family, "entry_contract": "causal_policy_evaluation", "policy_file": "dynamic-causal-policy.json", "graph_file": "dynamic-causal-capability-graph.json", "python_version": "3.13", "requirements": ["requirements-causal.txt"]}
    if family == "bayesian-network":
        return {"family": family, "entry_contract": "bayesian_network_inference", "policy_file": "dynamic-bayesian-policy.json", "graph_file": "dynamic-bayesian-capability-graph.json", "python_version": "3.12", "requirements": ["requirements-bayesian-network.txt"]}
    if family == "indirect-intelligence":
        return {"family": family, "entry_contract": "finance_decision_analysis:indirect_intelligence_analysis", "policy_file": "indirect-intelligence-mode-registry.json", "graph_file": "dynamic-indirect-intelligence-capability-graph.json", "python_version": "3.12", "requirements": list(INDIRECT_INTELLIGENCE_REQUIREMENTS)}
    if family == "state-estimation":
        return {"family": family, "entry_contract": "finance_decision_analysis:bounded_linear_kalman_filter", "policy_file": "dynamic-state-estimation-policy.json", "graph_file": "dynamic-state-estimation-capability-graph.json", "python_version": "3.12", "requirements": []}
    if family == "reliability":
        return {"family": family, "entry_contract": "descriptive_statistics:sample-normal-reliability", "policy_file": "dynamic-reliability-policy.json", "graph_file": "dynamic-reliability-capability-graph.json", "python_version": "3.12", "requirements": ["requirements-global-openturns.txt"]}
    if family == "optimization":
        return {"family": family, "entry_contract": "finance_decision_analysis:mixed_integer_optimization", "policy_file": "dynamic-optimization-policy.json", "graph_file": "dynamic-optimization-capability-graph.json", "python_version": "3.12", "requirements": ["requirements-ortools.txt", "requirements-thinktank-decision.txt"]}
    if family == "system-dynamics":
        return {"family": family, "entry_contract": "system_dynamics_simulation:<fixed-mode>", "policy_file": "dynamic-system-dynamics-policy.json", "graph_file": "dynamic-system-dynamics-capability-graph.json", "python_version": "3.12", "requirements": ["requirements-ortools.txt"]}
    if family == "game-theory":
        return {"family": family, "entry_contract": "finance_decision_analysis:open_spiel_policy_evaluation", "policy_file": "dynamic-game-theory-policy.json", "graph_file": "dynamic-game-theory-capability-graph.json", "python_version": "3.12", "requirements": ["requirements-ortools.txt", "requirements-strategy-open-spiel.txt", "requirements-strategy-pygambit.txt"]}
    if family == "drift":
        requirements = ["requirements-ortools.txt", *drift_requirements(), "requirements-thinktank-econometrics.txt"]
        return {"family": family, "entry_contract": "finance_decision_analysis:evidently_data_drift", "policy_file": "dynamic-drift-policy.json", "graph_file": "dynamic-drift-capability-graph.json", "python_version": "3.12", "requirements": requirements}
    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")


def run_dynamic_family_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
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
    if family == "indirect-intelligence":
        from dynamic_indirect_intelligence_planner import run_dynamic_indirect_intelligence_ticket
        return run_dynamic_indirect_intelligence_ticket(ticket, output_dir, operations)
    if family == "state-estimation":
        from dynamic_state_estimation_planner import run_dynamic_state_estimation_ticket
        return run_dynamic_state_estimation_ticket(ticket, output_dir, operations)
    if family == "reliability":
        from dynamic_reliability_planner import run_dynamic_reliability_ticket
        return run_dynamic_reliability_ticket(ticket, output_dir, operations)
    if family == "optimization":
        from dynamic_optimization_planner import run_dynamic_optimization_ticket
        return run_dynamic_optimization_ticket(ticket, output_dir, operations)
    if family == "system-dynamics":
        from dynamic_system_dynamics_planner import run_dynamic_system_dynamics_ticket
        return run_dynamic_system_dynamics_ticket(ticket, output_dir, operations)
    if family == "game-theory":
        from dynamic_game_theory_planner import run_dynamic_game_theory_ticket
        return run_dynamic_game_theory_ticket(ticket, output_dir, operations)
    if family == "drift":
        from dynamic_drift_planner import run_dynamic_drift_ticket
        return run_dynamic_drift_ticket(ticket, output_dir, operations)
    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")
