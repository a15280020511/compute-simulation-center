#!/usr/bin/env python3
"""Deterministic adapters for the Bayesian-network dynamic capability family."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from pipeline_adapters import ADAPTERS, PipelineAdapterError


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineAdapterError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineAdapterError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PipelineAdapterError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _network_nodes(initial_inputs: Mapping[str, Any]) -> list[str]:
    data = _mapping(initial_inputs.get("data"), "ticket inputs.data")
    nodes = {str(name) for name in data}
    for index, raw_edge in enumerate(_sequence(initial_inputs.get("edges", []), "ticket inputs.edges")):
        edge = _sequence(raw_edge, f"ticket inputs.edges[{index}]")
        if len(edge) != 2:
            raise PipelineAdapterError("Bayesian edge must contain source and target")
        nodes.update((str(edge[0]), str(edge[1])))
    if not nodes:
        raise PipelineAdapterError("Bayesian dynamic family requires network nodes")
    return sorted(nodes)


def _estimated_cpds(stage_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    estimation = _mapping(stage_results.get("parameter_estimation"), "stage results.parameter_estimation")
    if estimation.get("mode") != "bayesian_parameter_estimation":
        raise PipelineAdapterError("parameter_estimation stage returned the wrong Bayesian mode")
    if estimation.get("model_valid") is not True:
        raise PipelineAdapterError("parameter_estimation stage did not return a valid model")
    rows = _sequence(estimation.get("cpds"), "stage results.parameter_estimation.cpds")
    if not rows:
        raise PipelineAdapterError("parameter_estimation stage returned no CPDs")
    converted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"stage results.parameter_estimation.cpds[{index}]")
        variable = str(row.get("variable") or "").strip()
        if not variable or variable in seen:
            raise PipelineAdapterError("estimated CPDs require unique non-empty variables")
        seen.add(variable)
        variable_card = _integer(row.get("variable_card"), f"cpd[{variable}].variable_card", 2, 20)
        evidence = [str(item) for item in _sequence(row.get("evidence", []), f"cpd[{variable}].evidence")]
        cardinality = [
            _integer(item, f"cpd[{variable}].cardinality", 2, 20)
            for item in _sequence(row.get("cardinality"), f"cpd[{variable}].cardinality")
        ]
        if len(cardinality) != 1 + len(evidence) or cardinality[0] != variable_card:
            raise PipelineAdapterError(f"estimated CPD cardinality mismatch for {variable}")
        values = _clone(_sequence(row.get("values"), f"cpd[{variable}].values"))
        state_names_raw = row.get("state_names")
        state_names = _clone(dict(state_names_raw)) if isinstance(state_names_raw, Mapping) else {}
        converted.append(
            {
                "variable": variable,
                "variable_card": variable_card,
                "values": values,
                "evidence": evidence,
                "evidence_card": cardinality[1:],
                "state_names": state_names,
            }
        )
    return converted


def _inference_base(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Any]) -> dict[str, Any]:
    query_variables = [
        str(item) for item in _sequence(initial_inputs.get("query_variables"), "ticket inputs.query_variables")
    ]
    if not query_variables or len(query_variables) != len(set(query_variables)):
        raise PipelineAdapterError("query_variables must be non-empty and unique")
    result: dict[str, Any] = {
        "nodes": _network_nodes(initial_inputs),
        "edges": _clone(_sequence(initial_inputs.get("edges", []), "ticket inputs.edges")),
        "cpds": _estimated_cpds(stage_results),
        "query_variables": query_variables,
    }
    evidence = initial_inputs.get("evidence")
    if evidence is not None:
        result["evidence"] = _clone(dict(_mapping(evidence, "ticket inputs.evidence")))
    return result


def bayesian_ticket_to_parameter_estimation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "bayesian_parameter_estimation":
        raise PipelineAdapterError("Bayesian dynamic family entry mode must be bayesian_parameter_estimation")
    result: dict[str, Any] = {
        "mode": "bayesian_parameter_estimation",
        "edges": _clone(_sequence(initial_inputs.get("edges", []), "ticket inputs.edges")),
        "data": _clone(dict(_mapping(initial_inputs.get("data"), "ticket inputs.data"))),
    }
    if "equivalent_sample_size" in initial_inputs:
        result["equivalent_sample_size"] = _clone(initial_inputs["equivalent_sample_size"])
    return result


def bayesian_estimate_to_fixed_inference(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    result = _inference_base(initial_inputs, stage_results)
    result["mode"] = "fixed_network_inference"
    return result


def bayesian_estimate_to_evidence_sensitivity(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    result = _inference_base(initial_inputs, stage_results)
    scenarios = _sequence(initial_inputs.get("evidence_scenarios"), "ticket inputs.evidence_scenarios")
    if not scenarios:
        raise PipelineAdapterError("evidence_sensitivity stage requires evidence_scenarios")
    result["mode"] = "evidence_sensitivity"
    result["evidence_scenarios"] = _clone(scenarios)
    return result


def bayesian_estimate_to_virtual_evidence(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    result = _inference_base(initial_inputs, stage_results)
    rows = _sequence(initial_inputs.get("virtual_evidence"), "ticket inputs.virtual_evidence")
    if not rows:
        raise PipelineAdapterError("virtual_evidence_update stage requires virtual_evidence")
    result["mode"] = "virtual_evidence_update"
    result["virtual_evidence"] = _clone(rows)
    return result


BAYESIAN_ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "bayesian_ticket_to_parameter_estimation": bayesian_ticket_to_parameter_estimation,
    "bayesian_estimate_to_fixed_inference": bayesian_estimate_to_fixed_inference,
    "bayesian_estimate_to_evidence_sensitivity": bayesian_estimate_to_evidence_sensitivity,
    "bayesian_estimate_to_virtual_evidence": bayesian_estimate_to_virtual_evidence,
}


def install_bayesian_adapters() -> None:
    """Register the fixed Bayesian adapter set exactly once."""
    for name, handler in BAYESIAN_ADAPTERS.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"conflicting pipeline adapter registration: {name}")
        ADAPTERS[name] = handler
