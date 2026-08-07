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
    "causal_screening": "causal-did",
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


def _causal_did_metadata(inputs: Mapping[str, Any]) -> dict[str, Any]:
    names = ("treated_pre", "treated_post", "control_pre", "control_post")
    arrays = {name: _sequence(inputs.get(name), f"inputs.{name}") for name in names}
    if any(len(values) < 3 for values in arrays.values()):
        raise DynamicFamilyRoutingError("causal-did family requires at least three observations in every DID window")
    context = inputs.get("dynamic_context")
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise DynamicFamilyRoutingError("inputs.dynamic_context must be an object")
    design = str(context.get("causal_design") or "")
    advanced = context.get("allow_causal_policy_evaluation") is True
    if design and design != "difference_in_differences":
        raise DynamicFamilyRoutingError(
            "causal_screening dynamic family admits only causal_design=difference_in_differences"
        )
    if advanced and design != "difference_in_differences":
        raise DynamicFamilyRoutingError(
            "advanced causal DID evaluation requires causal_design=difference_in_differences"
        )
    aligned = len({len(values) for values in arrays.values()}) == 1
    if advanced and not aligned:
        raise DynamicFamilyRoutingError(
            "advanced causal DID evaluation requires equal-length treated/control pre/post windows"
        )
    return {
        "advanced_requested": advanced,
        "causal_design": design or None,
        "aligned_windows": aligned,
        "window_lengths": {name: len(values) for name, values in arrays.items()},
    }


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

    if family == "causal-did":
        _causal_did_metadata(inputs)
        return family

    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")


def family_runtime_metadata(ticket: Mapping[str, Any]) -> dict[str, Any]:
    family = resolve_dynamic_family(ticket)
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicFamilyRoutingError("dynamic ticket inputs must be an object")
    if family == "scenario-decision":
        return {
            "family": family,
            "entry_contract": "scenario_compare",
            "policy_file": "dynamic-orchestration-policy.json",
            "graph_file": "dynamic-capability-graph.json",
            "extra_requirements": [],
        }
    if family == "time-series":
        return {
            "family": family,
            "entry_contract": "time_series_forecast",
            "policy_file": "dynamic-time-series-policy.json",
            "graph_file": "dynamic-time-series-capability-graph.json",
            "extra_requirements": [],
        }
    if family == "causal-did":
        metadata = _causal_did_metadata(inputs)
        return {
            "family": family,
            "entry_contract": "causal_screening",
            "policy_file": "dynamic-causal-did-policy.json",
            "graph_file": "dynamic-causal-did-capability-graph.json",
            "extra_requirements": ["requirements-causal.txt"] if metadata["advanced_requested"] else [],
            "causal_design": metadata["causal_design"],
            "advanced_requested": metadata["advanced_requested"],
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
    if family == "causal-did":
        from dynamic_causal_did_planner import run_dynamic_causal_did_ticket

        return run_dynamic_causal_did_ticket(ticket, output_dir, operations)
    raise DynamicFamilyRoutingError(f"unsupported dynamic family: {family}")
