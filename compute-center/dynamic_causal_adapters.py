#!/usr/bin/env python3
"""Fixed deterministic adapters for the causal-policy dynamic capability family."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable

from pipeline_adapters import ADAPTERS, PipelineAdapterError


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _base_inputs(initial_inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(initial_inputs, Mapping):
        raise PipelineAdapterError("causal initial inputs must be an object")
    result = _clone(dict(initial_inputs))
    result.pop("dynamic_context", None)
    return result


def causal_outcome_to_descriptive_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    outcome = initial_inputs.get("outcome")
    if not isinstance(outcome, list) or not outcome:
        raise PipelineAdapterError("causal outcome must be a non-empty array")
    return {"data": _clone(outcome)}


def causal_to_alternate_estimate(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    result = _base_inputs(initial_inputs)
    primary = str(result.get("mode") or "")
    if primary == "backdoor_adjustment":
        result["mode"] = "propensity_weighting"
    elif primary == "propensity_weighting":
        result["mode"] = "backdoor_adjustment"
    else:
        raise PipelineAdapterError("causal alternate estimator requires backdoor or propensity primary mode")
    return result


def causal_to_placebo_refutation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    result = _base_inputs(initial_inputs)
    context = initial_inputs.get("dynamic_context")
    result["mode"] = "placebo_policy_test"
    if isinstance(context, Mapping):
        mapping = {
            "placebo_repetitions": "repetitions",
            "placebo_seed": "seed",
            "placebo_alpha": "alpha",
        }
        for source, target in mapping.items():
            if source in context:
                result[target] = _clone(context[source])
    return result


def causal_to_primary_estimate(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    result = _base_inputs(initial_inputs)
    if str(result.get("mode") or "") not in {"backdoor_adjustment", "propensity_weighting"}:
        raise PipelineAdapterError("causal primary estimator mode is not admitted")
    return result


CAUSAL_ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "causal_outcome_to_descriptive_statistics": causal_outcome_to_descriptive_statistics,
    "causal_to_alternate_estimate": causal_to_alternate_estimate,
    "causal_to_placebo_refutation": causal_to_placebo_refutation,
    "causal_to_primary_estimate": causal_to_primary_estimate,
}


def install_causal_adapters() -> None:
    """Install the fixed family adapter set into the shared allowlist exactly once."""
    for name, handler in CAUSAL_ADAPTERS.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"conflicting pipeline adapter registration: {name}")
        ADAPTERS[name] = handler
