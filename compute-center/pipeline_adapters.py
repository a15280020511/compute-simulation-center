#!/usr/bin/env python3
"""Fixed, allowlisted adapters between compute pipeline stages.

Adapters are deterministic data transformations only. They never execute ticket-supplied
code, access a network, call a model, install packages, or choose operations dynamically.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable


class PipelineAdapterError(ValueError):
    """Raised when a fixed stage-to-stage conversion cannot be performed safely."""


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


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineAdapterError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PipelineAdapterError(f"{name} must be finite")
    return number


def _scenario_context(ticket_inputs: Mapping[str, Any], results: Mapping[str, Any]) -> tuple[dict[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    model = dict(_mapping(ticket_inputs.get("model"), "ticket inputs.model"))
    coefficients = _mapping(model.get("coefficients"), "ticket inputs.model.coefficients")
    if not coefficients:
        raise PipelineAdapterError("model coefficients are required")
    scenario_result = _mapping(results.get("scenarios"), "stage results.scenarios")
    ranking_raw = _sequence(scenario_result.get("ranking"), "stage results.scenarios.ranking")
    ranking: list[Mapping[str, Any]] = []
    for index, row in enumerate(ranking_raw):
        ranking.append(_mapping(row, f"stage results.scenarios.ranking[{index}]"))
    if not ranking:
        raise PipelineAdapterError("scenario ranking is empty")
    best_values = _mapping(ranking[0].get("values"), "best scenario values")
    expected = {str(name) for name in coefficients}
    if set(best_values) != expected:
        raise PipelineAdapterError("best scenario variables must match model coefficients")
    return model, ranking, best_values


def ticket_inputs(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    return _clone(initial_inputs)


def scenario_ranking_to_sensitivity(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    model, ranking, best_values = _scenario_context(initial_inputs, stage_results)
    coefficients = _mapping(model.get("coefficients"), "model.coefficients")
    variables: list[dict[str, Any]] = []
    for name in coefficients:
        values = [
            _finite(_mapping(row.get("values"), f"scenario[{index}].values").get(name), f"scenario[{index}].values[{name}]")
            for index, row in enumerate(ranking)
        ]
        low = min(values)
        high = max(values)
        if low == high:
            raise PipelineAdapterError(
                f"scenario-derived sensitivity requires variation for variable {name}"
            )
        base = _finite(best_values.get(name), f"best scenario value[{name}]")
        variables.append({"name": str(name), "low": low, "base": base, "high": high})
    return {"model": _clone(model), "variables": variables}


def scenario_ranking_to_monte_carlo(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    model, ranking, best_values = _scenario_context(initial_inputs, stage_results)
    fixed = _mapping(stage.get("fixed_parameters", {}), "stage.fixed_parameters")
    iterations = int(fixed.get("iterations", 5000))
    seed = int(fixed.get("seed", 0))
    coefficients = _mapping(model.get("coefficients"), "model.coefficients")
    variables: list[dict[str, Any]] = []
    for name in coefficients:
        values = [
            _finite(_mapping(row.get("values"), f"scenario[{index}].values").get(name), f"scenario[{index}].values[{name}]")
            for index, row in enumerate(ranking)
        ]
        low = min(values)
        high = max(values)
        mode = _finite(best_values.get(name), f"best scenario value[{name}]")
        if low == high:
            variables.append({"name": str(name), "distribution": "constant", "value": low})
        else:
            variables.append(
                {
                    "name": str(name),
                    "distribution": "triangular",
                    "minimum": low,
                    "mode": min(max(mode, low), high),
                    "maximum": high,
                }
            )
    return {
        "model": _clone(model),
        "variables": variables,
        "iterations": iterations,
        "seed": seed,
    }


ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "ticket_inputs": ticket_inputs,
    "scenario_ranking_to_sensitivity": scenario_ranking_to_sensitivity,
    "scenario_ranking_to_monte_carlo": scenario_ranking_to_monte_carlo,
}
