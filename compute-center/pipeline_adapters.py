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


def scenario_ranking_to_descriptive_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del initial_inputs, stage
    scenario_result = _mapping(stage_results.get("scenarios"), "stage results.scenarios")
    ranking = _sequence(scenario_result.get("ranking"), "stage results.scenarios.ranking")
    scores = [
        _finite(_mapping(row, f"ranking[{index}]").get("score"), f"ranking[{index}].score")
        for index, row in enumerate(ranking)
    ]
    if not scores:
        raise PipelineAdapterError("scenario ranking is empty")
    return {"data": scores}


def scenario_ranking_to_sensitivity(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    model, ranking, best_values = _scenario_context(initial_inputs, stage_results)
    coefficients = _mapping(model.get("coefficients"), "model.coefficients")
    intercept = _finite(model.get("intercept", 0.0), "model.intercept")
    reduced_coefficients: dict[str, float] = {}
    variables: list[dict[str, Any]] = []
    for name, raw_coefficient in coefficients.items():
        coefficient = _finite(raw_coefficient, f"model.coefficients[{name}]")
        values = [
            _finite(_mapping(row.get("values"), f"scenario[{index}].values").get(name), f"scenario[{index}].values[{name}]")
            for index, row in enumerate(ranking)
        ]
        low = min(values)
        high = max(values)
        base = _finite(best_values.get(name), f"best scenario value[{name}]")
        if low == high:
            intercept += coefficient * low
            continue
        reduced_coefficients[str(name)] = coefficient
        variables.append({"name": str(name), "low": low, "base": base, "high": high})
    if not variables:
        raise PipelineAdapterError("scenario-derived sensitivity requires at least one varying variable")
    reduced_model = {"intercept": intercept, "coefficients": reduced_coefficients}
    return {"model": reduced_model, "variables": variables}


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


def scenario_ranking_to_constrained_optimization(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    model, ranking, _best_values = _scenario_context(initial_inputs, stage_results)
    context = _mapping(initial_inputs.get("dynamic_context"), "ticket inputs.dynamic_context")
    if context.get("continuous_decision_optimization") is not True:
        raise PipelineAdapterError("continuous decision optimization was not explicitly authorized")
    if context.get("allow_continuous_interpolation") is not True:
        raise PipelineAdapterError("continuous interpolation was not explicitly authorized")
    raw_names = _sequence(context.get("controllable_variables"), "dynamic_context.controllable_variables")
    controllable = [str(item) for item in raw_names]
    coefficients = _mapping(model.get("coefficients"), "model.coefficients")
    model_names = [str(name) for name in coefficients]
    if not controllable or len(controllable) != len(set(controllable)):
        raise PipelineAdapterError("controllable_variables must be non-empty and unique")
    if set(controllable) != set(model_names):
        raise PipelineAdapterError("every model variable must be explicitly declared controllable")

    bounds: list[list[float]] = []
    objective: list[float] = []
    for name in model_names:
        values = [
            _finite(_mapping(row.get("values"), f"scenario[{index}].values").get(name), f"scenario[{index}].values[{name}]")
            for index, row in enumerate(ranking)
        ]
        low = min(values)
        high = max(values)
        bounds.append([low, high])
        objective.append(_finite(coefficients.get(name), f"model.coefficients[{name}]"))
    return {
        "objective": objective,
        "maximize": True,
        "variable_names": model_names,
        "bounds": bounds,
        "A_ub": [],
        "b_ub": [],
    }


ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "ticket_inputs": ticket_inputs,
    "scenario_ranking_to_descriptive_statistics": scenario_ranking_to_descriptive_statistics,
    "scenario_ranking_to_sensitivity": scenario_ranking_to_sensitivity,
    "scenario_ranking_to_monte_carlo": scenario_ranking_to_monte_carlo,
    "scenario_ranking_to_constrained_optimization": scenario_ranking_to_constrained_optimization,
}
