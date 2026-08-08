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


def _time_series_data(initial_inputs: Mapping[str, Any]) -> list[float]:
    data = _sequence(initial_inputs.get("data"), "ticket inputs.data")
    values = [_finite(item, f"ticket inputs.data[{index}]") for index, item in enumerate(data)]
    if len(values) < 5:
        raise PipelineAdapterError("time-series dynamic family requires at least five observations")
    return values


def _causal_vectors(initial_inputs: Mapping[str, Any]) -> tuple[list[float], list[float], dict[str, list[float]]]:
    treatment_raw = _sequence(initial_inputs.get("treatment"), "ticket inputs.treatment")
    outcome_raw = _sequence(initial_inputs.get("outcome"), "ticket inputs.outcome")
    treatment = [_finite(item, f"ticket inputs.treatment[{index}]") for index, item in enumerate(treatment_raw)]
    outcome = [_finite(item, f"ticket inputs.outcome[{index}]") for index, item in enumerate(outcome_raw)]
    if len(treatment) < 8 or len(treatment) != len(outcome):
        raise PipelineAdapterError("causal family requires equal treatment/outcome arrays with at least eight observations")
    if any(item not in {0.0, 1.0} for item in treatment):
        raise PipelineAdapterError("causal family treatment must be binary")
    raw_confounders = initial_inputs.get("confounders", {})
    confounders: dict[str, list[float]] = {}
    if raw_confounders is not None:
        mapped = _mapping(raw_confounders, "ticket inputs.confounders")
        for name in sorted(str(key) for key in mapped):
            values_raw = _sequence(mapped[name], f"ticket inputs.confounders.{name}")
            values = [_finite(item, f"ticket inputs.confounders.{name}[{index}]") for index, item in enumerate(values_raw)]
            if len(values) != len(treatment):
                raise PipelineAdapterError("all causal confounders must match treatment length")
            confounders[name] = values
    return treatment, outcome, confounders


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
    """Stable fixed-pipeline adapter; preserves the original fail-closed contract."""
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


def dynamic_scenario_ranking_to_sensitivity(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Dynamic-only adapter that safely conditions on scenario-constant variables."""
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
    return {
        "model": {"intercept": intercept, "coefficients": reduced_coefficients},
        "variables": variables,
    }


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
        bounds.append([min(values), max(values)])
        objective.append(_finite(coefficients.get(name), f"model.coefficients[{name}]"))
    return {
        "objective": objective,
        "maximize": True,
        "variable_names": model_names,
        "bounds": bounds,
        "A_ub": [],
        "b_ub": [],
    }


def time_series_to_descriptive_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    return {"data": _time_series_data(initial_inputs)}


def time_series_to_pattern_discovery(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    return {"data": _time_series_data(initial_inputs)}


def time_series_to_assumption_validation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    result: dict[str, Any] = {"data": _time_series_data(initial_inputs)}
    for name in (
        "expected_minimum",
        "expected_maximum",
        "expected_mean",
        "mean_tolerance",
        "expected_distribution",
    ):
        if name in initial_inputs:
            result[name] = _clone(initial_inputs[name])
    return result


def time_series_to_forecast(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    result: dict[str, Any] = {"data": _time_series_data(initial_inputs)}
    for name in ("horizon", "holdout"):
        if name in initial_inputs:
            result[name] = _clone(initial_inputs[name])
    return result


def causal_outcome_to_descriptive_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    _treatment, outcome, _confounders = _causal_vectors(initial_inputs)
    return {"data": outcome}


def causal_ticket_to_estimate(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    treatment, outcome, confounders = _causal_vectors(initial_inputs)
    mode = str(initial_inputs.get("mode") or "")
    if mode not in {"backdoor_adjustment", "propensity_weighting"}:
        raise PipelineAdapterError("dynamic causal estimate admits only backdoor_adjustment or propensity_weighting")
    if not confounders:
        raise PipelineAdapterError("dynamic causal estimation requires at least one declared confounder")
    result: dict[str, Any] = {
        "mode": mode,
        "treatment": treatment,
        "outcome": outcome,
        "confounders": confounders,
    }
    if mode == "propensity_weighting" and "propensity_clip" in initial_inputs:
        result["propensity_clip"] = _clone(initial_inputs["propensity_clip"])
    return result


def causal_ticket_to_placebo_refutation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results
    treatment, outcome, confounders = _causal_vectors(initial_inputs)
    fixed = _mapping(stage.get("fixed_parameters", {}), "stage.fixed_parameters")
    repetitions = fixed.get("repetitions", 200)
    seed = fixed.get("seed", 0)
    alpha = fixed.get("alpha", 0.05)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 20 <= repetitions <= 2000:
        raise PipelineAdapterError("causal placebo repetitions must be an integer between 20 and 2000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise PipelineAdapterError("causal placebo seed is out of range")
    alpha_value = _finite(alpha, "stage.fixed_parameters.alpha")
    if not 0 < alpha_value < 1:
        raise PipelineAdapterError("causal placebo alpha must be within (0,1)")
    return {
        "mode": "placebo_policy_test",
        "treatment": treatment,
        "outcome": outcome,
        "confounders": confounders,
        "repetitions": repetitions,
        "seed": seed,
        "alpha": alpha_value,
    }


ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "ticket_inputs": ticket_inputs,
    "scenario_ranking_to_descriptive_statistics": scenario_ranking_to_descriptive_statistics,
    "scenario_ranking_to_sensitivity": scenario_ranking_to_sensitivity,
    "dynamic_scenario_ranking_to_sensitivity": dynamic_scenario_ranking_to_sensitivity,
    "scenario_ranking_to_monte_carlo": scenario_ranking_to_monte_carlo,
    "scenario_ranking_to_constrained_optimization": scenario_ranking_to_constrained_optimization,
    "time_series_to_descriptive_statistics": time_series_to_descriptive_statistics,
    "time_series_to_pattern_discovery": time_series_to_pattern_discovery,
    "time_series_to_assumption_validation": time_series_to_assumption_validation,
    "time_series_to_forecast": time_series_to_forecast,
    "causal_outcome_to_descriptive_statistics": causal_outcome_to_descriptive_statistics,
    "causal_ticket_to_estimate": causal_ticket_to_estimate,
    "causal_ticket_to_placebo_refutation": causal_ticket_to_placebo_refutation,
}
