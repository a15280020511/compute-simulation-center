#!/usr/bin/env python3
"""Deterministic adapters for the reliability dynamic capability family."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from pipeline_adapters import ADAPTERS, PipelineAdapterError


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineAdapterError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineAdapterError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise PipelineAdapterError(f"{name} must be between 0 and 1")
    return result


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    context = _mapping(initial_inputs.get("reliability_context"), "ticket inputs.reliability_context")
    tail = str(context.get("tail") or "lower").lower()
    if tail not in {"lower", "upper"}:
        raise PipelineAdapterError("reliability_context.tail must be lower or upper")
    return context


def _statistics(stage_results: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _mapping(stage_results.get("sample_statistics"), "stage results.sample_statistics")
    mean = _finite(result.get("mean"), "sample statistics.mean")
    std = _finite(result.get("standard_deviation_population"), "sample statistics.standard_deviation_population")
    if std <= 0:
        raise PipelineAdapterError("sample standard deviation must be positive for normal reliability analysis")
    return {**result, "mean": mean, "standard_deviation_population": std}


def reliability_ticket_to_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    data = initial_inputs.get("data")
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or len(data) < 2:
        raise PipelineAdapterError("reliability entry requires at least two sample observations")
    return {"data": _clone(data)}


def statistics_to_openturns_reliability(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    stats = _statistics(stage_results)
    context = _context(initial_inputs)
    return {
        "mode": "openturns_reliability_probability",
        "mean": stats["mean"],
        "standard_deviation": stats["standard_deviation_population"],
        "threshold": _finite(context.get("threshold"), "reliability_context.threshold"),
        "tail": str(context.get("tail") or "lower").lower(),
    }


def statistics_to_reliability_monte_carlo(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    stats = _statistics(stage_results)
    context = _context(initial_inputs)
    iterations = context.get("monte_carlo_iterations", 20000)
    seed = context.get("monte_carlo_seed", 0)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 100 <= iterations <= 100000:
        raise PipelineAdapterError("monte_carlo_iterations must be an integer between 100 and 100000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise PipelineAdapterError("monte_carlo_seed must be an integer between 0 and 2^32-1")
    return {
        "iterations": iterations,
        "seed": seed,
        "variables": [
            {
                "name": "reliability_variable",
                "distribution": "normal",
                "mean": stats["mean"],
                "standard_deviation": stats["standard_deviation_population"],
            }
        ],
        "model": {
            "intercept": 0.0,
            "coefficients": {"reliability_variable": 1.0},
        },
        "threshold": _finite(context.get("threshold"), "reliability_context.threshold"),
    }


def analytic_mc_to_benchmark(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    analytic = _mapping(stage_results.get("analytic_reliability"), "stage results.analytic_reliability")
    simulation = _mapping(stage_results.get("monte_carlo_validation"), "stage results.monte_carlo_validation")
    context = _context(initial_inputs)
    analytic_failure = _probability(analytic.get("failure_probability"), "analytic failure_probability")
    probability_below = _probability(simulation.get("probability_below_threshold"), "Monte Carlo probability_below_threshold")
    tail = str(context.get("tail") or "lower").lower()
    simulated_failure = probability_below if tail == "lower" else 1.0 - probability_below
    tolerance = _finite(context.get("mc_agreement_tolerance", 0.02), "reliability_context.mc_agreement_tolerance")
    if not 0 < tolerance <= 0.25:
        raise PipelineAdapterError("mc_agreement_tolerance must be greater than 0 and at most 0.25")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "analytic_vs_monte_carlo_failure_probability",
                "observed": simulated_failure,
                "benchmark": analytic_failure,
                "tolerance": tolerance,
                "direction": "absolute",
            }
        ],
    }


def analytic_reliability_to_external_benchmark(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    analytic = _mapping(stage_results.get("analytic_reliability"), "stage results.analytic_reliability")
    context = _context(initial_inputs)
    expected = _probability(context.get("external_failure_probability"), "reliability_context.external_failure_probability")
    tolerance = _finite(context.get("external_benchmark_tolerance"), "reliability_context.external_benchmark_tolerance")
    if not 0 <= tolerance <= 0.25:
        raise PipelineAdapterError("external_benchmark_tolerance must be between 0 and 0.25")
    observed = _probability(analytic.get("failure_probability"), "analytic failure_probability")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "analytic_failure_probability_external_benchmark",
                "observed": observed,
                "benchmark": expected,
                "tolerance": tolerance,
                "direction": "absolute",
            }
        ],
    }


RELIABILITY_ADAPTERS: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "reliability_ticket_to_statistics": reliability_ticket_to_statistics,
    "statistics_to_openturns_reliability": statistics_to_openturns_reliability,
    "statistics_to_reliability_monte_carlo": statistics_to_reliability_monte_carlo,
    "analytic_mc_to_benchmark": analytic_mc_to_benchmark,
    "analytic_reliability_to_external_benchmark": analytic_reliability_to_external_benchmark,
}


def install_reliability_adapters() -> None:
    for name, handler in RELIABILITY_ADAPTERS.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"conflicting pipeline adapter registration: {name}")
        ADAPTERS[name] = handler
