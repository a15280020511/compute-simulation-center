#!/usr/bin/env python3
"""Deterministic adapters for the dynamic policy-simulation family."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pipeline_adapters import ADAPTERS, PipelineAdapterError


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineAdapterError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("policy_context")
    return {} if raw is None else _mapping(raw, "inputs.policy_context")


def policy_ticket_to_microsimulation(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "policy_microsimulation":
        raise PipelineAdapterError("policy-simulation family requires policy_microsimulation entry mode")
    return {key: value for key, value in initial_inputs.items() if key != "policy_context"}


def microsimulation_to_disposable_statistics(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del initial_inputs, stage
    primary = _mapping(stage_results.get("policy_microsimulation"), "stage_results.policy_microsimulation")
    values = primary.get("individual_results")
    if not isinstance(values, list) or len(values) < 10:
        raise PipelineAdapterError("policy microsimulation must emit at least ten individual_results")
    return {"data": list(values)}


def policy_stats_to_mean_consistency_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("policy_microsimulation"), "stage_results.policy_microsimulation")
    stats = _mapping(stage_results.get("disposable_distribution_statistics"), "stage_results.disposable_distribution_statistics")
    observed = _finite(stats.get("mean"), "disposable_distribution_statistics.mean")
    benchmark = _finite(primary.get("mean_disposable_income"), "policy_microsimulation.mean_disposable_income")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("mean_consistency_tolerance", 1e-9), "policy_context.mean_consistency_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("mean_consistency_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "disposable-income-mean-cross-tool-consistency",
            "observed": observed,
            "benchmark": benchmark,
            "tolerance": tolerance,
            "direction": "absolute",
        }],
    }


def microsimulation_to_policy_target_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("policy_microsimulation"), "stage_results.policy_microsimulation")
    context = _context(initial_inputs)
    specs = (
        ("net_fiscal_balance", "expected_net_fiscal_balance", "net_fiscal_balance_tolerance"),
        ("gini_after", "expected_gini_after", "gini_after_tolerance"),
        ("poverty_rate_after", "expected_poverty_rate_after", "poverty_rate_after_tolerance"),
    )
    candidates = []
    for observed_name, expected_name, tolerance_name in specs:
        if expected_name not in context:
            continue
        observed = _finite(primary.get(observed_name), f"policy_microsimulation.{observed_name}")
        expected = _finite(context.get(expected_name), f"policy_context.{expected_name}")
        tolerance = _finite(context.get(tolerance_name, 0.0), f"policy_context.{tolerance_name}")
        if tolerance < 0:
            raise PipelineAdapterError(f"{tolerance_name} must be non-negative")
        if observed_name in {"gini_after", "poverty_rate_after"} and not 0.0 <= expected <= 1.0:
            raise PipelineAdapterError(f"{expected_name} must be between 0 and 1")
        candidates.append({
            "name": f"policy-target-{observed_name}",
            "observed": observed,
            "benchmark": expected,
            "tolerance": tolerance,
            "direction": "absolute",
        })
    if not candidates:
        raise PipelineAdapterError("policy_target_audit requires at least one explicit expected policy target")
    return {"mode": "benchmark_comparison", "candidates": candidates}


def install_policy_simulation_adapters() -> None:
    adapters = {
        "policy_ticket_to_microsimulation": policy_ticket_to_microsimulation,
        "microsimulation_to_disposable_statistics": microsimulation_to_disposable_statistics,
        "policy_stats_to_mean_consistency_audit": policy_stats_to_mean_consistency_audit,
        "microsimulation_to_policy_target_audit": microsimulation_to_policy_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"policy-simulation adapter name collision: {name}")
        ADAPTERS[name] = handler
