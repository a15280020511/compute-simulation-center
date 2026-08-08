#!/usr/bin/env python3
"""Deterministic adapters for the dynamic game-theory capability family."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pipeline_adapters import ADAPTERS, PipelineAdapterError

ALLOWED_GAMES = {"matrix_rps", "matrix_pd"}


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
        raise PipelineAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    value = initial_inputs.get("game_context")
    return {} if value is None else _mapping(value, "inputs.game_context")


def game_ticket_to_open_spiel(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "open_spiel_policy_evaluation":
        raise PipelineAdapterError("game-theory family requires open_spiel_policy_evaluation entry mode")
    game_id = str(initial_inputs.get("game_id") or "matrix_rps")
    if game_id not in ALLOWED_GAMES:
        raise PipelineAdapterError("game_id is outside the controlled game allowlist")
    return {key: value for key, value in initial_inputs.items() if key != "game_context"}


def open_spiel_to_pygambit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del initial_inputs, stage
    primary = _mapping(stage_results.get("policy_evaluation"), "stage_results.policy_evaluation")
    tensor = _sequence(primary.get("payoff_tensor"), "policy_evaluation.payoff_tensor")
    row_payoffs: list[list[float]] = []
    column_payoffs: list[list[float]] = []
    width: int | None = None
    for i, raw_row in enumerate(tensor):
        row = _sequence(raw_row, f"payoff_tensor[{i}]")
        if width is None:
            width = len(row)
        if not row or len(row) != width:
            raise PipelineAdapterError("payoff_tensor must be rectangular")
        row_values: list[float] = []
        column_values: list[float] = []
        for j, raw_cell in enumerate(row):
            cell = _sequence(raw_cell, f"payoff_tensor[{i}][{j}]")
            if len(cell) < 2:
                raise PipelineAdapterError("payoff_tensor cells require two player utilities")
            row_values.append(_finite(cell[0], f"payoff_tensor[{i}][{j}][0]"))
            column_values.append(_finite(cell[1], f"payoff_tensor[{i}][{j}][1]"))
        row_payoffs.append(row_values)
        column_payoffs.append(column_values)
    if not row_payoffs:
        raise PipelineAdapterError("payoff_tensor must be non-empty")
    return {
        "mode": "pygambit_pure_equilibria",
        "row_payoffs": row_payoffs,
        "column_payoffs": column_payoffs,
    }


def pure_equilibria_to_count_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    equilibrium = _mapping(stage_results.get("pure_equilibria"), "stage_results.pure_equilibria")
    rows = _sequence(equilibrium.get("pure_equilibria"), "pure_equilibria.pure_equilibria")
    context = _context(initial_inputs)
    expected = context.get("expected_pure_equilibrium_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or not 0 <= expected <= 900:
        raise PipelineAdapterError("expected_pure_equilibrium_count must be an integer from 0 to 900")
    return {
        "mode": "benchmark_comparison",
        "candidates": [{
            "name": "pure-equilibrium-count",
            "observed": float(len(rows)),
            "benchmark": float(expected),
            "tolerance": 0.0,
            "direction": "absolute",
        }],
    }


def policy_evaluation_to_utility_audit(
    initial_inputs: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    del stage
    primary = _mapping(stage_results.get("policy_evaluation"), "stage_results.policy_evaluation")
    observed = _sequence(primary.get("expected_utility"), "policy_evaluation.expected_utility")
    if len(observed) != 2:
        raise PipelineAdapterError("expected_utility must contain two player values")
    context = _context(initial_inputs)
    expected = _sequence(context.get("expected_policy_utility"), "game_context.expected_policy_utility")
    if len(expected) != 2:
        raise PipelineAdapterError("expected_policy_utility must contain two values")
    tolerance = _finite(context.get("utility_tolerance"), "game_context.utility_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("utility_tolerance must be non-negative")
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {
                "name": "row-player-expected-utility",
                "observed": _finite(observed[0], "policy_evaluation.expected_utility[0]"),
                "benchmark": _finite(expected[0], "game_context.expected_policy_utility[0]"),
                "tolerance": tolerance,
                "direction": "absolute",
            },
            {
                "name": "column-player-expected-utility",
                "observed": _finite(observed[1], "policy_evaluation.expected_utility[1]"),
                "benchmark": _finite(expected[1], "game_context.expected_policy_utility[1]"),
                "tolerance": tolerance,
                "direction": "absolute",
            },
        ],
    }


def install_game_theory_adapters() -> None:
    adapters = {
        "game_ticket_to_open_spiel": game_ticket_to_open_spiel,
        "open_spiel_to_pygambit": open_spiel_to_pygambit,
        "pure_equilibria_to_count_audit": pure_equilibria_to_count_audit,
        "policy_evaluation_to_utility_audit": policy_evaluation_to_utility_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"game-theory adapter name collision: {name}")
        ADAPTERS[name] = handler
