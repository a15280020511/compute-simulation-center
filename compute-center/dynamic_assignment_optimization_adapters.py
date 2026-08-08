#!/usr/bin/env python3
"""Deterministic adapters and independent SciPy checks for assignment optimization."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline_adapters import ADAPTERS, PipelineAdapterError

PRIMARY_STAGE = "assignment_optimization"


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
    result = float(value)
    if not math.isfinite(result):
        raise PipelineAdapterError(f"{name} must be finite")
    return result


def _context(initial_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = initial_inputs.get("assignment_optimization_context")
    return {} if raw is None else _mapping(raw, "inputs.assignment_optimization_context")


def _problem(initial_inputs: Mapping[str, Any]) -> tuple[list[str], list[str], np.ndarray, bool]:
    workers = [str(item) for item in _sequence(initial_inputs.get("workers"), "inputs.workers")]
    tasks = [str(item) for item in _sequence(initial_inputs.get("tasks"), "inputs.tasks")]
    if not 1 <= len(workers) <= 100 or not 1 <= len(tasks) <= 100:
        raise PipelineAdapterError("workers and tasks must contain 1 to 100 entries")
    if any(not item for item in workers + tasks) or len(set(workers)) != len(workers) or len(set(tasks)) != len(tasks):
        raise PipelineAdapterError("worker and task names must be non-empty and unique")
    if len(workers) < len(tasks):
        raise PipelineAdapterError("require_all_tasks=true needs at least as many workers as tasks")
    if initial_inputs.get("require_all_tasks", True) is not True:
        raise PipelineAdapterError("dynamic assignment-optimization v1 requires require_all_tasks=true")
    raw_costs = _sequence(initial_inputs.get("costs"), "inputs.costs")
    if len(raw_costs) != len(workers):
        raise PipelineAdapterError("costs must have one row per worker")
    rows: list[list[float]] = []
    for i, raw_row in enumerate(raw_costs):
        row = [_finite(value, f"inputs.costs[{i}][{j}]") for j, value in enumerate(_sequence(raw_row, f"inputs.costs[{i}]"))]
        if len(row) != len(tasks):
            raise PipelineAdapterError("costs must be a worker-by-task matrix")
        rows.append(row)
    maximize = initial_inputs.get("maximize", False)
    if not isinstance(maximize, bool):
        raise PipelineAdapterError("maximize must be boolean")
    return workers, tasks, np.asarray(rows, dtype=float), maximize


def _primary(stage_results: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    return _mapping(stage_results.get(PRIMARY_STAGE), f"stage_results.{PRIMARY_STAGE}")


def assignment_ticket_to_ortools(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage_results, stage
    if str(initial_inputs.get("mode") or "") != "assignment_optimization":
        raise PipelineAdapterError("assignment family requires assignment_optimization entry mode")
    _problem(initial_inputs)
    return {key: value for key, value in initial_inputs.items() if key != "assignment_optimization_context"}


def assignment_to_scipy_exact_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    workers, tasks, costs, maximize = _problem(initial_inputs)
    primary = _primary(stage_results)
    assignments = _sequence(primary.get("assignments"), "primary.assignments")
    if len(assignments) != len(tasks):
        raise PipelineAdapterError("primary assignment count must equal task count")
    worker_index = {name: index for index, name in enumerate(workers)}
    task_index = {name: index for index, name in enumerate(tasks)}
    seen_workers: set[str] = set()
    seen_tasks: set[str] = set()
    recomputed = 0.0
    for index, raw in enumerate(assignments):
        row = _mapping(raw, f"primary.assignments[{index}]")
        worker = str(row.get("worker") or "")
        task = str(row.get("task") or "")
        if worker not in worker_index or task not in task_index:
            raise PipelineAdapterError("primary assignment references unknown worker or task")
        if worker in seen_workers or task in seen_tasks:
            raise PipelineAdapterError("primary assignment reuses a worker or task")
        seen_workers.add(worker); seen_tasks.add(task)
        expected_value = float(costs[worker_index[worker], task_index[task]])
        reported_value = _finite(row.get("value"), f"primary.assignments[{index}].value")
        tolerance = _finite(_context(initial_inputs).get("exact_consistency_tolerance", 1e-9), "assignment_optimization_context.exact_consistency_tolerance")
        if abs(reported_value - expected_value) > tolerance:
            raise PipelineAdapterError("primary assignment value disagrees with original cost matrix")
        recomputed += expected_value
    if seen_tasks != set(tasks):
        raise PipelineAdapterError("primary assignments do not cover every task exactly once")
    objective = _finite(primary.get("objective_value"), "primary.objective_value")
    context = _context(initial_inputs)
    tolerance = _finite(context.get("exact_consistency_tolerance", 1e-9), "assignment_optimization_context.exact_consistency_tolerance")
    if not 0 <= tolerance <= 1e-6:
        raise PipelineAdapterError("exact_consistency_tolerance must be between 0 and 1e-6")
    solve_matrix = -costs if maximize else costs
    row_ind, col_ind = linear_sum_assignment(solve_matrix)
    if len(col_ind) != len(tasks) or set(int(value) for value in col_ind) != set(range(len(tasks))):
        raise PipelineAdapterError("SciPy exact assignment did not cover every task")
    scipy_objective = float(sum(costs[int(r), int(c)] for r, c in zip(row_ind, col_ind, strict=True)))
    return {
        "mode": "benchmark_comparison",
        "candidates": [
            {"name": "reported-vs-recomputed-objective", "observed": objective, "benchmark": recomputed, "tolerance": tolerance, "direction": "absolute"},
            {"name": "ortools-vs-scipy-optimum", "observed": objective, "benchmark": scipy_objective, "tolerance": tolerance, "direction": "absolute"},
        ],
    }


def assignment_to_objective_target_audit(initial_inputs: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]], stage: Mapping[str, Any]) -> dict[str, Any]:
    del stage
    _, _, _, maximize = _problem(initial_inputs)
    primary = _primary(stage_results)
    context = _context(initial_inputs)
    tolerance = _finite(context.get("objective_target_tolerance", 0.0), "assignment_optimization_context.objective_target_tolerance")
    if tolerance < 0:
        raise PipelineAdapterError("objective_target_tolerance must be non-negative")
    if maximize:
        if "minimum_objective_value" not in context:
            raise PipelineAdapterError("maximize=true objective target requires minimum_objective_value")
        benchmark = _finite(context["minimum_objective_value"], "assignment_optimization_context.minimum_objective_value")
        direction = "minimum"
        name = "minimum-objective-value"
    else:
        if "maximum_objective_value" not in context:
            raise PipelineAdapterError("maximize=false objective target requires maximum_objective_value")
        benchmark = _finite(context["maximum_objective_value"], "assignment_optimization_context.maximum_objective_value")
        direction = "maximum"
        name = "maximum-objective-value"
    return {"mode": "benchmark_comparison", "candidates": [{"name": name, "observed": _finite(primary.get("objective_value"), "primary.objective_value"), "benchmark": benchmark, "tolerance": tolerance, "direction": direction}]}


def install_assignment_optimization_adapters() -> None:
    adapters = {
        "assignment_ticket_to_ortools": assignment_ticket_to_ortools,
        "assignment_to_scipy_exact_audit": assignment_to_scipy_exact_audit,
        "assignment_to_objective_target_audit": assignment_to_objective_target_audit,
    }
    for name, handler in adapters.items():
        existing = ADAPTERS.get(name)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"assignment-optimization adapter name collision: {name}")
        ADAPTERS[name] = handler
