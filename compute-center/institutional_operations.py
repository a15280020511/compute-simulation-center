#!/usr/bin/env python3
"""Fixed system-dynamics and crisis-warning operations using NumPy and SciPy only."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog

from compute_runner import ComputeError

MAX_STEPS = 10_000
MAX_STOCKS = 50
MAX_INDICATORS = 200
MAX_SCENARIOS = 50


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ComputeError(f"{name} must be between 0 and 1")
    return result


def _sample(history: list[dict[str, Any]], step: int, steps: int, row: dict[str, Any]) -> None:
    every = max(1, steps // 100)
    if step in {0, steps} or step % every == 0:
        history.append({"step": step, **row})


def _system_base(inputs: Mapping[str, Any]) -> tuple[int, float]:
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    dt = _finite(inputs.get("dt", 1.0), "inputs.dt")
    if dt <= 0 or dt > 1000:
        raise ComputeError("inputs.dt must be in (0,1000]")
    return steps, dt


def _stock_flow(inputs: Mapping[str, Any]) -> dict[str, Any]:
    steps, dt = _system_base(inputs)
    raw = _sequence(inputs.get("stocks"), "inputs.stocks")
    if not 1 <= len(raw) <= MAX_STOCKS:
        raise ComputeError(f"inputs.stocks must contain 1 to {MAX_STOCKS} entries")
    stocks = []
    for index, item in enumerate(raw):
        row = _mapping(item, f"inputs.stocks[{index}]")
        name = str(row.get("name") or "").strip()
        initial = _finite(row.get("initial"), f"stocks[{index}].initial")
        inflow = _finite(row.get("inflow", 0.0), f"stocks[{index}].inflow")
        outflow_rate = _finite(row.get("outflow_rate", 0.0), f"stocks[{index}].outflow_rate")
        capacity = _finite(row.get("capacity", max(initial, 1e12)), f"stocks[{index}].capacity")
        if not name or initial < 0 or inflow < 0 or outflow_rate < 0 or capacity < initial:
            raise ComputeError("stock definitions must have non-negative flows and capacity >= initial")
        stocks.append({"name": name, "value": initial, "initial": initial, "inflow": inflow, "outflow_rate": outflow_rate, "capacity": capacity, "total_inflow": 0.0, "total_outflow": 0.0})
    if len({row["name"] for row in stocks}) != len(stocks):
        raise ComputeError("stock names must be unique")
    history: list[dict[str, Any]] = []
    _sample(history, 0, steps, {"stocks": [row["value"] for row in stocks]})
    for step in range(1, steps + 1):
        for row in stocks:
            inflow = row["inflow"] * dt
            outflow = min(row["value"] + inflow, row["outflow_rate"] * row["value"] * dt)
            row["value"] = min(row["capacity"], max(0.0, row["value"] + inflow - outflow))
            row["total_inflow"] += inflow
            row["total_outflow"] += outflow
        _sample(history, step, steps, {"stocks": [row["value"] for row in stocks]})
    results = []
    for row in stocks:
        residual = row["initial"] + row["total_inflow"] - row["total_outflow"] - row["value"]
        results.append({"name": row["name"], "initial": row["initial"], "final": row["value"], "total_inflow": row["total_inflow"], "total_outflow": row["total_outflow"], "conservation_residual": residual})
    return {"mode": "stock_flow", "engine": {"name": "numpy", "version": np.__version__}, "steps": steps, "dt": dt, "stocks": results, "history": history, "conservation_passed": all(abs(row["conservation_residual"]) < 1e-7 for row in results)}


def _feedback_delay(inputs: Mapping[str, Any]) -> dict[str, Any]:
    steps, dt = _system_base(inputs)
    state = _finite(inputs.get("initial_state"), "inputs.initial_state")
    exogenous = _finite(inputs.get("exogenous_input", 0.0), "inputs.exogenous_input")
    decay = _finite(inputs.get("decay_rate", 0.0), "inputs.decay_rate")
    gain = _finite(inputs.get("feedback_gain", 0.0), "inputs.feedback_gain")
    delay = _integer(inputs.get("delay_steps", 1), "inputs.delay_steps", 1, min(steps, 1000))
    if decay < 0:
        raise ComputeError("decay_rate must be non-negative")
    values = [state]
    history: list[dict[str, Any]] = []
    _sample(history, 0, steps, {"state": state})
    for step in range(1, steps + 1):
        delayed = values[max(0, len(values) - delay)]
        derivative = exogenous + gain * delayed - decay * state
        state = state + dt * derivative
        if not math.isfinite(state):
            raise ComputeError("feedback-delay system became numerically unstable")
        values.append(state)
        _sample(history, step, steps, {"state": state, "delayed_state": delayed})
    return {"mode": "feedback_delay", "engine": {"name": "numpy", "version": np.__version__}, "final_state": state, "minimum_state": min(values), "maximum_state": max(values), "delay_steps": delay, "history": history, "unstable": abs(state) > 1e12}


def _policy_switch(inputs: Mapping[str, Any]) -> dict[str, Any]:
    steps, dt = _system_base(inputs)
    state = _finite(inputs.get("initial_state"), "inputs.initial_state")
    capacity = _finite(inputs.get("capacity"), "inputs.capacity")
    before = _finite(inputs.get("growth_rate_before"), "inputs.growth_rate_before")
    after = _finite(inputs.get("growth_rate_after"), "inputs.growth_rate_after")
    switch = _integer(inputs.get("switch_step"), "inputs.switch_step", 1, steps)
    if not 0 <= state <= capacity or capacity <= 0:
        raise ComputeError("initial_state must be within a positive capacity")
    history: list[dict[str, Any]] = []
    _sample(history, 0, steps, {"state": state, "policy_active": False})
    for step in range(1, steps + 1):
        rate = after if step >= switch else before
        state = float(np.clip(state + dt * rate * state * (1 - state / capacity), 0, capacity))
        _sample(history, step, steps, {"state": state, "policy_active": step >= switch})
    return {"mode": "policy_switch", "engine": {"name": "numpy", "version": np.__version__}, "switch_step": switch, "final_state": state, "capacity": capacity, "history": history}


def _coupled_capacity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    steps, dt = _system_base(inputs)
    demand = _finite(inputs.get("initial_demand"), "inputs.initial_demand")
    capacity = _finite(inputs.get("initial_capacity"), "inputs.initial_capacity")
    backlog = _finite(inputs.get("initial_backlog", 0.0), "inputs.initial_backlog")
    growth = _finite(inputs.get("demand_growth", 0.0), "inputs.demand_growth")
    addition = _finite(inputs.get("capacity_addition", 0.0), "inputs.capacity_addition")
    service_rate = _probability(inputs.get("service_rate", 1.0), "inputs.service_rate")
    if min(demand, capacity, backlog, addition) < 0:
        raise ComputeError("demand, capacity, backlog and capacity_addition must be non-negative")
    history: list[dict[str, Any]] = []
    _sample(history, 0, steps, {"demand": demand, "capacity": capacity, "backlog": backlog})
    for step in range(1, steps + 1):
        demand = max(0.0, demand * (1 + growth * dt))
        capacity = max(0.0, capacity + addition * dt)
        available = demand + backlog
        served = min(available, capacity * service_rate * dt)
        backlog = max(0.0, available - served)
        _sample(history, step, steps, {"demand": demand, "capacity": capacity, "backlog": backlog, "served": served})
    return {"mode": "coupled_capacity", "engine": {"name": "numpy", "version": np.__version__}, "final_demand": demand, "final_capacity": capacity, "final_backlog": backlog, "capacity_gap": demand + backlog - capacity, "history": history}


def _resource_depletion(inputs: Mapping[str, Any]) -> dict[str, Any]:
    steps, dt = _system_base(inputs)
    stock = _finite(inputs.get("initial_stock"), "inputs.initial_stock")
    capacity = _finite(inputs.get("carrying_capacity"), "inputs.carrying_capacity")
    regeneration = _finite(inputs.get("regeneration_rate", 0.0), "inputs.regeneration_rate")
    extraction = _finite(inputs.get("extraction", 0.0), "inputs.extraction")
    if not 0 <= stock <= capacity or capacity <= 0 or regeneration < 0 or extraction < 0:
        raise ComputeError("resource parameters are outside valid bounds")
    depleted_at = None
    history: list[dict[str, Any]] = []
    _sample(history, 0, steps, {"stock": stock})
    for step in range(1, steps + 1):
        growth = regeneration * stock * (1 - stock / capacity)
        stock = max(0.0, min(capacity, stock + dt * growth - dt * extraction))
        if stock <= 1e-9 and depleted_at is None:
            depleted_at = step
        _sample(history, step, steps, {"stock": stock})
    return {"mode": "resource_depletion", "engine": {"name": "numpy", "version": np.__version__}, "final_stock": stock, "depleted": depleted_at is not None, "depleted_at_step": depleted_at, "history": history}


def _adoption_saturation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    steps, dt = _system_base(inputs)
    adoption = _probability(inputs.get("initial_adoption", 0.01), "inputs.initial_adoption")
    innovation = _finite(inputs.get("innovation_rate", 0.01), "inputs.innovation_rate")
    imitation = _finite(inputs.get("imitation_rate", 0.1), "inputs.imitation_rate")
    if innovation < 0 or imitation < 0:
        raise ComputeError("innovation_rate and imitation_rate must be non-negative")
    history: list[dict[str, Any]] = []
    _sample(history, 0, steps, {"adoption": adoption})
    for step in range(1, steps + 1):
        adoption = float(np.clip(adoption + dt * (innovation + imitation * adoption) * (1 - adoption), 0, 1))
        _sample(history, step, steps, {"adoption": adoption})
    return {"mode": "adoption_saturation", "engine": {"name": "numpy", "version": np.__version__}, "final_adoption": adoption, "history": history}


def system_dynamics_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {"stock_flow": _stock_flow, "feedback_delay": _feedback_delay, "policy_switch": _policy_switch, "coupled_capacity": _coupled_capacity, "resource_depletion": _resource_depletion, "adoption_saturation": _adoption_saturation}
    if mode not in handlers:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(handlers))}")
    return handlers[mode](inputs)


def _composite_risk_index(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw = _sequence(inputs.get("indicators"), "inputs.indicators")
    if not 1 <= len(raw) <= MAX_INDICATORS:
        raise ComputeError(f"inputs.indicators must contain 1 to {MAX_INDICATORS} entries")
    rows = []
    total_weight = 0.0
    for index, item in enumerate(raw):
        row = _mapping(item, f"inputs.indicators[{index}]")
        name = str(row.get("name") or "").strip()
        value = _finite(row.get("value"), f"indicators[{index}].value")
        minimum = _finite(row.get("minimum"), f"indicators[{index}].minimum")
        maximum = _finite(row.get("maximum"), f"indicators[{index}].maximum")
        weight = _finite(row.get("weight", 1.0), f"indicators[{index}].weight")
        direction = str(row.get("direction") or "higher_risk")
        if not name or maximum <= minimum or weight < 0 or direction not in {"higher_risk", "lower_risk"}:
            raise ComputeError("invalid composite-risk indicator")
        normalized = float(np.clip((value - minimum) / (maximum - minimum), 0, 1))
        if direction == "lower_risk":
            normalized = 1 - normalized
        rows.append({"name": name, "normalized_risk": normalized, "weight": weight, "contribution": normalized * weight})
        total_weight += weight
    if total_weight <= 0:
        raise ComputeError("indicator weights must have positive total")
    score = sum(row["contribution"] for row in rows) / total_weight
    return {"mode": "composite_risk_index", "risk_score": score, "risk_level": "high" if score >= 0.7 else "medium" if score >= 0.4 else "low", "indicators": rows}


def _change_point_warning(inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray([_finite(item, f"inputs.values[{index}]") for index, item in enumerate(_sequence(inputs.get("values"), "inputs.values"))], dtype=float)
    if not 10 <= values.size <= 100_000:
        raise ComputeError("inputs.values must contain 10 to 100000 observations")
    baseline_window = _integer(inputs.get("baseline_window", max(5, values.size // 5)), "inputs.baseline_window", 5, values.size - 1)
    threshold = _finite(inputs.get("threshold", 3.0), "inputs.threshold")
    baseline = values[:baseline_window]
    mean = float(np.mean(baseline)); scale = max(float(np.std(baseline, ddof=1)), 1e-12)
    z = (values - mean) / scale
    alarms = [int(index) for index, value in enumerate(z) if index >= baseline_window and abs(value) >= threshold]
    return {"mode": "change_point_warning", "baseline_mean": mean, "baseline_standard_deviation": scale, "threshold": threshold, "alarm_indices": alarms, "alarm_count": len(alarms), "maximum_absolute_z_score": float(np.max(np.abs(z[baseline_window:]))) if baseline_window < values.size else 0.0}


def _binary_arrays(inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray([_probability(item, f"inputs.probabilities[{index}]") for index, item in enumerate(_sequence(inputs.get("probabilities"), "inputs.probabilities"))], dtype=float)
    outcomes = np.asarray(list(_sequence(inputs.get("outcomes"), "inputs.outcomes")), dtype=int)
    if probabilities.ndim != 1 or probabilities.size == 0 or probabilities.shape != outcomes.shape or np.any((outcomes != 0) & (outcomes != 1)):
        raise ComputeError("probabilities and binary outcomes must be equal-length non-empty arrays")
    return probabilities, outcomes


def _classification_metrics(probabilities: np.ndarray, outcomes: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    actual = outcomes == 1
    tp = int(np.sum(predicted & actual)); fp = int(np.sum(predicted & ~actual)); tn = int(np.sum(~predicted & ~actual)); fn = int(np.sum(~predicted & actual))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn, "precision": precision, "recall": recall, "false_alarm_rate": fp / max(fp + tn, 1), "miss_rate": fn / max(fn + tp, 1)}


def _alert_threshold_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    probabilities, outcomes = _binary_arrays(inputs)
    fp_cost = _finite(inputs.get("false_positive_cost", 1.0), "inputs.false_positive_cost")
    fn_cost = _finite(inputs.get("false_negative_cost", 5.0), "inputs.false_negative_cost")
    if min(fp_cost, fn_cost) < 0:
        raise ComputeError("alert costs must be non-negative")
    thresholds = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    rows = []
    for threshold in thresholds:
        metrics = _classification_metrics(probabilities, outcomes, float(threshold))
        loss = metrics["false_positive"] * fp_cost + metrics["false_negative"] * fn_cost
        rows.append({"threshold": float(threshold), "decision_loss": float(loss), **metrics})
    selected = min(rows, key=lambda row: (row["decision_loss"], -row["recall"], row["threshold"]))
    return {"mode": "alert_threshold_optimization", "selected": selected, "candidate_count": len(rows), "false_positive_cost": fp_cost, "false_negative_cost": fn_cost}


def _scenario_escalation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(_sequence(inputs.get("transition_matrix"), "inputs.transition_matrix"), dtype=float)
    initial = np.asarray(_sequence(inputs.get("initial_distribution"), "inputs.initial_distribution"), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not 2 <= matrix.shape[0] <= MAX_SCENARIOS or initial.shape != (matrix.shape[0],):
        raise ComputeError("transition_matrix must be square and match initial_distribution")
    if np.any(matrix < 0) or np.any(initial < 0) or not np.allclose(matrix.sum(axis=1), 1, atol=1e-8) or not np.isclose(initial.sum(), 1, atol=1e-8):
        raise ComputeError("transition probabilities must be non-negative and normalized")
    steps = _integer(inputs.get("steps", 10), "inputs.steps", 1, 1000)
    severe = [int(item) for item in _sequence(inputs.get("severe_states", [matrix.shape[0] - 1]), "inputs.severe_states")]
    if any(item < 0 or item >= matrix.shape[0] for item in severe):
        raise ComputeError("severe state index is outside transition matrix")
    distribution = initial.copy(); history = [{"step": 0, "distribution": distribution.tolist()}]
    for step in range(1, steps + 1):
        distribution = distribution @ matrix
        if step == steps or step % max(1, steps // 100) == 0:
            history.append({"step": step, "distribution": distribution.tolist()})
    return {"mode": "scenario_escalation", "steps": steps, "final_distribution": distribution.tolist(), "severe_probability": float(np.sum(distribution[severe])), "history": history}


def _response_resource_allocation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw = _sequence(inputs.get("demands"), "inputs.demands")
    if not 1 <= len(raw) <= 200:
        raise ComputeError("inputs.demands must contain 1 to 200 entries")
    available = _finite(inputs.get("available_resource"), "inputs.available_resource")
    if available < 0:
        raise ComputeError("available_resource must be non-negative")
    rows = []
    for index, item in enumerate(raw):
        row = _mapping(item, f"inputs.demands[{index}]")
        name = str(row.get("name") or "").strip()
        need = _finite(row.get("need"), f"demands[{index}].need")
        priority = _finite(row.get("priority", 1.0), f"demands[{index}].priority")
        if not name or need < 0 or priority < 0:
            raise ComputeError("response demand has invalid name, need or priority")
        rows.append({"name": name, "need": need, "priority": priority})
    result = linprog(c=-np.asarray([row["priority"] for row in rows]), A_ub=np.ones((1, len(rows))), b_ub=[available], bounds=[(0, row["need"]) for row in rows], method="highs")
    if not result.success:
        raise ComputeError(f"resource allocation failed: {result.message}")
    allocations = []
    for row, value in zip(rows, result.x, strict=True):
        allocations.append({**row, "allocation": float(value), "coverage": float(value / row["need"]) if row["need"] > 0 else 1.0})
    return {"mode": "response_resource_allocation", "engine": {"name": "scipy-linprog", "version": 1}, "available_resource": available, "allocated_resource": float(np.sum(result.x)), "unallocated_resource": max(0.0, available - float(np.sum(result.x))), "allocations": allocations}


def _ece(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); score = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & ((probabilities <= edges[index + 1]) if index == bins - 1 else (probabilities < edges[index + 1]))
        if np.any(mask):
            score += float(np.mean(mask)) * abs(float(np.mean(probabilities[mask])) - float(np.mean(outcomes[mask])))
    return score


def _warning_performance_evaluation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    probabilities, outcomes = _binary_arrays(inputs)
    threshold = _probability(inputs.get("threshold", 0.5), "inputs.threshold")
    metrics = _classification_metrics(probabilities, outcomes, threshold)
    lead_times_raw = inputs.get("lead_times")
    lead_time = None
    if lead_times_raw is not None:
        lead_times = np.asarray([_finite(item, f"inputs.lead_times[{index}]") for index, item in enumerate(_sequence(lead_times_raw, "inputs.lead_times"))], dtype=float)
        if lead_times.shape != probabilities.shape:
            raise ComputeError("lead_times must match probabilities")
        positive_leads = lead_times[(probabilities >= threshold) & (outcomes == 1)]
        lead_time = float(np.mean(positive_leads)) if positive_leads.size else None
    return {"mode": "warning_performance_evaluation", "threshold": threshold, **metrics, "brier_score": float(np.mean((probabilities - outcomes) ** 2)), "expected_calibration_error": _ece(probabilities, outcomes), "mean_true_positive_lead_time": lead_time, "alert_rate": float(np.mean(probabilities >= threshold))}


def crisis_early_warning(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {"composite_risk_index": _composite_risk_index, "change_point_warning": _change_point_warning, "alert_threshold_optimization": _alert_threshold_optimization, "scenario_escalation": _scenario_escalation, "response_resource_allocation": _response_resource_allocation, "warning_performance_evaluation": _warning_performance_evaluation}
    if mode not in handlers:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(handlers))}")
    return handlers[mode](inputs)


OPERATIONS = {"system_dynamics_simulation": system_dynamics_simulation, "crisis_early_warning": crisis_early_warning}
