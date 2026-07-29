#!/usr/bin/env python3
"""Fixed lightweight simulation operations for the independent compute center.

Only NumPy, SciPy, and SimPy are used. No network, arbitrary code, model calls,
or runtime plugin installation is permitted.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from compute_runner import ComputeError

MAX_ENTITIES = 10_000
MAX_STAGES = 20
MAX_ROUNDS = 10_000
MAX_TRIALS = 2_000
MAX_STRATEGIES = 20
MAX_GENERATIONS = 10_000
MAX_SERIES = 100_000
MAX_FORECAST_HORIZON = 1_000
MAX_ODE_STEPS = 2_000
MAX_MARKOV_STATES = 50


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
    number = float(value)
    if not math.isfinite(number):
        raise ComputeError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComputeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _probabilities(value: Any, name: str, size: int) -> np.ndarray:
    raw = np.asarray([_finite(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name))], dtype=float)
    if raw.shape != (size,):
        raise ComputeError(f"{name} must contain exactly {size} values")
    if np.any(raw < 0) or raw.sum() <= 0:
        raise ComputeError(f"{name} must contain non-negative values with a positive sum")
    return raw / raw.sum()


def _sample_duration(rng: np.random.Generator, spec: Mapping[str, Any], name: str) -> float:
    distribution = str(spec.get("distribution") or "constant")
    if distribution == "constant":
        value = _finite(spec.get("value"), f"{name}.value")
    elif distribution == "uniform":
        low = _finite(spec.get("minimum"), f"{name}.minimum")
        high = _finite(spec.get("maximum"), f"{name}.maximum")
        if not low < high:
            raise ComputeError(f"{name}: minimum must be lower than maximum")
        value = float(rng.uniform(low, high))
    elif distribution == "exponential":
        mean = _finite(spec.get("mean"), f"{name}.mean")
        if mean <= 0:
            raise ComputeError(f"{name}.mean must be positive")
        value = float(rng.exponential(mean))
    elif distribution == "triangular":
        low = _finite(spec.get("minimum"), f"{name}.minimum")
        mode = _finite(spec.get("mode"), f"{name}.mode")
        high = _finite(spec.get("maximum"), f"{name}.maximum")
        if not low <= mode <= high or low == high:
            raise ComputeError(f"{name}: minimum <= mode <= maximum is required")
        value = float(rng.triangular(low, mode, high))
    else:
        raise ComputeError(f"{name}.distribution must be constant, uniform, exponential, or triangular")
    if value < 0:
        raise ComputeError(f"{name} sampled a negative duration")
    return value


def discrete_event_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import simpy

    seed = _integer(inputs.get("seed"), "inputs.seed", 0, 2**32 - 1)
    entities = _integer(inputs.get("entities", 1000), "inputs.entities", 1, MAX_ENTITIES)
    arrival = _mapping(inputs.get("arrival"), "inputs.arrival")
    stages_raw = _sequence(inputs.get("stages"), "inputs.stages")
    if not stages_raw or len(stages_raw) > MAX_STAGES:
        raise ComputeError(f"inputs.stages must contain 1 to {MAX_STAGES} stages")

    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    resources: list[Any] = []
    stages: list[dict[str, Any]] = []
    stage_names: set[str] = set()
    for index, raw in enumerate(stages_raw):
        stage = _mapping(raw, f"inputs.stages[{index}]")
        allowed = {"name", "capacity", "service"}
        unexpected = sorted(set(stage) - allowed)
        if unexpected:
            raise ComputeError(f"stage contains unsupported fields: {unexpected}")
        name = str(stage.get("name") or "")
        if not name or name in stage_names:
            raise ComputeError("stage names must be non-empty and unique")
        stage_names.add(name)
        capacity = _integer(stage.get("capacity"), f"stage[{name}].capacity", 1, 1000)
        service = dict(_mapping(stage.get("service"), f"stage[{name}].service"))
        resources.append(simpy.Resource(env, capacity=capacity))
        stages.append({"name": name, "capacity": capacity, "service": service})

    waits: list[list[float]] = [[] for _ in stages]
    service_totals = [0.0 for _ in stages]
    max_queues = [0 for _ in stages]
    cycle_times: list[float] = []

    def entity_process(entity_id: int, created_at: float):
        del entity_id
        for index, (stage, resource) in enumerate(zip(stages, resources, strict=True)):
            max_queues[index] = max(max_queues[index], len(resource.queue))
            queued_at = env.now
            with resource.request() as request:
                yield request
                wait = float(env.now - queued_at)
                waits[index].append(wait)
                duration = _sample_duration(rng, stage["service"], f"stage[{stage['name']}].service")
                service_totals[index] += duration
                yield env.timeout(duration)
        cycle_times.append(float(env.now - created_at))

    def source():
        for entity_id in range(entities):
            created = float(env.now)
            env.process(entity_process(entity_id, created))
            if entity_id + 1 < entities:
                yield env.timeout(_sample_duration(rng, arrival, "inputs.arrival"))

    env.process(source())
    env.run()
    total_time = float(env.now)
    rows = []
    for index, stage in enumerate(stages):
        array = np.asarray(waits[index], dtype=float)
        utilization = 0.0 if total_time <= 0 else min(1.0, service_totals[index] / (stage["capacity"] * total_time))
        rows.append(
            {
                "name": stage["name"],
                "capacity": stage["capacity"],
                "mean_wait": float(np.mean(array)) if array.size else 0.0,
                "p50_wait": float(np.percentile(array, 50)) if array.size else 0.0,
                "p95_wait": float(np.percentile(array, 95)) if array.size else 0.0,
                "maximum_wait": float(np.max(array)) if array.size else 0.0,
                "maximum_queue_observed": max_queues[index],
                "utilization": utilization,
            }
        )
    cycle = np.asarray(cycle_times, dtype=float)
    bottleneck = max(rows, key=lambda row: (row["mean_wait"], row["utilization"]))["name"]
    return {
        "engine": {"name": "simpy", "version": simpy.__version__},
        "seed": seed,
        "entities_completed": len(cycle_times),
        "simulation_time": total_time,
        "throughput_per_time_unit": 0.0 if total_time <= 0 else len(cycle_times) / total_time,
        "mean_cycle_time": float(np.mean(cycle)) if cycle.size else 0.0,
        "p95_cycle_time": float(np.percentile(cycle, 95)) if cycle.size else 0.0,
        "stages": rows,
        "bottleneck_stage": bottleneck,
    }


def _payoff_matrix(value: Any, name: str) -> np.ndarray:
    rows = _sequence(value, name)
    if not rows or len(rows) > MAX_STRATEGIES:
        raise ComputeError(f"{name} must contain 1 to {MAX_STRATEGIES} rows")
    matrix = np.asarray(
        [[_finite(item, f"{name}[{i}][{j}]") for j, item in enumerate(_sequence(row, f"{name}[{i}]"))] for i, row in enumerate(rows)],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[1] < 1 or matrix.shape[1] > MAX_STRATEGIES:
        raise ComputeError(f"{name} must be a non-empty matrix within strategy limits")
    return matrix


def _policy_action(
    rng: np.random.Generator,
    policy: Mapping[str, Any],
    own_payoffs: np.ndarray,
    opponent_counts: np.ndarray,
    fixed_size: int,
) -> int:
    kind = str(policy.get("type") or "fixed")
    if kind == "fixed":
        probabilities = _probabilities(policy.get("probabilities"), "policy.probabilities", fixed_size)
        return int(rng.choice(fixed_size, p=probabilities))
    if kind == "epsilon_best_response":
        epsilon = _finite(policy.get("epsilon", 0.05), "policy.epsilon")
        if not 0 <= epsilon <= 1:
            raise ComputeError("policy.epsilon must be between 0 and 1")
        if rng.random() < epsilon:
            return int(rng.integers(0, fixed_size))
        opponent_distribution = (opponent_counts + 1.0) / (opponent_counts.sum() + opponent_counts.size)
        expected = own_payoffs @ opponent_distribution
        best = np.flatnonzero(np.isclose(expected, np.max(expected)))
        return int(rng.choice(best))
    raise ComputeError("policy.type must be fixed or epsilon_best_response")


def repeated_game(inputs: Mapping[str, Any]) -> dict[str, Any]:
    seed = _integer(inputs.get("seed"), "inputs.seed", 0, 2**32 - 1)
    rounds = _integer(inputs.get("rounds", 100), "inputs.rounds", 1, MAX_ROUNDS)
    trials = _integer(inputs.get("trials", 100), "inputs.trials", 1, MAX_TRIALS)
    if rounds * trials > 500_000:
        raise ComputeError("rounds * trials cannot exceed 500000")
    red = _payoff_matrix(inputs.get("red_payoffs"), "inputs.red_payoffs")
    blue = _payoff_matrix(inputs.get("blue_payoffs"), "inputs.blue_payoffs")
    if red.shape != blue.shape:
        raise ComputeError("red_payoffs and blue_payoffs must have identical shapes")
    red_policy = _mapping(inputs.get("red_policy"), "inputs.red_policy")
    blue_policy = _mapping(inputs.get("blue_policy"), "inputs.blue_policy")
    rng = np.random.default_rng(seed)
    red_totals: list[float] = []
    blue_totals: list[float] = []
    red_frequency = np.zeros(red.shape[0], dtype=int)
    blue_frequency = np.zeros(red.shape[1], dtype=int)
    red_round_wins = blue_round_wins = draws = 0
    for _ in range(trials):
        red_counts_seen = np.zeros(red.shape[0], dtype=float)
        blue_counts_seen = np.zeros(red.shape[1], dtype=float)
        red_total = blue_total = 0.0
        for _round in range(rounds):
            red_action = _policy_action(rng, red_policy, red, blue_counts_seen, red.shape[0])
            blue_action = _policy_action(rng, blue_policy, blue.T, red_counts_seen, red.shape[1])
            red_value = float(red[red_action, blue_action])
            blue_value = float(blue[red_action, blue_action])
            red_total += red_value
            blue_total += blue_value
            red_frequency[red_action] += 1
            blue_frequency[blue_action] += 1
            red_counts_seen[red_action] += 1
            blue_counts_seen[blue_action] += 1
            if red_value > blue_value:
                red_round_wins += 1
            elif blue_value > red_value:
                blue_round_wins += 1
            else:
                draws += 1
        red_totals.append(red_total)
        blue_totals.append(blue_total)
    total_rounds = rounds * trials
    return {
        "seed": seed,
        "rounds": rounds,
        "trials": trials,
        "red_mean_total_payoff": float(np.mean(red_totals)),
        "blue_mean_total_payoff": float(np.mean(blue_totals)),
        "red_total_payoff_p10": float(np.percentile(red_totals, 10)),
        "red_total_payoff_p90": float(np.percentile(red_totals, 90)),
        "blue_total_payoff_p10": float(np.percentile(blue_totals, 10)),
        "blue_total_payoff_p90": float(np.percentile(blue_totals, 90)),
        "red_round_win_rate": red_round_wins / total_rounds,
        "blue_round_win_rate": blue_round_wins / total_rounds,
        "draw_rate": draws / total_rounds,
        "red_action_frequencies": (red_frequency / total_rounds).tolist(),
        "blue_action_frequencies": (blue_frequency / total_rounds).tolist(),
    }


def agent_evolution(inputs: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _payoff_matrix(inputs.get("payoff_matrix"), "inputs.payoff_matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ComputeError("payoff_matrix must be square")
    size = matrix.shape[0]
    shares = _probabilities(inputs.get("initial_shares"), "inputs.initial_shares", size)
    generations = _integer(inputs.get("generations", 100), "inputs.generations", 1, MAX_GENERATIONS)
    mutation = _finite(inputs.get("mutation_rate", 0.01), "inputs.mutation_rate")
    strength = _finite(inputs.get("selection_strength", 1.0), "inputs.selection_strength")
    if not 0 <= mutation <= 0.5:
        raise ComputeError("mutation_rate must be between 0 and 0.5")
    if not 0 <= strength <= 20:
        raise ComputeError("selection_strength must be between 0 and 20")
    sample_every = max(1, generations // 100)
    history = [{"generation": 0, "shares": shares.tolist()}]
    for generation in range(1, generations + 1):
        fitness = matrix @ shares
        centered = fitness - float(np.dot(shares, fitness))
        weights = shares * np.exp(np.clip(strength * centered, -50, 50))
        if weights.sum() <= 0 or not np.isfinite(weights).all():
            raise ComputeError("agent evolution became numerically unstable")
        selected = weights / weights.sum()
        shares = (1 - mutation) * selected + mutation * np.full(size, 1.0 / size)
        shares /= shares.sum()
        if generation % sample_every == 0 or generation == generations:
            history.append({"generation": generation, "shares": shares.tolist()})
    return {
        "generations": generations,
        "final_shares": shares.tolist(),
        "dominant_strategy_index": int(np.argmax(shares)),
        "dominant_share": float(np.max(shares)),
        "history": history,
    }


def _forecast_one(method: str, history: Sequence[float], parameter: float | int | None = None) -> float:
    array = np.asarray(history, dtype=float)
    if method == "naive_last":
        return float(array[-1])
    if method == "drift":
        return float(array[-1] + (array[-1] - array[0]) / max(1, len(array) - 1))
    if method == "moving_average":
        window = int(parameter or 3)
        return float(np.mean(array[-min(window, len(array)):]))
    if method == "exponential_smoothing":
        alpha = float(parameter or 0.5)
        level = float(array[0])
        for value in array[1:]:
            level = alpha * float(value) + (1 - alpha) * level
        return level
    raise ComputeError(f"unknown forecast method: {method}")


def time_series_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    data = np.asarray([_finite(item, f"inputs.data[{index}]") for index, item in enumerate(_sequence(inputs.get("data"), "inputs.data"))], dtype=float)
    if not 5 <= data.size <= MAX_SERIES:
        raise ComputeError(f"inputs.data must contain 5 to {MAX_SERIES} values")
    horizon = _integer(inputs.get("horizon", 1), "inputs.horizon", 1, MAX_FORECAST_HORIZON)
    holdout = _integer(inputs.get("holdout", min(20, max(1, data.size // 5))), "inputs.holdout", 1, data.size - 2)
    candidates: list[tuple[str, float | int | None]] = [("naive_last", None), ("drift", None)]
    for window in (3, 5, 7):
        if window < data.size - holdout:
            candidates.append(("moving_average", window))
    candidates.extend(("exponential_smoothing", alpha) for alpha in (0.2, 0.5, 0.8))
    scores = []
    train_end = data.size - holdout
    for method, parameter in candidates:
        predictions = []
        actual = []
        for index in range(train_end, data.size):
            predictions.append(_forecast_one(method, data[:index], parameter))
            actual.append(float(data[index]))
        residuals = np.asarray(actual) - np.asarray(predictions)
        scores.append(
            {
                "method": method,
                "parameter": parameter,
                "mae": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "residual_standard_deviation": float(np.std(residuals, ddof=0)),
            }
        )
    scores.sort(key=lambda row: (row["mae"], row["rmse"], row["method"]))
    selected = scores[0]
    history = data.tolist()
    forecasts = []
    lower = []
    upper = []
    residual_std = max(selected["residual_standard_deviation"], 0.0)
    for step in range(1, horizon + 1):
        value = _forecast_one(selected["method"], history, selected["parameter"])
        history.append(value)
        forecasts.append(value)
        margin = 1.96 * residual_std * math.sqrt(step)
        lower.append(value - margin)
        upper.append(value + margin)
    slope, intercept, r_value, p_value, _stderr = stats.linregress(np.arange(data.size), data)
    del intercept
    return {
        "selected_method": selected,
        "candidate_scores": scores,
        "forecast": forecasts,
        "prediction_interval_95": {"lower": lower, "upper": upper},
        "trend": {"slope_per_period": float(slope), "r_squared": float(r_value**2), "p_value": float(p_value)},
        "holdout_points": holdout,
    }


def causal_screening(inputs: Mapping[str, Any]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        array = np.asarray([_finite(item, f"inputs.{key}[{index}]") for index, item in enumerate(_sequence(inputs.get(key), f"inputs.{key}"))], dtype=float)
        if not 3 <= array.size <= MAX_SERIES:
            raise ComputeError(f"inputs.{key} must contain 3 to {MAX_SERIES} values")
        return array

    treated_pre = values("treated_pre")
    treated_post = values("treated_post")
    control_pre = values("control_pre")
    control_post = values("control_post")
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    bootstrap = _integer(inputs.get("bootstrap", 1000), "inputs.bootstrap", 100, 5000)
    effect = (float(np.mean(treated_post)) - float(np.mean(treated_pre))) - (
        float(np.mean(control_post)) - float(np.mean(control_pre))
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap, dtype=float)
    for index in range(bootstrap):
        tp = rng.choice(treated_pre, treated_pre.size, replace=True)
        tpost = rng.choice(treated_post, treated_post.size, replace=True)
        cp = rng.choice(control_pre, control_pre.size, replace=True)
        cpost = rng.choice(control_post, control_post.size, replace=True)
        boot[index] = (np.mean(tpost) - np.mean(tp)) - (np.mean(cpost) - np.mean(cp))
    lower, upper = np.percentile(boot, [2.5, 97.5])
    treated_slope = stats.linregress(np.arange(treated_pre.size), treated_pre).slope
    control_slope = stats.linregress(np.arange(control_pre.size), control_pre).slope
    pretrend_gap = float(abs(treated_slope - control_slope))
    scale = max(float(np.std(np.concatenate([treated_pre, control_pre]))), 1e-12)
    parallel_score = pretrend_gap / scale
    if min(treated_pre.size, treated_post.size, control_pre.size, control_post.size) < 5:
        status = "INSUFFICIENT_DATA"
    elif parallel_score > 0.5:
        status = "CONFLICT"
    elif lower > 0 or upper < 0:
        status = "SUPPORTED"
    else:
        status = "WEAK"
    return {
        "method": "difference_in_differences_screening",
        "effect_estimate": effect,
        "bootstrap_interval_95": [float(lower), float(upper)],
        "pretrend_gap": pretrend_gap,
        "pretrend_gap_standardized": parallel_score,
        "status": status,
        "warning": "This is a screening result, not proof of causality.",
    }


def nonlinear_dynamics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    model = str(inputs.get("model") or "")
    initial = np.asarray([_finite(item, f"inputs.initial_state[{index}]") for index, item in enumerate(_sequence(inputs.get("initial_state"), "inputs.initial_state"))], dtype=float)
    duration = _finite(inputs.get("duration"), "inputs.duration")
    steps = _integer(inputs.get("steps", 200), "inputs.steps", 2, MAX_ODE_STEPS)
    if duration <= 0:
        raise ComputeError("inputs.duration must be positive")
    parameters = _mapping(inputs.get("parameters"), "inputs.parameters")

    if model == "logistic":
        if initial.size != 1:
            raise ComputeError("logistic model requires one initial state")
        growth = _finite(parameters.get("growth_rate"), "parameters.growth_rate")
        capacity = _finite(parameters.get("carrying_capacity"), "parameters.carrying_capacity")
        if capacity <= 0:
            raise ComputeError("carrying_capacity must be positive")

        def derivative(_time: float, state: np.ndarray) -> list[float]:
            return [growth * state[0] * (1 - state[0] / capacity)]

    elif model == "lotka_volterra":
        if initial.size != 2:
            raise ComputeError("lotka_volterra model requires two initial states")
        alpha = _finite(parameters.get("alpha"), "parameters.alpha")
        beta = _finite(parameters.get("beta"), "parameters.beta")
        delta = _finite(parameters.get("delta"), "parameters.delta")
        gamma = _finite(parameters.get("gamma"), "parameters.gamma")

        def derivative(_time: float, state: np.ndarray) -> list[float]:
            prey, predator = state
            return [alpha * prey - beta * prey * predator, delta * prey * predator - gamma * predator]

    elif model == "sir":
        if initial.size != 3:
            raise ComputeError("sir model requires S, I, R initial states")
        beta = _finite(parameters.get("beta"), "parameters.beta")
        gamma = _finite(parameters.get("gamma"), "parameters.gamma")
        population = float(np.sum(initial))
        if population <= 0:
            raise ComputeError("SIR population must be positive")

        def derivative(_time: float, state: np.ndarray) -> list[float]:
            susceptible, infected, recovered = state
            return [
                -beta * susceptible * infected / population,
                beta * susceptible * infected / population - gamma * infected,
                gamma * infected,
            ]

    else:
        raise ComputeError("inputs.model must be logistic, lotka_volterra, or sir")

    times = np.linspace(0.0, duration, steps)
    solution = solve_ivp(derivative, (0.0, duration), initial, t_eval=times, rtol=1e-7, atol=1e-9)
    if not solution.success or not np.isfinite(solution.y).all():
        raise ComputeError(f"nonlinear integration failed: {solution.message}")
    trajectory = solution.y.T
    return {
        "model": model,
        "final_state": trajectory[-1].tolist(),
        "minimum_state": np.min(trajectory, axis=0).tolist(),
        "maximum_state": np.max(trajectory, axis=0).tolist(),
        "trajectory": [{"time": float(time), "state": state.tolist()} for time, state in zip(times, trajectory, strict=True)],
        "solver_message": str(solution.message),
    }


def pattern_discovery(inputs: Mapping[str, Any]) -> dict[str, Any]:
    data = np.asarray([_finite(item, f"inputs.data[{index}]") for index, item in enumerate(_sequence(inputs.get("data"), "inputs.data"))], dtype=float)
    if not 10 <= data.size <= MAX_SERIES:
        raise ComputeError(f"inputs.data must contain 10 to {MAX_SERIES} values")
    slope, _intercept, r_value, p_value, _stderr = stats.linregress(np.arange(data.size), data)
    max_lag = min(20, data.size // 3)
    autocorrelations = []
    centered = data - np.mean(data)
    variance = float(np.dot(centered, centered))
    for lag in range(1, max_lag + 1):
        value = 0.0 if variance == 0 else float(np.dot(centered[:-lag], centered[lag:]) / variance)
        autocorrelations.append({"lag": lag, "correlation": value})
    min_segment = max(3, data.size // 10)
    best_index = None
    best_change = 0.0
    for index in range(min_segment, data.size - min_segment + 1):
        change = float(np.mean(data[index:]) - np.mean(data[:index]))
        if abs(change) > abs(best_change):
            best_change = change
            best_index = index
    std = float(np.std(data, ddof=0))
    z_scores = np.zeros_like(data) if std == 0 else (data - np.mean(data)) / std
    outliers = [int(index) for index in np.flatnonzero(np.abs(z_scores) >= 3)]
    periodic = max(autocorrelations, key=lambda row: abs(row["correlation"])) if autocorrelations else {"lag": None, "correlation": 0.0}
    return {
        "trend": {"slope_per_period": float(slope), "r_squared": float(r_value**2), "p_value": float(p_value)},
        "autocorrelations": autocorrelations,
        "strongest_period_candidate": periodic,
        "change_point_candidate": {"index": best_index, "mean_shift": best_change},
        "outlier_indices_z3": outliers,
        "summary": {
            "mean": float(np.mean(data)),
            "standard_deviation": std,
            "coefficient_of_variation": None if np.mean(data) == 0 else abs(std / float(np.mean(data))),
        },
        "warning": "Patterns are candidates requiring external validation; correlation is not causation.",
    }


def assumption_validation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    data = np.asarray([_finite(item, f"inputs.data[{index}]") for index, item in enumerate(_sequence(inputs.get("data"), "inputs.data"))], dtype=float)
    if not 3 <= data.size <= MAX_SERIES:
        raise ComputeError(f"inputs.data must contain 3 to {MAX_SERIES} values")
    checks = []
    conflict = False
    weak = False
    sensitive = False
    expected_min = inputs.get("expected_minimum")
    expected_max = inputs.get("expected_maximum")
    if expected_min is not None or expected_max is not None:
        low = -math.inf if expected_min is None else _finite(expected_min, "inputs.expected_minimum")
        high = math.inf if expected_max is None else _finite(expected_max, "inputs.expected_maximum")
        if low > high:
            raise ComputeError("expected_minimum cannot exceed expected_maximum")
        outside = float(np.mean((data < low) | (data > high)))
        checks.append({"name": "expected_range", "outside_fraction": outside, "pass": outside <= 0.05})
        conflict = conflict or outside > 0.20
        weak = weak or outside > 0.05
    if "expected_mean" in inputs:
        expected_mean = _finite(inputs.get("expected_mean"), "inputs.expected_mean")
        tolerance = _finite(inputs.get("mean_tolerance", max(float(np.std(data)), 1e-9)), "inputs.mean_tolerance")
        if tolerance <= 0:
            raise ComputeError("mean_tolerance must be positive")
        deviation = abs(float(np.mean(data)) - expected_mean)
        checks.append({"name": "expected_mean", "deviation": deviation, "tolerance": tolerance, "pass": deviation <= tolerance})
        conflict = conflict or deviation > 2 * tolerance
        weak = weak or deviation > tolerance
        sensitive = sensitive or 0.8 * tolerance <= deviation <= 1.2 * tolerance
    distribution = inputs.get("expected_distribution")
    if distribution is not None:
        distribution = str(distribution)
        if data.size < 8:
            checks.append({"name": "distribution_fit", "status": "INSUFFICIENT_DATA"})
            weak = True
        elif distribution == "normal":
            mean = float(np.mean(data))
            std = float(np.std(data, ddof=0))
            p_value = 0.0 if std == 0 else float(stats.kstest(data, stats.norm(loc=mean, scale=std).cdf).pvalue)
            checks.append({"name": "distribution_fit", "distribution": "normal", "p_value": p_value, "pass": p_value >= 0.05})
            weak = weak or p_value < 0.05
            conflict = conflict or p_value < 0.001
        elif distribution == "uniform":
            low = float(np.min(data))
            width = float(np.max(data) - low)
            p_value = 0.0 if width == 0 else float(stats.kstest(data, stats.uniform(loc=low, scale=width).cdf).pvalue)
            checks.append({"name": "distribution_fit", "distribution": "uniform", "p_value": p_value, "pass": p_value >= 0.05})
            weak = weak or p_value < 0.05
            conflict = conflict or p_value < 0.001
        else:
            raise ComputeError("expected_distribution must be normal or uniform")
    if data.size < 8:
        status = "INSUFFICIENT_DATA"
    elif conflict:
        status = "CONFLICT"
    elif sensitive:
        status = "HIGHLY_SENSITIVE"
    elif weak:
        status = "WEAK"
    else:
        status = "PASS"
    return {
        "status": status,
        "checks": checks,
        "observed": {
            "count": int(data.size),
            "minimum": float(np.min(data)),
            "maximum": float(np.max(data)),
            "mean": float(np.mean(data)),
            "standard_deviation": float(np.std(data, ddof=0)),
        },
        "warning": "A PASS means consistent with supplied data, not proven true in reality.",
    }


def markov_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(
        [[_finite(item, f"inputs.transition_matrix[{i}][{j}]") for j, item in enumerate(_sequence(row, f"inputs.transition_matrix[{i}]"))] for i, row in enumerate(_sequence(inputs.get("transition_matrix"), "inputs.transition_matrix"))],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not 1 <= matrix.shape[0] <= MAX_MARKOV_STATES:
        raise ComputeError(f"transition_matrix must be square with 1 to {MAX_MARKOV_STATES} states")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ComputeError("each transition_matrix row must contain non-negative probabilities summing to 1")
    initial = _probabilities(inputs.get("initial_distribution"), "inputs.initial_distribution", matrix.shape[0])
    steps = _integer(inputs.get("steps", 10), "inputs.steps", 1, 100_000)
    rewards_raw = inputs.get("state_rewards")
    rewards = None if rewards_raw is None else np.asarray([_finite(item, f"inputs.state_rewards[{i}]") for i, item in enumerate(_sequence(rewards_raw, "inputs.state_rewards"))], dtype=float)
    if rewards is not None and rewards.shape != initial.shape:
        raise ComputeError("state_rewards must match the number of states")
    distribution = initial.copy()
    trajectory = [{"step": 0, "distribution": distribution.tolist()}]
    total_reward = 0.0
    sample_every = max(1, steps // 100)
    for step in range(1, steps + 1):
        if rewards is not None:
            total_reward += float(np.dot(distribution, rewards))
        distribution = distribution @ matrix
        if step % sample_every == 0 or step == steps:
            trajectory.append({"step": step, "distribution": distribution.tolist()})
    steady = np.full_like(initial, 1.0 / initial.size)
    for _ in range(10_000):
        updated = steady @ matrix
        if np.max(np.abs(updated - steady)) < 1e-12:
            steady = updated
            break
        steady = updated
    return {
        "steps": steps,
        "final_distribution": distribution.tolist(),
        "approximate_stationary_distribution": steady.tolist(),
        "expected_cumulative_reward": total_reward if rewards is not None else None,
        "trajectory": trajectory,
    }


OPERATIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "discrete_event_simulation": discrete_event_simulation,
    "repeated_game": repeated_game,
    "agent_evolution": agent_evolution,
    "time_series_forecast": time_series_forecast,
    "causal_screening": causal_screening,
    "nonlinear_dynamics": nonlinear_dynamics,
    "pattern_discovery": pattern_discovery,
    "assumption_validation": assumption_validation,
    "markov_simulation": markov_simulation,
}
