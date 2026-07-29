"""Bounded allowlisted agent-based simulations implemented with Mesa.

The module never accepts Python code, imports ticket-supplied modules, accesses the
network, or installs packages at runtime. Mesa is an optional pinned engine loaded
only when the allowlisted operation is executed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from compute_runner import ComputeError

MESA_VERSION = "3.5.1"
MAX_AGENTS = 5_000
MAX_STEPS = 1_000
MAX_OPTIONS = 20
MAX_RESOURCES = 20
MAX_EDGES = 100_000


def _mesa():
    try:
        import mesa  # type: ignore
    except ImportError as exc:
        raise ComputeError(
            "Mesa optional engine is not installed; install compute-center/requirements-mesa.txt"
        ) from exc
    if getattr(mesa, "__version__", None) != MESA_VERSION:
        raise ComputeError(
            f"Mesa version must be exactly {MESA_VERSION}; found {getattr(mesa, '__version__', 'unknown')}"
        )
    return mesa


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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComputeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ComputeError(f"{name} must be between 0 and 1")
    return result


def _softmax(values: np.ndarray, sensitivity: float) -> np.ndarray:
    shifted = sensitivity * (values - float(np.max(values)))
    weights = np.exp(np.clip(shifted, -60, 60))
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ComputeError("agent choice probabilities became numerically unstable")
    return weights / total


def _history_append(history: list[dict[str, Any]], step: int, steps: int, row: dict[str, Any]) -> None:
    sample_every = max(1, steps // 100)
    if step == 0 or step == steps or step % sample_every == 0:
        history.append({"step": step, **row})


def _engine(mode: str) -> dict[str, Any]:
    return {
        "name": "mesa",
        "version": MESA_VERSION,
        "mode": mode,
        "dependency_scope": "optional-isolated",
        "arbitrary_agent_code": False,
        "network_used": False,
    }


def _heterogeneous_worker_choice(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mesa = _mesa()
    agent_count = _integer(inputs.get("agent_count", 500), "inputs.agent_count", 2, MAX_AGENTS)
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    learning_rate = _probability(inputs.get("learning_rate", 0.2), "inputs.learning_rate")
    choice_sensitivity = _finite(inputs.get("choice_sensitivity", 2.0), "inputs.choice_sensitivity")
    if not 0 < choice_sensitivity <= 20:
        raise ComputeError("inputs.choice_sensitivity must be greater than 0 and at most 20")
    switching_cost = _finite(inputs.get("switching_cost", 0.0), "inputs.switching_cost")
    preference_sd = _finite(inputs.get("preference_standard_deviation", 1.0), "inputs.preference_standard_deviation")
    reward_sd = _finite(inputs.get("reward_standard_deviation", 0.0), "inputs.reward_standard_deviation")
    if min(switching_cost, preference_sd, reward_sd) < 0:
        raise ComputeError("switching_cost and standard deviations must be non-negative")

    raw_options = _sequence(inputs.get("options"), "inputs.options")
    if not 2 <= len(raw_options) <= MAX_OPTIONS:
        raise ComputeError(f"inputs.options must contain 2 to {MAX_OPTIONS} entries")
    options: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_options):
        row = _mapping(raw, f"inputs.options[{index}]")
        allowed = {"name", "base_reward", "cost", "risk_cost", "capacity", "congestion_penalty"}
        unexpected = sorted(set(row) - allowed)
        if unexpected:
            raise ComputeError(f"inputs.options[{index}] contains unsupported fields: {unexpected}")
        name = str(row.get("name") or "").strip()
        if not name or name in names:
            raise ComputeError("option names must be non-empty and unique")
        names.add(name)
        capacity = _integer(row.get("capacity", agent_count), f"inputs.options[{index}].capacity", 1, MAX_AGENTS)
        congestion_penalty = _finite(row.get("congestion_penalty", 0.0), f"inputs.options[{index}].congestion_penalty")
        if congestion_penalty < 0:
            raise ComputeError("option congestion_penalty must be non-negative")
        options.append(
            {
                "name": name,
                "base_reward": _finite(row.get("base_reward"), f"inputs.options[{index}].base_reward"),
                "cost": _finite(row.get("cost", 0.0), f"inputs.options[{index}].cost"),
                "risk_cost": _finite(row.get("risk_cost", 0.0), f"inputs.options[{index}].risk_cost"),
                "capacity": capacity,
                "congestion_penalty": congestion_penalty,
            }
        )

    initial = inputs.get("initial_shares")
    if initial is None:
        initial_shares = np.full(len(options), 1 / len(options), dtype=float)
    else:
        values = np.asarray([_finite(v, f"inputs.initial_shares[{i}]") for i, v in enumerate(_sequence(initial, "inputs.initial_shares"))], dtype=float)
        if values.size != len(options) or np.any(values < 0) or float(values.sum()) <= 0:
            raise ComputeError("inputs.initial_shares must match options and contain non-negative positive-total values")
        initial_shares = values / values.sum()

    class Worker(mesa.Agent):
        def __init__(self, model, option_index: int, preferences: np.ndarray):
            super().__init__(model)
            self.option_index = option_index
            self.preferences = preferences
            self.expected = model.base_utilities + preferences
            self.total_reward = 0.0
            self.switches = 0

        def step(self) -> None:
            occupancy = self.model.occupancy
            penalties = np.asarray(
                [
                    max(0.0, occupancy[i] - option["capacity"]) / option["capacity"] * option["congestion_penalty"]
                    for i, option in enumerate(options)
                ],
                dtype=float,
            )
            utilities = self.expected - penalties
            utilities = utilities.copy()
            utilities[np.arange(len(options)) != self.option_index] -= switching_cost
            probabilities = _softmax(utilities, choice_sensitivity)
            selected = int(self.model.rng.choice(len(options), p=probabilities))
            if selected != self.option_index:
                self.switches += 1
            self.option_index = selected
            option = options[selected]
            reward = option["base_reward"] - option["cost"] - option["risk_cost"] - penalties[selected]
            if reward_sd > 0:
                reward += float(self.model.rng.normal(0.0, reward_sd))
            self.total_reward += reward
            self.expected[selected] = (1 - learning_rate) * self.expected[selected] + learning_rate * reward
            self.model.period_rewards.append(reward)

    class WorkerModel(mesa.Model):
        def __init__(self) -> None:
            super().__init__(rng=seed)
            self.base_utilities = np.asarray(
                [row["base_reward"] - row["cost"] - row["risk_cost"] for row in options], dtype=float
            )
            self.period_rewards: list[float] = []
            choices = self.rng.choice(len(options), size=agent_count, p=initial_shares)
            preferences = self.rng.normal(0.0, preference_sd, size=(agent_count, len(options)))
            for index in range(agent_count):
                Worker(self, int(choices[index]), preferences[index])

        @property
        def occupancy(self) -> np.ndarray:
            counts = np.zeros(len(options), dtype=int)
            for agent in self.agents:
                counts[agent.option_index] += 1
            return counts

        def step(self) -> None:
            self.period_rewards = []
            self.agents.shuffle_do("step")

    model = WorkerModel()
    history: list[dict[str, Any]] = []
    counts = model.occupancy
    _history_append(history, 0, steps, {"shares": (counts / agent_count).tolist(), "mean_reward": None})
    for step in range(1, steps + 1):
        model.step()
        counts = model.occupancy
        _history_append(
            history,
            step,
            steps,
            {
                "shares": (counts / agent_count).tolist(),
                "mean_reward": float(np.mean(model.period_rewards)) if model.period_rewards else 0.0,
            },
        )
    rewards = np.asarray([agent.total_reward for agent in model.agents], dtype=float)
    switches = int(sum(agent.switches for agent in model.agents))
    return {
        "engine": _engine("heterogeneous_worker_choice"),
        "agent_count": agent_count,
        "steps": steps,
        "seed": seed,
        "option_names": [row["name"] for row in options],
        "final_shares": (model.occupancy / agent_count).tolist(),
        "cumulative_reward": {
            "mean_per_agent": float(np.mean(rewards)),
            "median_per_agent": float(np.median(rewards)),
            "minimum": float(np.min(rewards)),
            "maximum": float(np.max(rewards)),
        },
        "switches": switches,
        "history": history,
    }


def _network_contagion(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mesa = _mesa()
    agent_count = _integer(inputs.get("agent_count", 500), "inputs.agent_count", 2, MAX_AGENTS)
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    average_degree = _integer(
        inputs.get("average_degree", min(6, agent_count - 1)),
        "inputs.average_degree",
        1,
        min(50, agent_count - 1),
    )
    initial_rate = _probability(inputs.get("initial_adoption_rate", 0.05), "inputs.initial_adoption_rate")
    threshold_mean = _probability(inputs.get("threshold_mean", 0.35), "inputs.threshold_mean")
    threshold_sd = _finite(inputs.get("threshold_standard_deviation", 0.1), "inputs.threshold_standard_deviation")
    external = _probability(inputs.get("external_influence", 0.0), "inputs.external_influence")
    recovery = _probability(inputs.get("recovery_rate", 0.0), "inputs.recovery_rate")
    if threshold_sd < 0 or threshold_sd > 1:
        raise ComputeError("inputs.threshold_standard_deviation must be between 0 and 1")

    target_edges = min(MAX_EDGES, agent_count * average_degree // 2)

    class ContagionAgent(mesa.Agent):
        def __init__(self, model, adopted: bool, threshold: float):
            super().__init__(model)
            self.adopted = adopted
            self.threshold = threshold
            self.next_adopted = adopted

    class ContagionModel(mesa.Model):
        def __init__(self) -> None:
            super().__init__(rng=seed)
            for _ in range(agent_count):
                ContagionAgent(
                    self,
                    bool(self.rng.random() < initial_rate),
                    float(np.clip(self.rng.normal(threshold_mean, threshold_sd), 0.0, 1.0)),
                )
            self.neighbors = [set() for _ in range(agent_count)]
            for i in range(agent_count):
                j = (i + 1) % agent_count
                self.neighbors[i].add(j)
                self.neighbors[j].add(i)
            edges = {tuple(sorted((i, j))) for i in range(agent_count) for j in self.neighbors[i] if i < j}
            attempts = 0
            while len(edges) < target_edges and attempts < max(10_000, target_edges * 20):
                a, b = self.rng.integers(0, agent_count, size=2)
                attempts += 1
                if a == b:
                    continue
                edge = tuple(sorted((int(a), int(b))))
                if edge in edges:
                    continue
                edges.add(edge)
                self.neighbors[edge[0]].add(edge[1])
                self.neighbors[edge[1]].add(edge[0])
            self.edge_count = len(edges)

        def step(self) -> None:
            agents = list(self.agents)
            adopted = np.asarray([agent.adopted for agent in agents], dtype=bool)
            for index, agent in enumerate(agents):
                neighbor_ids = self.neighbors[index]
                share = float(np.mean(adopted[list(neighbor_ids)])) if neighbor_ids else 0.0
                if agent.adopted:
                    agent.next_adopted = not (recovery > 0 and self.rng.random() < recovery)
                else:
                    agent.next_adopted = share + external >= agent.threshold
            for agent in agents:
                agent.adopted = agent.next_adopted

    model = ContagionModel()
    history: list[dict[str, Any]] = []

    def rate() -> float:
        return float(np.mean([agent.adopted for agent in model.agents]))

    initial_observed = rate()
    peak = initial_observed
    _history_append(history, 0, steps, {"adoption_rate": initial_observed})
    for step in range(1, steps + 1):
        model.step()
        current = rate()
        peak = max(peak, current)
        _history_append(history, step, steps, {"adoption_rate": current})
    degrees = np.asarray([len(row) for row in model.neighbors], dtype=float)
    return {
        "engine": _engine("network_contagion"),
        "agent_count": agent_count,
        "steps": steps,
        "seed": seed,
        "edge_count": model.edge_count,
        "mean_degree": float(np.mean(degrees)),
        "initial_adoption_rate_observed": initial_observed,
        "final_adoption_rate": rate(),
        "peak_adoption_rate": peak,
        "history": history,
    }


def _gini(values: np.ndarray) -> float:
    if values.size == 0 or np.allclose(values, 0):
        return 0.0
    shifted = values - min(0.0, float(np.min(values)))
    ordered = np.sort(shifted)
    total = float(ordered.sum())
    if total <= 0:
        return 0.0
    n = ordered.size
    return float((2 * np.dot(np.arange(1, n + 1), ordered) / (n * total)) - (n + 1) / n)


def _resource_competition(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mesa = _mesa()
    agent_count = _integer(inputs.get("agent_count", 500), "inputs.agent_count", 2, MAX_AGENTS)
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    demand_mean = _finite(inputs.get("demand_mean", 1.0), "inputs.demand_mean")
    demand_sd = _finite(inputs.get("demand_standard_deviation", 0.2), "inputs.demand_standard_deviation")
    learning_rate = _probability(inputs.get("learning_rate", 0.2), "inputs.learning_rate")
    exploration_rate = _probability(inputs.get("exploration_rate", 0.05), "inputs.exploration_rate")
    choice_sensitivity = _finite(inputs.get("choice_sensitivity", 2.0), "inputs.choice_sensitivity")
    if demand_mean < 0 or demand_sd < 0 or not 0 < choice_sensitivity <= 20:
        raise ComputeError("demand parameters must be non-negative and choice_sensitivity must be in (0, 20]")

    raw_resources = _sequence(inputs.get("resources"), "inputs.resources")
    if not 1 <= len(raw_resources) <= MAX_RESOURCES:
        raise ComputeError(f"inputs.resources must contain 1 to {MAX_RESOURCES} entries")
    resources: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_resources):
        row = _mapping(raw, f"inputs.resources[{index}]")
        allowed = {"name", "initial_stock", "capacity", "regeneration", "unit_value"}
        unexpected = sorted(set(row) - allowed)
        if unexpected:
            raise ComputeError(f"inputs.resources[{index}] contains unsupported fields: {unexpected}")
        name = str(row.get("name") or "").strip()
        if not name or name in names:
            raise ComputeError("resource names must be non-empty and unique")
        names.add(name)
        capacity = _finite(row.get("capacity"), f"inputs.resources[{index}].capacity")
        initial_stock = _finite(row.get("initial_stock", capacity), f"inputs.resources[{index}].initial_stock")
        regeneration = _finite(row.get("regeneration", 0.0), f"inputs.resources[{index}].regeneration")
        unit_value = _finite(row.get("unit_value", 1.0), f"inputs.resources[{index}].unit_value")
        if capacity <= 0 or not 0 <= initial_stock <= capacity or regeneration < 0:
            raise ComputeError("resource capacity must be positive; stock and regeneration must be valid")
        resources.append(
            {
                "name": name,
                "capacity": capacity,
                "stock": initial_stock,
                "regeneration": regeneration,
                "unit_value": unit_value,
            }
        )

    class ResourceAgent(mesa.Agent):
        def __init__(self, model):
            super().__init__(model)
            self.expected = np.asarray([row["unit_value"] for row in resources], dtype=float)
            self.total_reward = 0.0

        def step(self) -> None:
            if self.model.rng.random() < exploration_rate:
                selected = int(self.model.rng.integers(0, len(resources)))
            else:
                selected = int(self.model.rng.choice(len(resources), p=_softmax(self.expected, choice_sensitivity)))
            demand = max(0.0, float(self.model.rng.normal(demand_mean, demand_sd)))
            allocation = min(resources[selected]["stock"], demand)
            resources[selected]["stock"] -= allocation
            resources[selected]["harvested"] += allocation
            reward = allocation * resources[selected]["unit_value"]
            self.total_reward += reward
            observed = reward / max(demand, 1e-12)
            self.expected[selected] = (1 - learning_rate) * self.expected[selected] + learning_rate * observed
            self.model.period_rewards.append(reward)

    class ResourceModel(mesa.Model):
        def __init__(self) -> None:
            super().__init__(rng=seed)
            for row in resources:
                row["harvested"] = 0.0
            self.period_rewards: list[float] = []
            for _ in range(agent_count):
                ResourceAgent(self)

        def step(self) -> None:
            self.period_rewards = []
            self.agents.shuffle_do("step")
            for row in resources:
                row["stock"] = min(row["capacity"], row["stock"] + row["regeneration"])

    model = ResourceModel()
    history: list[dict[str, Any]] = []
    _history_append(history, 0, steps, {"stocks": [row["stock"] for row in resources], "mean_reward": None})
    for step in range(1, steps + 1):
        model.step()
        _history_append(
            history,
            step,
            steps,
            {
                "stocks": [row["stock"] for row in resources],
                "mean_reward": float(np.mean(model.period_rewards)) if model.period_rewards else 0.0,
            },
        )
    rewards = np.asarray([agent.total_reward for agent in model.agents], dtype=float)
    return {
        "engine": _engine("resource_competition"),
        "agent_count": agent_count,
        "steps": steps,
        "seed": seed,
        "resources": [
            {
                "name": row["name"],
                "final_stock": float(row["stock"]),
                "total_harvested": float(row["harvested"]),
                "capacity": row["capacity"],
            }
            for row in resources
        ],
        "agent_reward": {
            "mean": float(np.mean(rewards)),
            "median": float(np.median(rewards)),
            "minimum": float(np.min(rewards)),
            "maximum": float(np.max(rewards)),
            "gini": _gini(rewards),
        },
        "history": history,
    }


def agent_based_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {
        "heterogeneous_worker_choice": _heterogeneous_worker_choice,
        "network_contagion": _network_contagion,
        "resource_competition": _resource_competition,
    }
    if mode not in handlers:
        raise ComputeError(
            "inputs.mode must be one of heterogeneous_worker_choice, network_contagion, resource_competition"
        )
    return handlers[mode](inputs)


OPERATIONS = {"agent_based_simulation": agent_based_simulation}
