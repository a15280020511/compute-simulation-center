#!/usr/bin/env python3
"""Eight bounded group-level social-behavior modes implemented with Mesa.

These modes are scenario simulators, not individual psychological diagnosis tools.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

import numpy as np

from compute_runner import ComputeError

MESA_VERSION = "3.5.1"
MAX_AGENTS = 5_000
MAX_STEPS = 1_000
MODES = {
    "prospect_theory_choice",
    "bounded_rational_adoption",
    "trust_update",
    "social_norm_compliance",
    "risk_perception",
    "fatigue_and_adaptation",
    "institutional_confidence",
    "group_identity_choice",
}


def _mesa():
    try:
        import mesa
    except ImportError as exc:
        raise ComputeError("Mesa optional engine is not installed; install requirements-mesa.txt") from exc
    if version("mesa") != MESA_VERSION:
        raise ComputeError(f"Mesa version must be exactly {MESA_VERSION}")
    return mesa


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


def _mode_target(mode: str, state: float, peer: float, group_peer: float, inputs: Mapping[str, Any], rng: np.random.Generator) -> float:
    if mode == "prospect_theory_choice":
        gain = _finite(inputs.get("gain", 1.0), "inputs.gain")
        loss = _finite(inputs.get("loss", 0.5), "inputs.loss")
        reference = _finite(inputs.get("reference_point", 0.0), "inputs.reference_point")
        loss_aversion = _finite(inputs.get("loss_aversion", 2.25), "inputs.loss_aversion")
        value = max(gain - reference, 0.0) ** 0.88 - loss_aversion * max(loss - reference, 0.0) ** 0.88
        return 1 / (1 + math.exp(-float(np.clip(value + peer - 0.5, -60, 60))))
    if mode == "bounded_rational_adoption":
        benefit = _finite(inputs.get("benefit", 1.0), "inputs.benefit")
        cost = _finite(inputs.get("cost", 0.5), "inputs.cost")
        influence = _finite(inputs.get("social_influence", 1.0), "inputs.social_influence")
        attention = _probability(inputs.get("attention_probability", 0.8), "inputs.attention_probability")
        if rng.random() > attention:
            return state
        return 1 / (1 + math.exp(-float(np.clip(benefit - cost + influence * (peer - 0.5), -60, 60))))
    if mode == "trust_update":
        positive = _probability(inputs.get("positive_signal_probability", 0.7), "inputs.positive_signal_probability")
        reliability = _probability(inputs.get("signal_reliability", 0.8), "inputs.signal_reliability")
        decay = _probability(inputs.get("trust_decay", 0.02), "inputs.trust_decay")
        signal = 1.0 if rng.random() < positive else 0.0
        evidence = reliability * signal + (1 - reliability) * (1 - signal)
        return float(np.clip((1 - decay) * state + decay * 0.5 + 0.2 * (evidence - state), 0, 1))
    if mode == "social_norm_compliance":
        strength = _finite(inputs.get("norm_strength", 1.0), "inputs.norm_strength")
        enforcement = _probability(inputs.get("enforcement_probability", 0.2), "inputs.enforcement_probability")
        cost = _finite(inputs.get("compliance_cost", 0.2), "inputs.compliance_cost")
        utility = strength * (peer - 0.5) + enforcement - cost
        return 1 / (1 + math.exp(-float(np.clip(utility, -60, 60))))
    if mode == "risk_perception":
        event_probability = _probability(inputs.get("event_probability", 0.1), "inputs.event_probability")
        media = _finite(inputs.get("media_intensity", 1.0), "inputs.media_intensity")
        memory = _probability(inputs.get("memory", 0.8), "inputs.memory")
        observed = 1.0 if rng.random() < event_probability else 0.0
        return float(np.clip(memory * state + (1 - memory) * (observed * media + 0.2 * peer), 0, 1))
    if mode == "fatigue_and_adaptation":
        intensity = _probability(inputs.get("policy_intensity", 0.8), "inputs.policy_intensity")
        fatigue = _probability(inputs.get("fatigue_rate", 0.05), "inputs.fatigue_rate")
        adaptation = _probability(inputs.get("adaptation_rate", 0.02), "inputs.adaptation_rate")
        return float(np.clip(state + adaptation * intensity * peer - fatigue * state, 0, 1))
    if mode == "institutional_confidence":
        performance = _probability(inputs.get("institutional_performance", 0.7), "inputs.institutional_performance")
        transparency = _probability(inputs.get("transparency", 0.6), "inputs.transparency")
        scandal = _probability(inputs.get("scandal_probability", 0.02), "inputs.scandal_probability")
        shock = -0.5 if rng.random() < scandal else 0.0
        return float(np.clip(0.7 * state + 0.2 * performance + 0.1 * transparency + shock, 0, 1))
    if mode == "group_identity_choice":
        in_norm = _probability(inputs.get("in_group_norm", 0.8), "inputs.in_group_norm")
        out_norm = _probability(inputs.get("out_group_norm", 0.3), "inputs.out_group_norm")
        strength = _probability(inputs.get("identity_strength", 0.7), "inputs.identity_strength")
        return float(np.clip((1 - strength) * peer + strength * (0.7 * in_norm + 0.3 * out_norm) + 0.1 * (group_peer - peer), 0, 1))
    raise ComputeError(f"unsupported social behavior mode: {mode}")


def social_behavior_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode not in MODES:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(MODES))}")
    mesa = _mesa()
    agent_count = _integer(inputs.get("agent_count", 500), "inputs.agent_count", 2, MAX_AGENTS)
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    initial_mean = _probability(inputs.get("initial_mean", 0.5), "inputs.initial_mean")
    initial_sd = _finite(inputs.get("initial_standard_deviation", 0.15), "inputs.initial_standard_deviation")
    learning_rate = _probability(inputs.get("learning_rate", 0.25), "inputs.learning_rate")
    noise_sd = _finite(inputs.get("noise_standard_deviation", 0.01), "inputs.noise_standard_deviation")
    group_share = _probability(inputs.get("group_share", 0.5), "inputs.group_share")
    if initial_sd < 0 or noise_sd < 0:
        raise ComputeError("standard deviations must be non-negative")

    class SocialAgent(mesa.Agent):
        def __init__(self, model, state: float, group: int):
            super().__init__(model)
            self.state = state
            self.group = group
            self.next_state = state

        def step(self) -> None:
            peer = self.model.population_mean
            group_peer = self.model.group_means[self.group]
            target = _mode_target(mode, self.state, peer, group_peer, inputs, self.model.rng)
            noise = float(self.model.rng.normal(0, noise_sd)) if noise_sd else 0.0
            self.next_state = float(np.clip((1 - learning_rate) * self.state + learning_rate * target + noise, 0, 1))

    class SocialModel(mesa.Model):
        def __init__(self) -> None:
            super().__init__(rng=seed)
            groups = self.rng.random(agent_count) < group_share
            states = np.clip(self.rng.normal(initial_mean, initial_sd, size=agent_count), 0, 1)
            for state, group in zip(states, groups, strict=True):
                SocialAgent(self, float(state), int(group))

        @property
        def population_mean(self) -> float:
            return float(np.mean([agent.state for agent in self.agents]))

        @property
        def group_means(self) -> dict[int, float]:
            result = {}
            for group in (0, 1):
                values = [agent.state for agent in self.agents if agent.group == group]
                result[group] = float(np.mean(values)) if values else self.population_mean
            return result

        def step(self) -> None:
            self.agents.shuffle_do("step")
            for agent in self.agents:
                agent.state = agent.next_state

    model = SocialModel()
    history = []
    every = max(1, steps // 100)
    for step in range(0, steps + 1):
        if step in {0, steps} or step % every == 0:
            history.append({"step": step, "mean": model.population_mean, "group_means": {str(key): value for key, value in model.group_means.items()}})
        if step < steps:
            model.step()
    values = np.asarray([agent.state for agent in model.agents], dtype=float)
    groups = np.asarray([agent.group for agent in model.agents], dtype=int)
    group_results = {}
    for group in (0, 1):
        subset = values[groups == group]
        group_results[str(group)] = {"count": int(subset.size), "mean": float(np.mean(subset)) if subset.size else None}
    return {
        "engine": {"name": "mesa", "version": MESA_VERSION, "mode": mode, "network_used": False},
        "mode": mode,
        "agent_count": agent_count,
        "steps": steps,
        "seed": seed,
        "result_distribution": {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)), "p10": float(np.quantile(values, 0.1)), "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))},
        "group_results": group_results,
        "history": history,
        "interpretation_boundary": "Group-level scenario simulation only; not an individual psychological diagnosis or prediction.",
    }
