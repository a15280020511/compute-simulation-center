#!/usr/bin/env python3
"""Bounded population-level epidemiology scenario mode."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, integer, package


def epidemic_intervention_scenario(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("starsim")
    import starsim as ss

    agents = integer(inputs.get("agents", 1000), "inputs.agents", 100, 20_000)
    duration = integer(inputs.get("duration", 60), "inputs.duration", 5, 730)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    beta = finite(inputs.get("beta", 0.05), "inputs.beta")
    initial_prevalence = finite(inputs.get("initial_prevalence", 0.01), "inputs.initial_prevalence")
    contacts = finite(inputs.get("mean_contacts", 8.0), "inputs.mean_contacts")
    beta_multiplier = finite(inputs.get("beta_multiplier", 0.6), "inputs.beta_multiplier")
    if not 0 < beta <= 1 or not 0 < initial_prevalence < 1 or contacts <= 0:
        raise ComputeError("epidemiological probabilities and contact rate are invalid")
    if not 0 <= beta_multiplier <= 1:
        raise ComputeError("beta_multiplier must be between zero and one")

    def run(beta_value: float) -> Any:
        sim = ss.Sim(dict(
            n_agents=agents,
            dur=duration,
            rand_seed=seed,
            networks=dict(type="random", n_contacts=contacts),
            diseases=dict(type="sir", init_prev=initial_prevalence, beta=beta_value),
        ))
        sim.run()
        return sim

    def infected_trajectory(sim: Any) -> np.ndarray:
        results = sim.diseases.sir.results
        candidate = None
        for name in ("n_infected", "infected", "prevalence"):
            if hasattr(results, "get"):
                candidate = results.get(name)
            if candidate is None:
                candidate = getattr(results, name, None)
            if candidate is not None:
                break
        if candidate is None:
            raise ComputeError("Starsim SIR infected trajectory is unavailable")
        values = np.asarray(getattr(candidate, "values", candidate), dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ComputeError("Starsim returned an invalid infected trajectory")
        return values

    baseline = infected_trajectory(run(beta))
    intervention = infected_trajectory(run(beta * beta_multiplier))
    return {
        "mode": "epidemic_intervention_scenario",
        "comparison_type": "constant-transmission counterfactual",
        "baseline_peak_infected": float(np.max(baseline)),
        "intervention_peak_infected": float(np.max(intervention)),
        "peak_reduction": float(np.max(baseline) - np.max(intervention)),
        "baseline_infected_trajectory": baseline.tolist(),
        "intervention_infected_trajectory": intervention.tolist(),
        "beta_multiplier": beta_multiplier,
        "engine": {"starsim": package("starsim")},
        "clinical_use_allowed": False,
        "individual_risk_scoring_allowed": False,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "epidemic_intervention_scenario": epidemic_intervention_scenario,
}
