#!/usr/bin/env python3
"""Bounded multi-objective, policy, strategic and algebraic decision modes."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from gekko_backend import solve_nonnegative_linear_program
from think_tank_common import MAX_ACTORS, MAX_PERIODS, finite, identifiers, integer, mapping, matrix, package, probability, sequence, vector


def multiobjective_pareto(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pymoo")
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    coefficients = matrix(
        inputs.get("objective_coefficients"), "inputs.objective_coefficients", max_rows=10, max_columns=30
    )
    lower = vector(inputs.get("lower_bounds"), "inputs.lower_bounds", maximum=30)
    upper = vector(inputs.get("upper_bounds"), "inputs.upper_bounds", maximum=30)
    if coefficients.shape[1] != lower.size or lower.size != upper.size or np.any(lower >= upper):
        raise ComputeError("objective coefficients and valid bounds must align")
    population = integer(inputs.get("population", 80), "inputs.population", 20, 300)
    generations = integer(inputs.get("generations", 100), "inputs.generations", 10, 500)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)

    class LinearProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=lower.size, n_obj=coefficients.shape[0], n_ieq_constr=0, xl=lower, xu=upper)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = x @ coefficients.T

    result = minimize(
        LinearProblem(), NSGA2(pop_size=population), ("n_gen", generations), seed=seed, verbose=False
    )
    if result.X is None or result.F is None:
        raise ComputeError("multi-objective optimization produced no Pareto set")
    limit = min(100, len(result.X))
    return {
        "mode": "multiobjective_pareto",
        "solutions": [
            {"decision": result.X[i].tolist(), "objectives": result.F[i].tolist()} for i in range(limit)
        ],
        "solution_count": int(len(result.X)),
        "engine": {"pymoo": package("pymoo")},
    }


def bounded_hyperparameter_search(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("optuna")
    import optuna

    parameters = sequence(inputs.get("parameters"), "inputs.parameters")
    if not 1 <= len(parameters) <= 20:
        raise ComputeError("inputs.parameters must contain 1 to 20 definitions")
    definitions = []
    for i, raw in enumerate(parameters):
        spec = mapping(raw, f"inputs.parameters[{i}]")
        name = str(spec.get("name") or "")
        low = finite(spec.get("minimum"), f"parameter[{name}].minimum")
        high = finite(spec.get("maximum"), f"parameter[{name}].maximum")
        target = finite(spec.get("target"), f"parameter[{name}].target")
        weight = finite(spec.get("weight", 1.0), f"parameter[{name}].weight")
        if not name or low >= high or not low <= target <= high or weight <= 0:
            raise ComputeError("parameter names, bounds, target and weights are invalid")
        definitions.append((name, low, high, target, weight))
    trials = integer(inputs.get("trials", 100), "inputs.trials", 10, 1_000)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)

    def objective(trial):
        loss = 0.0
        for name, low, high, target, weight in definitions:
            value = trial.suggest_float(name, low, high)
            scale = max(high - low, 1e-12)
            loss += weight * ((value - target) / scale) ** 2
        return loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, timeout=120, show_progress_bar=False)
    return {
        "mode": "bounded_hyperparameter_search",
        "best_parameters": {str(k): float(v) for k, v in study.best_params.items()},
        "best_value": float(study.best_value),
        "trial_count": len(study.trials),
        "engine": {"optuna": package("optuna"), "sampler": "TPE-fixed-seed"},
        "ticket_supplied_objective_code": False,
    }


def algebraic_resource_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    objective = vector(inputs.get("objective"), "inputs.objective", maximum=200)
    constraints = matrix(
        inputs.get("constraint_matrix"), "inputs.constraint_matrix", max_rows=1_000, max_columns=200
    )
    bounds = vector(inputs.get("constraint_bounds"), "inputs.constraint_bounds", maximum=1_000)
    if constraints.shape[1] != objective.size or constraints.shape[0] != bounds.size:
        raise ComputeError("objective and constraint dimensions must align")

    maximize = bool(inputs.get("maximize", True))
    solver_engine = str(inputs.get("solver_engine") or "highs").strip().lower()
    if solver_engine not in {"highs", "gekko"}:
        raise ComputeError("inputs.solver_engine must be 'highs' or 'gekko'")

    if solver_engine == "gekko":
        gekko_version = package("gekko")
        result = solve_nonnegative_linear_program(
            objective,
            constraints,
            bounds,
            maximize=maximize,
        )
        return {
            "mode": "algebraic_resource_optimization",
            "decision": result["decision"],
            "objective_value": result["objective_value"],
            "termination": result["termination"],
            "engines": {
                "gekko": gekko_version,
                "solver": result["solver"],
                "remote": result["remote"],
            },
        }

    package("pyomo")
    package("highspy")
    import pyomo.environ as pyo

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, objective.size - 1)
    model.J = pyo.RangeSet(0, constraints.shape[0] - 1)
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    sense = pyo.maximize if maximize else pyo.minimize
    model.objective = pyo.Objective(expr=sum(float(objective[i]) * model.x[i] for i in model.I), sense=sense)
    model.constraints = pyo.ConstraintList()
    for j in range(constraints.shape[0]):
        model.constraints.add(sum(float(constraints[j, i]) * model.x[i] for i in model.I) <= float(bounds[j]))
    try:
        result = pyo.SolverFactory("appsi_highs").solve(model)
    except Exception as exc:
        raise ComputeError(f"Pyomo/HiGHS optimization failed: {type(exc).__name__}: {exc}") from exc
    termination = str(result.solver.termination_condition)
    if "optimal" not in termination.lower():
        raise ComputeError(f"algebraic optimization did not reach optimality: {termination}")
    values = [float(pyo.value(model.x[i])) for i in model.I]
    return {
        "mode": "algebraic_resource_optimization",
        "decision": values,
        "objective_value": float(pyo.value(model.objective)),
        "termination": termination,
        "engines": {"pyomo": package("pyomo"), "highspy": package("highspy")},
    }


def strategic_sandbox(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actors = identifiers(inputs.get("actors"), "inputs.actors", MAX_ACTORS)
    payoff = matrix(inputs.get("payoff_matrix"), "inputs.payoff_matrix", max_rows=MAX_ACTORS, max_columns=MAX_ACTORS)
    if payoff.shape != (len(actors), len(actors)):
        raise ComputeError("payoff_matrix must be square and match actors")
    resources = vector(inputs.get("initial_resources"), "inputs.initial_resources", maximum=MAX_ACTORS)
    if resources.size != len(actors) or np.any(resources < 0) or float(np.sum(resources)) <= 0:
        raise ComputeError("initial_resources must match actors, be non-negative and have positive total")
    periods = integer(inputs.get("periods", 50), "inputs.periods", 1, MAX_PERIODS)
    adaptation = probability(inputs.get("adaptation_rate", 0.2), "inputs.adaptation_rate")
    volatility = finite(inputs.get("shock_standard_deviation", 0.0), "inputs.shock_standard_deviation")
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    if volatility < 0:
        raise ComputeError("shock_standard_deviation must be non-negative")
    rng = np.random.default_rng(seed)
    shares = resources / float(np.sum(resources))
    history = []
    for period in range(periods + 1):
        scores = payoff @ shares
        history.append({"period": period, "shares": shares.tolist(), "scores": scores.tolist()})
        if period == periods:
            break
        shock = rng.normal(0.0, volatility, size=len(actors)) if volatility else np.zeros(len(actors))
        adjusted = scores + shock
        growth = np.exp(np.clip(adaptation * (adjusted - float(np.dot(shares, adjusted))), -20, 20))
        shares = shares * growth
        shares = shares / float(np.sum(shares))
    winner = int(np.argmax(shares))
    return {
        "mode": "strategic_sandbox",
        "actors": actors,
        "final_shares": shares.tolist(),
        "dominant_actor": actors[winner],
        "history": history,
        "interpretation_boundary": "Scenario sandbox only; outputs depend on declared payoff and adaptation assumptions.",
    }


def influence_diagram(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actions = identifiers(inputs.get("actions"), "inputs.actions", 50)
    states = identifiers(inputs.get("states"), "inputs.states", 50)
    probabilities = vector(inputs.get("state_probabilities"), "inputs.state_probabilities", maximum=50)
    utility = matrix(inputs.get("utilities"), "inputs.utilities", max_rows=50, max_columns=50)
    if probabilities.size != len(states) or utility.shape != (len(actions), len(states)):
        raise ComputeError("probabilities and utilities must align with actions and states")
    if np.any(probabilities < 0) or not math.isclose(float(np.sum(probabilities)), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ComputeError("state_probabilities must be non-negative and sum to 1")
    expected = utility @ probabilities
    order = np.argsort(-expected)
    return {
        "mode": "influence_diagram",
        "ranking": [
            {"action": actions[int(i)], "expected_utility": float(expected[int(i)]), "rank": rank + 1}
            for rank, i in enumerate(order)
        ],
        "best_action": actions[int(order[0])],
        "perfect_information_value": float(np.dot(probabilities, np.max(utility, axis=0)) - np.max(expected)),
    }


def _gini(values: np.ndarray) -> float:
    sorted_values = np.sort(np.maximum(values, 0.0))
    total = float(np.sum(sorted_values))
    if total <= 0:
        return 0.0
    index = np.arange(1, sorted_values.size + 1)
    return float((2 * np.sum(index * sorted_values) / (sorted_values.size * total)) - (sorted_values.size + 1) / sorted_values.size)


def policy_microsimulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    incomes = vector(inputs.get("incomes"), "inputs.incomes", minimum=10)
    brackets_raw = sequence(inputs.get("tax_brackets"), "inputs.tax_brackets")
    brackets = []
    last = 0.0
    for i, raw in enumerate(brackets_raw):
        spec = mapping(raw, f"inputs.tax_brackets[{i}]")
        threshold = finite(spec.get("threshold"), f"tax_brackets[{i}].threshold")
        rate = probability(spec.get("rate"), f"tax_brackets[{i}].rate")
        if threshold < last:
            raise ComputeError("tax bracket thresholds must be non-decreasing")
        brackets.append((threshold, rate))
        last = threshold
    transfer = finite(inputs.get("universal_transfer", 0.0), "inputs.universal_transfer")

    def tax(income: float) -> float:
        total = 0.0
        previous = 0.0
        for threshold, rate in brackets:
            taxable = max(0.0, min(income, threshold) - previous)
            total += taxable * rate
            previous = threshold
            if income <= threshold:
                return total
        if brackets:
            total += max(0.0, income - previous) * brackets[-1][1]
        return total

    taxes = np.asarray([tax(float(value)) for value in incomes], dtype=float)
    disposable = incomes - taxes + transfer
    return {
        "mode": "policy_microsimulation",
        "population": int(incomes.size),
        "tax_revenue": float(np.sum(taxes)),
        "transfer_cost": float(transfer * incomes.size),
        "net_fiscal_balance": float(np.sum(taxes) - transfer * incomes.size),
        "mean_disposable_income": float(np.mean(disposable)),
        "gini_before": _gini(incomes),
        "gini_after": _gini(disposable),
        "poverty_rate_before": float(np.mean(incomes < finite(inputs.get("poverty_line", 0.0), "inputs.poverty_line"))),
        "poverty_rate_after": float(np.mean(disposable < finite(inputs.get("poverty_line", 0.0), "inputs.poverty_line"))),
        "individual_results": disposable.tolist(),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "multiobjective_pareto": multiobjective_pareto,
    "bounded_hyperparameter_search": bounded_hyperparameter_search,
    "algebraic_resource_optimization": algebraic_resource_optimization,
    "strategic_sandbox": strategic_sandbox,
    "influence_diagram": influence_diagram,
    "policy_microsimulation": policy_microsimulation,
}
