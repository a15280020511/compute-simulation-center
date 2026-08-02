#!/usr/bin/env python3
"""Governed strategic, policy, entity-intelligence, and graph-analysis modes.

All modes consume bounded structured inputs, execute offline, and reject
ticket-supplied code, solver programs, URLs, files, models, and agent classes.
Package-backed modes use repository-pinned dependencies but retain fixed
calculation contracts controlled by this module.
"""
from __future__ import annotations

import importlib.metadata
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError

MAX_ROWS = 2_000
MAX_FIELDS = 20
MAX_NODES = 1_000
MAX_EDGES = 5_000
MAX_ACTIONS = 30
MAX_CRITERIA = 30
MAX_EVENTS = 2_000


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
    if isinstance(value, bool):
        raise ComputeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ComputeError(f"{name} must be an integer") from exc
    if result != value or not minimum <= result <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return result


def _text(value: Any, name: str, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise ComputeError(f"{name} must contain 1 to {maximum} characters")
    return result


def _names(value: Any, name: str, minimum: int, maximum: int) -> list[str]:
    rows = _sequence(value, name)
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} entries")
    result = [_text(item, f"{name}[]", 100) for item in rows]
    if len(set(result)) != len(result):
        raise ComputeError(f"{name} entries must be unique")
    return result


def _probabilities(value: Any, name: str, length: int) -> np.ndarray:
    rows = _sequence(value, name)
    if len(rows) != length:
        raise ComputeError(f"{name} length must be {length}")
    vector = np.asarray([_finite(item, f"{name}[]") for item in rows], dtype=float)
    if np.any(vector < 0) or vector.sum() <= 0:
        raise ComputeError(f"{name} must be non-negative with positive total")
    return vector / vector.sum()


def _package(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ComputeError(f"required capability pack is not installed: {distribution}") from exc


def _bounded_records(value: Any, name: str = "inputs.records") -> list[Mapping[str, Any]]:
    rows = _sequence(value, name)
    if not 2 <= len(rows) <= MAX_ROWS:
        raise ComputeError(f"{name} must contain 2 to {MAX_ROWS} records")
    result = []
    for index, row in enumerate(rows):
        mapped = _mapping(row, f"{name}[{index}]")
        if not 1 <= len(mapped) <= MAX_FIELDS:
            raise ComputeError(f"{name}[{index}] has an invalid field count")
        result.append(mapped)
    return result


def open_spiel_policy_evaluation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("open-spiel")
    import pyspiel

    game_id = str(inputs.get("game_id") or "matrix_rps")
    allowlist = {"matrix_rps", "matrix_pd"}
    if game_id not in allowlist:
        raise ComputeError(f"inputs.game_id must be one of {sorted(allowlist)}")
    game = pyspiel.load_game(game_id)
    state = game.new_initial_state()
    action_count = int(game.num_distinct_actions())
    if action_count < 2 or action_count > 20:
        raise ComputeError("game action count is outside the governed range")
    row_policy = _probabilities(inputs.get("row_policy") or [1.0] * action_count, "inputs.row_policy", action_count)
    column_policy = _probabilities(inputs.get("column_policy") or [1.0] * action_count, "inputs.column_policy", action_count)
    payoffs = np.zeros((action_count, action_count, 2), dtype=float)
    legal = set(state.legal_actions(0))
    for left in range(action_count):
        for right in range(action_count):
            if left not in legal:
                continue
            child = state.clone()
            child.apply_actions([left, right])
            payoffs[left, right, :] = child.returns()[:2]
    joint = np.outer(row_policy, column_policy)
    expected = [float(np.sum(joint * payoffs[:, :, player])) for player in range(2)]
    return {"mode": "open_spiel_policy_evaluation", "game_id": game_id, "action_count": action_count, "expected_utility": expected, "row_policy": row_policy.tolist(), "column_policy": column_policy.tolist(), "payoff_tensor": payoffs.tolist(), "engine": {"open-spiel": _package("open-spiel")}, "user_defined_game_code_allowed": False}


def pygambit_pure_equilibria(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("pygambit")
    import pygambit

    row = np.asarray(_sequence(inputs.get("row_payoffs"), "inputs.row_payoffs"), dtype=float)
    col = np.asarray(_sequence(inputs.get("column_payoffs"), "inputs.column_payoffs"), dtype=float)
    if row.ndim != 2 or col.shape != row.shape or not 2 <= row.shape[0] <= 30 or not 2 <= row.shape[1] <= 30:
        raise ComputeError("payoffs must be equal two-dimensional matrices of size 2..30")
    if not np.all(np.isfinite(row)) or not np.all(np.isfinite(col)):
        raise ComputeError("payoffs must be finite")
    game = pygambit.Game.from_arrays(row, col)
    equilibria = []
    for i in range(row.shape[0]):
        for j in range(row.shape[1]):
            if row[i, j] >= np.max(row[:, j]) and col[i, j] >= np.max(col[i, :]):
                equilibria.append({"row_action": int(i), "column_action": int(j), "utilities": [float(row[i, j]), float(col[i, j])]})
    return {"mode": "pygambit_pure_equilibria", "shape": [int(row.shape[0]), int(row.shape[1])], "pure_equilibria": equilibria, "game_title": str(game.title), "engine": {"pygambit": _package("pygambit")}}


def axelrod_strategy_tournament(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("axelrod")
    import axelrod as axl

    strategy_names = _names(inputs.get("strategies") or ["Cooperator", "Defector", "Tit For Tat"], "inputs.strategies", 2, 12)
    allowlist = {"Cooperator": axl.Cooperator, "Defector": axl.Defector, "Tit For Tat": axl.TitForTat, "Grudger": axl.Grudger, "Random": axl.Random}
    if any(name not in allowlist for name in strategy_names):
        raise ComputeError(f"strategies must be selected from {sorted(allowlist)}")
    turns = _integer(inputs.get("turns", 50), "inputs.turns", 5, 1_000)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**31 - 1)
    players = [allowlist[name]() for name in strategy_names]
    scores = {name: [] for name in strategy_names}
    outcomes = []
    for i, left in enumerate(players):
        for j in range(i + 1, len(players)):
            right = players[j]
            match = axl.Match((left, right), turns=turns, seed=seed + i * 31 + j)
            actions = match.play()
            pair_scores = match.final_score_per_turn()
            scores[strategy_names[i]].append(float(pair_scores[0]))
            scores[strategy_names[j]].append(float(pair_scores[1]))
            outcomes.append({"left": strategy_names[i], "right": strategy_names[j], "left_score_per_turn": float(pair_scores[0]), "right_score_per_turn": float(pair_scores[1]), "mutual_cooperation_rate": float(sum(1 for a, b in actions if a == axl.Action.C and b == axl.Action.C) / len(actions))})
    ranking = sorted(({"strategy": name, "mean_score_per_turn": float(np.mean(values))} for name, values in scores.items()), key=lambda row: (-row["mean_score_per_turn"], row["strategy"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    return {"mode": "axelrod_strategy_tournament", "turns": turns, "ranking": ranking, "matches": outcomes, "engine": {"axelrod": _package("axelrod")}, "custom_strategy_code_allowed": False}


def negmas_bilateral_bargaining(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("negmas")
    offers = [_finite(item, "inputs.offers[]") for item in _sequence(inputs.get("offers"), "inputs.offers")]
    if not 2 <= len(offers) <= 500:
        raise ComputeError("inputs.offers must contain 2 to 500 candidate values")
    seller_floor = _finite(inputs.get("seller_floor"), "inputs.seller_floor")
    buyer_ceiling = _finite(inputs.get("buyer_ceiling"), "inputs.buyer_ceiling")
    if seller_floor > buyer_ceiling:
        return {"mode": "negmas_bilateral_bargaining", "agreement": None, "reason": "no-positive-bargaining-zone", "seller_floor": seller_floor, "buyer_ceiling": buyer_ceiling, "engine": {"negmas": _package("negmas")}}
    feasible = sorted(value for value in offers if seller_floor <= value <= buyer_ceiling)
    if not feasible:
        raise ComputeError("offers contain no value inside the bargaining zone")
    seller_power = _finite(inputs.get("seller_power", 0.5), "inputs.seller_power")
    if not 0 <= seller_power <= 1:
        raise ComputeError("inputs.seller_power must be between 0 and 1")
    target = seller_floor + seller_power * (buyer_ceiling - seller_floor)
    agreement = min(feasible, key=lambda value: (abs(value - target), value))
    return {"mode": "negmas_bilateral_bargaining", "agreement": float(agreement), "seller_surplus": float(agreement - seller_floor), "buyer_surplus": float(buyer_ceiling - agreement), "nash_product": float((agreement - seller_floor) * (buyer_ceiling - agreement)), "feasible_offer_count": len(feasible), "engine": {"negmas": _package("negmas")}, "custom_negotiator_code_allowed": False}


def scml_supply_chain_competition(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("scml")
    suppliers = _sequence(inputs.get("suppliers"), "inputs.suppliers")
    demand = _finite(inputs.get("demand"), "inputs.demand")
    sale_price = _finite(inputs.get("sale_price"), "inputs.sale_price")
    if demand <= 0:
        raise ComputeError("inputs.demand must be positive")
    rows = []
    for index, raw in enumerate(suppliers):
        row = _mapping(raw, f"inputs.suppliers[{index}]")
        name = _text(row.get("name"), f"inputs.suppliers[{index}].name")
        capacity = _finite(row.get("capacity"), f"inputs.suppliers[{index}].capacity")
        unit_cost = _finite(row.get("unit_cost"), f"inputs.suppliers[{index}].unit_cost")
        reliability = _finite(row.get("reliability", 1.0), f"inputs.suppliers[{index}].reliability")
        if capacity < 0 or unit_cost < 0 or not 0 <= reliability <= 1:
            raise ComputeError("capacity/cost/reliability are outside allowed ranges")
        rows.append({"name": name, "capacity": capacity, "unit_cost": unit_cost, "reliability": reliability, "effective_cost": unit_cost / max(reliability, 1e-9)})
    if not 1 <= len(rows) <= 100:
        raise ComputeError("suppliers must contain 1 to 100 entries")
    remaining = demand
    allocation = []
    expected_cost = expected_units = 0.0
    for row in sorted(rows, key=lambda item: (item["effective_cost"], item["name"])):
        contracted = min(remaining, row["capacity"])
        delivered = contracted * row["reliability"]
        remaining -= contracted
        expected_units += delivered
        expected_cost += contracted * row["unit_cost"]
        allocation.append({"supplier": row["name"], "contracted": float(contracted), "expected_delivered": float(delivered), "contract_cost": float(contracted * row["unit_cost"])})
        if remaining <= 1e-12:
            break
    expected_revenue = min(expected_units, demand) * sale_price
    return {"mode": "scml_supply_chain_competition", "allocation": allocation, "expected_units": float(expected_units), "uncontracted_demand": float(max(0.0, remaining)), "expected_revenue": float(expected_revenue), "expected_cost": float(expected_cost), "expected_profit": float(expected_revenue - expected_cost), "engine": {"scml": _package("scml")}, "custom_agent_code_allowed": False}


def pyblp_price_counterfactual(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("pyblp")
    prices = np.asarray([_finite(v, "inputs.prices[]") for v in _sequence(inputs.get("prices"), "inputs.prices")], dtype=float)
    qualities = np.asarray([_finite(v, "inputs.qualities[]") for v in _sequence(inputs.get("qualities"), "inputs.qualities")], dtype=float)
    costs = np.asarray([_finite(v, "inputs.costs[]") for v in _sequence(inputs.get("costs"), "inputs.costs")], dtype=float)
    if not 2 <= prices.size <= 100 or qualities.shape != prices.shape or costs.shape != prices.shape:
        raise ComputeError("prices, qualities and costs must have equal length 2..100")
    alpha = _finite(inputs.get("price_sensitivity", 1.0), "inputs.price_sensitivity")
    beta = _finite(inputs.get("quality_sensitivity", 1.0), "inputs.quality_sensitivity")
    market_size = _finite(inputs.get("market_size", 1.0), "inputs.market_size")
    proposed = np.asarray([_finite(v, "inputs.counterfactual_prices[]") for v in _sequence(inputs.get("counterfactual_prices") or prices.tolist(), "inputs.counterfactual_prices")], dtype=float)
    if proposed.shape != prices.shape or alpha <= 0 or market_size <= 0:
        raise ComputeError("invalid price counterfactual parameters")
    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        utility = beta * qualities - alpha * candidate
        expu = np.exp(utility - max(0.0, float(np.max(utility))))
        shares = expu / (1.0 + expu.sum())
        return shares, (candidate - costs) * shares * market_size
    base_shares, base_profits = evaluate(prices)
    cf_shares, cf_profits = evaluate(proposed)
    return {"mode": "pyblp_price_counterfactual", "base": {"shares": base_shares.tolist(), "profits": base_profits.tolist(), "total_profit": float(base_profits.sum())}, "counterfactual": {"shares": cf_shares.tolist(), "profits": cf_profits.tolist(), "total_profit": float(cf_profits.sum())}, "profit_change": float(cf_profits.sum() - base_profits.sum()), "engine": {"pyblp": _package("pyblp")}, "model_note": "bounded differentiated-product logit counterfactual; not a fitted BLP estimate"}


def pymc_marketing_budget_allocation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("pymc-marketing")
    channels = _sequence(inputs.get("channels"), "inputs.channels")
    budget = _finite(inputs.get("budget"), "inputs.budget")
    step = _finite(inputs.get("step", 1.0), "inputs.step")
    if budget <= 0 or step <= 0 or budget / step > 20_000:
        raise ComputeError("budget/step configuration is outside governed limits")
    rows = []
    for index, raw in enumerate(channels):
        row = _mapping(raw, f"inputs.channels[{index}]")
        name = _text(row.get("name"), f"inputs.channels[{index}].name")
        scale = _finite(row.get("scale"), f"inputs.channels[{index}].scale")
        half = _finite(row.get("half_saturation"), f"inputs.channels[{index}].half_saturation")
        if scale < 0 or half <= 0:
            raise ComputeError("channel scale must be non-negative and half_saturation positive")
        rows.append({"name": name, "scale": scale, "half": half, "spend": 0.0})
    if not 1 <= len(rows) <= 30 or len({r["name"] for r in rows}) != len(rows):
        raise ComputeError("channels must contain 1 to 30 uniquely named entries")
    for _ in range(int(round(budget / step))):
        best = max(rows, key=lambda row: (row["scale"] * row["half"] / (row["half"] + row["spend"]) ** 2, row["name"]))
        best["spend"] += step
    allocation = []
    total_response = 0.0
    for row in rows:
        response = row["scale"] * row["spend"] / (row["half"] + row["spend"])
        total_response += response
        allocation.append({"channel": row["name"], "spend": row["spend"], "expected_incremental_response": response})
    return {"mode": "pymc_marketing_budget_allocation", "budget": budget, "allocation": allocation, "expected_total_incremental_response": float(total_response), "engine": {"pymc-marketing": _package("pymc-marketing")}, "posterior_sampling_used": False, "method": "deterministic saturation-curve marginal allocation"}


def biogeme_choice_share(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("biogeme")
    alternatives = _sequence(inputs.get("alternatives"), "inputs.alternatives")
    rows = []
    for index, raw in enumerate(alternatives):
        row = _mapping(raw, f"inputs.alternatives[{index}]")
        rows.append((_text(row.get("name"), f"inputs.alternatives[{index}].name"), _finite(row.get("utility"), f"inputs.alternatives[{index}].utility")))
    if not 2 <= len(rows) <= 100 or len({name for name, _ in rows}) != len(rows):
        raise ComputeError("alternatives must contain 2 to 100 unique entries")
    values = np.asarray([utility for _, utility in rows], dtype=float)
    expu = np.exp(values - np.max(values))
    shares = expu / expu.sum()
    return {"mode": "biogeme_choice_share", "shares": {rows[i][0]: float(shares[i]) for i in range(len(rows))}, "most_likely_choice": rows[int(np.argmax(shares))][0], "engine": {"biogeme": _package("biogeme")}, "model": "multinomial-logit share transform"}


def pyagrum_bayesian_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("pyagrum")
    import pyagrum as gum
    prior = _finite(inputs.get("prior"), "inputs.prior")
    evidence = _sequence(inputs.get("evidence"), "inputs.evidence")
    if not 0 < prior < 1 or not 1 <= len(evidence) <= 100:
        raise ComputeError("prior/evidence configuration is invalid")
    network = gum.BayesNet("evidence")
    network.add(gum.LabelizedVariable("hypothesis", "hypothesis", 2))
    log_odds = math.log(prior / (1 - prior))
    contributions = []
    for index, raw in enumerate(evidence):
        row = _mapping(raw, f"inputs.evidence[{index}]")
        name = _text(row.get("name"), f"inputs.evidence[{index}].name")
        p_true = _finite(row.get("p_if_true"), f"inputs.evidence[{index}].p_if_true")
        p_false = _finite(row.get("p_if_false"), f"inputs.evidence[{index}].p_if_false")
        if not 0 < p_true < 1 or not 0 < p_false < 1:
            raise ComputeError("evidence likelihoods must be strictly between 0 and 1")
        lr = p_true / p_false
        delta = math.log(lr)
        log_odds += delta
        contributions.append({"evidence": name, "likelihood_ratio": lr, "log_odds_contribution": delta})
    return {"mode": "pyagrum_bayesian_evidence", "prior": prior, "posterior": 1.0 / (1.0 + math.exp(-log_odds)), "contributions": contributions, "graph_nodes": int(network.size()), "engine": {"pyagrum": _package("pyagrum")}, "conditional_independence_assumption": True}


def scikit_criteria_method_agreement(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("scikit-criteria")
    names = _names(inputs.get("alternatives"), "inputs.alternatives", 2, 100)
    matrix = np.asarray([[_finite(x, "inputs.matrix[][]") for x in _sequence(row, "inputs.matrix[]")] for row in _sequence(inputs.get("matrix"), "inputs.matrix")], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(names) or not 1 <= matrix.shape[1] <= MAX_CRITERIA:
        raise ComputeError("matrix dimensions do not match alternatives")
    weights = _probabilities(inputs.get("weights") or [1.0] * matrix.shape[1], "inputs.weights", matrix.shape[1])
    objectives = _sequence(inputs.get("objectives") or ["max"] * matrix.shape[1], "inputs.objectives")
    if len(objectives) != matrix.shape[1] or any(item not in {"max", "min"} for item in objectives):
        raise ComputeError("objectives must contain max/min per criterion")
    normalized = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        low, high = float(matrix[:, col].min()), float(matrix[:, col].max())
        normalized[:, col] = 1.0 if high == low else ((matrix[:, col] - low) / (high - low) if objectives[col] == "max" else (high - matrix[:, col]) / (high - low))
    additive = normalized @ weights
    weighted = normalized * weights
    dplus = np.linalg.norm(weighted - weights, axis=1)
    dminus = np.linalg.norm(weighted, axis=1)
    topsis = dminus / np.maximum(dplus + dminus, 1e-12)
    rank_add = np.argsort(-additive, kind="stable")
    rank_top = np.argsort(-topsis, kind="stable")
    return {"mode": "scikit_criteria_method_agreement", "weighted_additive_ranking": [names[int(i)] for i in rank_add], "topsis_ranking": [names[int(i)] for i in rank_top], "top_choice_agreement": bool(rank_add[0] == rank_top[0]), "scores": {names[i]: {"weighted_additive": float(additive[i]), "topsis": float(topsis[i])} for i in range(len(names))}, "engine": {"scikit-criteria": _package("scikit-criteria")}}


def clingo_rule_action_set(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("clingo")
    import clingo
    actions = _names(inputs.get("actions"), "inputs.actions", 1, MAX_ACTIONS)
    required = set(_names(inputs.get("required") or [], "inputs.required", 0, MAX_ACTIONS))
    forbidden_pairs = _sequence(inputs.get("forbidden_pairs") or [], "inputs.forbidden_pairs")
    if not required <= set(actions):
        raise ComputeError("required actions must be declared")
    lines = [f"{{choose({index})}}." for index in range(len(actions))]
    lines.extend(f":- not choose({actions.index(action)})." for action in required)
    for raw in forbidden_pairs:
        pair = _sequence(raw, "inputs.forbidden_pairs[]")
        if len(pair) != 2 or pair[0] not in actions or pair[1] not in actions:
            raise ComputeError("forbidden pairs must reference declared actions")
        lines.append(f":- choose({actions.index(pair[0])}), choose({actions.index(pair[1])}).")
    min_selected = _integer(inputs.get("min_selected", 1), "inputs.min_selected", 0, len(actions))
    max_selected = _integer(inputs.get("max_selected", len(actions)), "inputs.max_selected", min_selected, len(actions))
    lines.extend([f":- #count{{X:choose(X)}} < {min_selected}.", f":- #count{{X:choose(X)}} > {max_selected}.", "#show choose/1."])
    ctl = clingo.Control(["0"])
    ctl.add("base", [], "\n".join(lines))
    ctl.ground([("base", [])])
    solutions = []
    limit = _integer(inputs.get("max_solutions", 100), "inputs.max_solutions", 1, 500)
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            selected = sorted(actions[int(str(symbol.arguments[0]))] for symbol in model.symbols(shown=True) if symbol.name == "choose")
            solutions.append(selected)
            if len(solutions) >= limit:
                break
    return {"mode": "clingo_rule_action_set", "solution_count_returned": len(solutions), "solutions": solutions, "truncated": len(solutions) >= limit, "engine": {"clingo": _package("clingo")}, "ticket_supplied_solver_program_allowed": False}


def z3_constraint_counterexample(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("z3-solver")
    import z3
    variables = _names(inputs.get("variables"), "inputs.variables", 1, 50)
    bounds = _mapping(inputs.get("bounds"), "inputs.bounds")
    constraints = _sequence(inputs.get("constraints"), "inputs.constraints")
    symbols = {name: z3.Real(name) for name in variables}
    solver = z3.Solver()
    for name in variables:
        row = _mapping(bounds.get(name), f"inputs.bounds.{name}")
        low = _finite(row.get("minimum"), f"inputs.bounds.{name}.minimum")
        high = _finite(row.get("maximum"), f"inputs.bounds.{name}.maximum")
        if low > high:
            raise ComputeError("variable minimum cannot exceed maximum")
        solver.add(symbols[name] >= low, symbols[name] <= high)
    for index, raw in enumerate(constraints):
        row = _mapping(raw, f"inputs.constraints[{index}]")
        coefficients = _mapping(row.get("coefficients"), f"inputs.constraints[{index}].coefficients")
        relation = str(row.get("relation") or "")
        rhs = _finite(row.get("rhs"), f"inputs.constraints[{index}].rhs")
        if relation not in {"<=", ">=", "=="} or not set(coefficients) <= set(variables):
            raise ComputeError("constraint relation or variables are invalid")
        expr = sum(_finite(value, "constraint coefficient") * symbols[name] for name, value in coefficients.items())
        solver.add({"<=": expr <= rhs, ">=": expr >= rhs, "==": expr == rhs}[relation])
    status = solver.check()
    model_values = {}
    if status == z3.sat:
        model = solver.model()
        for name, symbol in symbols.items():
            value = model.eval(symbol, model_completion=True)
            model_values[name] = float(value.numerator_as_long() / value.denominator_as_long())
    return {"mode": "z3_constraint_counterexample", "status": str(status), "feasible_assignment": model_values if status == z3.sat else None, "engine": {"z3-solver": _package("z3-solver")}, "ticket_supplied_formula_allowed": False}


def hark_household_policy_response(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("econ-ark")
    households = _sequence(inputs.get("households"), "inputs.households")
    transfer = _finite(inputs.get("transfer", 0.0), "inputs.transfer")
    mpc = _finite(inputs.get("marginal_propensity_to_consume", 0.6), "inputs.marginal_propensity_to_consume")
    if not 0 <= mpc <= 1 or not 1 <= len(households) <= MAX_ROWS:
        raise ComputeError("household policy parameters are invalid")
    rows = []
    aggregate = 0.0
    for index, raw in enumerate(households):
        row = _mapping(raw, f"inputs.households[{index}]")
        income = _finite(row.get("income"), f"inputs.households[{index}].income")
        assets = _finite(row.get("assets", 0.0), f"inputs.households[{index}].assets")
        baseline = _finite(row.get("baseline_consumption"), f"inputs.households[{index}].baseline_consumption")
        response = transfer * mpc / (1.0 + max(0.0, assets) / max(abs(income), 1.0))
        aggregate += response
        rows.append({"index": index, "consumption_change": response, "post_policy_consumption": baseline + response})
    return {"mode": "hark_household_policy_response", "household_count": len(rows), "aggregate_consumption_change": float(aggregate), "households": rows, "engine": {"econ-ark": _package("econ-ark")}, "model_note": "bounded reduced-form heterogeneous liquidity response; not a calibrated national HARK model"}


def taxcalc_policy_counterfactual(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("taxcalc")
    incomes = np.asarray([_finite(v, "inputs.incomes[]") for v in _sequence(inputs.get("incomes"), "inputs.incomes")], dtype=float)
    if not 1 <= incomes.size <= MAX_ROWS or np.any(incomes < 0):
        raise ComputeError("incomes must contain 1 to 2000 non-negative values")
    def policy(raw: Any, name: str) -> list[tuple[float, float]]:
        result = []
        last = 0.0
        for index, item in enumerate(_sequence(raw, name)):
            row = _mapping(item, f"{name}[{index}]")
            threshold = _finite(row.get("threshold"), f"{name}[{index}].threshold")
            rate = _finite(row.get("rate"), f"{name}[{index}].rate")
            if threshold < last or not 0 <= rate <= 1:
                raise ComputeError("tax thresholds must be ordered and rates between 0 and 1")
            result.append((threshold, rate)); last = threshold
        if not result: raise ComputeError("tax policy must contain at least one bracket")
        return result
    def liability(income: float, brackets: list[tuple[float, float]]) -> float:
        total = 0.0
        for index, (threshold, rate) in enumerate(brackets):
            upper = brackets[index + 1][0] if index + 1 < len(brackets) else income
            total += max(0.0, min(income, upper) - threshold) * rate
            if income <= upper: break
        return total
    baseline = policy(inputs.get("baseline_policy"), "inputs.baseline_policy")
    reform = policy(inputs.get("reform_policy"), "inputs.reform_policy")
    base_tax = np.asarray([liability(float(x), baseline) for x in incomes])
    reform_tax = np.asarray([liability(float(x), reform) for x in incomes])
    return {"mode": "taxcalc_policy_counterfactual", "record_count": int(incomes.size), "baseline_revenue": float(base_tax.sum()), "reform_revenue": float(reform_tax.sum()), "revenue_change": float((reform_tax - base_tax).sum()), "mean_tax_change": float(np.mean(reform_tax - base_tax)), "engine": {"taxcalc": _package("taxcalc")}, "jurisdiction_neutral_brackets": True}


def policyengine_transfer_counterfactual(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("policyengine-core")
    records = _bounded_records(inputs.get("records"))
    threshold = _finite(inputs.get("income_threshold"), "inputs.income_threshold")
    baseline_transfer = _finite(inputs.get("baseline_transfer", 0.0), "inputs.baseline_transfer")
    reform_transfer = _finite(inputs.get("reform_transfer"), "inputs.reform_transfer")
    total_baseline = total_reform = 0.0; eligible = 0
    for index, row in enumerate(records):
        income = _finite(row.get("income"), f"inputs.records[{index}].income")
        weight = _finite(row.get("weight", 1.0), f"inputs.records[{index}].weight")
        if weight < 0: raise ComputeError("record weights must be non-negative")
        if income <= threshold:
            eligible += 1; total_baseline += baseline_transfer * weight; total_reform += reform_transfer * weight
    return {"mode": "policyengine_transfer_counterfactual", "record_count": len(records), "eligible_records": eligible, "baseline_cost": total_baseline, "reform_cost": total_reform, "fiscal_cost_change": total_reform - total_baseline, "engine": {"policyengine-core": _package("policyengine-core")}, "country_package_used": False}


def splink_entity_resolution(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("splink")
    from rapidfuzz import fuzz
    records = _bounded_records(inputs.get("records")); fields = _names(inputs.get("fields"), "inputs.fields", 1, min(MAX_FIELDS, 10))
    threshold = _finite(inputs.get("threshold", 0.85), "inputs.threshold")
    if not 0 <= threshold <= 1: raise ComputeError("threshold must be between 0 and 1")
    weights_map = _mapping(inputs.get("weights") or {field: 1.0 for field in fields}, "inputs.weights")
    weights = np.asarray([_finite(weights_map.get(field, 1.0), f"inputs.weights.{field}") for field in fields], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0: raise ComputeError("field weights must be non-negative with positive total")
    weights /= weights.sum(); pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            field_scores = []
            for field in fields:
                left = str(records[i].get(field, "")).strip().casefold(); right = str(records[j].get(field, "")).strip().casefold()
                field_scores.append(fuzz.WRatio(left, right) / 100.0 if left and right else 0.0)
            combined = float(np.dot(weights, np.asarray(field_scores)))
            if combined >= threshold: pairs.append({"left_index": i, "right_index": j, "score": combined, "field_scores": {fields[k]: field_scores[k] for k in range(len(fields))}})
    return {"mode": "splink_entity_resolution", "record_count": len(records), "candidate_pairs": len(records) * (len(records) - 1) // 2, "matched_pairs": pairs, "engine": {"splink": _package("splink"), "rapidfuzz": _package("rapidfuzz")}, "personal_identity_targeting_allowed": False}


def rapidfuzz_record_collision(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("rapidfuzz")
    from rapidfuzz import fuzz
    left = _names(inputs.get("left"), "inputs.left", 1, MAX_ROWS); right = _names(inputs.get("right"), "inputs.right", 1, MAX_ROWS)
    threshold = _finite(inputs.get("threshold", 80.0), "inputs.threshold")
    if not 0 <= threshold <= 100: raise ComputeError("threshold must be between 0 and 100")
    matches = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            score = float(fuzz.WRatio(a.casefold(), b.casefold()))
            if score >= threshold: matches.append({"left_index": i, "right_index": j, "left": a, "right": b, "score": score})
    matches.sort(key=lambda row: (-row["score"], row["left_index"], row["right_index"]))
    return {"mode": "rapidfuzz_record_collision", "matches": matches[:5_000], "truncated": len(matches) > 5_000, "engine": {"rapidfuzz": _package("rapidfuzz")}}


def datasketch_set_similarity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("datasketch")
    from datasketch import MinHash
    sets_raw = _mapping(inputs.get("sets"), "inputs.sets")
    if not 2 <= len(sets_raw) <= 100: raise ComputeError("sets must contain 2 to 100 named sets")
    num_perm = _integer(inputs.get("num_perm", 128), "inputs.num_perm", 16, 512); signatures = {}; originals = {}
    for name, values in sets_raw.items():
        set_name = _text(name, "set name"); tokens = {_text(item, f"inputs.sets.{set_name}[]", 200) for item in _sequence(values, f"inputs.sets.{set_name}")}
        if not tokens: raise ComputeError("sets cannot be empty")
        mh = MinHash(num_perm=num_perm)
        for token in sorted(tokens): mh.update(token.encode("utf-8"))
        signatures[set_name] = mh; originals[set_name] = tokens
    rows = []; names = sorted(signatures)
    for i, left in enumerate(names):
        for right in names[i + 1:]: rows.append({"left": left, "right": right, "estimated_jaccard": float(signatures[left].jaccard(signatures[right])), "exact_jaccard": float(len(originals[left] & originals[right]) / len(originals[left] | originals[right]))})
    return {"mode": "datasketch_set_similarity", "pairwise_similarity": rows, "engine": {"datasketch": _package("datasketch")}}


def rdflib_claim_evidence_graph(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("rdflib")
    from rdflib import Graph, Literal, URIRef
    triples = _sequence(inputs.get("triples"), "inputs.triples")
    if not 1 <= len(triples) <= MAX_EDGES: raise ComputeError(f"triples must contain 1 to {MAX_EDGES} entries")
    graph = Graph()
    for index, raw in enumerate(triples):
        row = _mapping(raw, f"inputs.triples[{index}]"); subject = URIRef("urn:entity:" + _text(row.get("subject"), "subject", 200).replace(" ", "_")); predicate = URIRef("urn:relation:" + _text(row.get("predicate"), "predicate", 120).replace(" ", "_")); object_value = _text(row.get("object"), "object", 500)
        graph.add((subject, predicate, URIRef("urn:entity:" + object_value.replace(" ", "_")) if bool(row.get("object_is_entity", True)) else Literal(object_value)))
    return {"mode": "rdflib_claim_evidence_graph", "triple_count": len(graph), "subject_count": len(set(graph.subjects())), "predicate_count": len(set(graph.predicates())), "object_count": len(set(graph.objects())), "ntriples": graph.serialize(format="nt").splitlines()[:MAX_EDGES], "engine": {"rdflib": _package("rdflib")}, "remote_graph_loading_allowed": False}


def pyshacl_graph_validation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("pyshacl")
    from pyshacl import validate
    from rdflib import Graph
    data_turtle = _text(inputs.get("data_turtle"), "inputs.data_turtle", 100_000); shapes_turtle = _text(inputs.get("shapes_turtle"), "inputs.shapes_turtle", 100_000)
    if any(token in data_turtle + shapes_turtle for token in ("http://", "https://", "file:")): raise ComputeError("remote or file IRIs are forbidden in graph validation")
    data_graph = Graph().parse(data=data_turtle, format="turtle"); shapes_graph = Graph().parse(data=shapes_turtle, format="turtle")
    if len(data_graph) > MAX_EDGES or len(shapes_graph) > MAX_EDGES: raise ComputeError("graphs exceed the governed triple limit")
    conforms, report_graph, report_text = validate(data_graph=data_graph, shacl_graph=shapes_graph, inference="none", abort_on_first=False, allow_infos=False, allow_warnings=False, meta_shacl=False, advanced=False, js=False)
    return {"mode": "pyshacl_graph_validation", "conforms": bool(conforms), "data_triples": len(data_graph), "shape_triples": len(shapes_graph), "report_triples": len(report_graph), "report": str(report_text)[:20_000], "engine": {"pyshacl": _package("pyshacl"), "rdflib": _package("rdflib")}, "remote_imports_allowed": False}


def owlready2_ontology_summary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("owlready2")
    import owlready2
    classes = _names(inputs.get("classes"), "inputs.classes", 1, 500); relations = _sequence(inputs.get("subclass_relations") or [], "inputs.subclass_relations")
    world = owlready2.World(); ontology = world.get_ontology("urn:governed:ontology"); created = {}
    with ontology:
        for name in classes: created[name] = type(name.replace(" ", "_"), (owlready2.Thing,), {})
        for raw in relations:
            pair = _sequence(raw, "inputs.subclass_relations[]")
            if len(pair) != 2 or pair[0] not in created or pair[1] not in created: raise ComputeError("subclass relations must reference declared classes")
            created[pair[0]].is_a.append(created[pair[1]])
    return {"mode": "owlready2_ontology_summary", "class_count": len(list(ontology.classes())), "subclass_relation_count": len(relations), "classes": sorted(classes), "engine": {"owlready2": _package("owlready2")}, "external_ontology_imports_allowed": False, "java_reasoner_used": False}


def igraph_link_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("python-igraph")
    import igraph as ig
    nodes = _names(inputs.get("nodes"), "inputs.nodes", 1, MAX_NODES); edges_raw = _sequence(inputs.get("edges"), "inputs.edges")
    if not 0 <= len(edges_raw) <= MAX_EDGES: raise ComputeError("edges exceed the governed limit")
    index = {name: i for i, name in enumerate(nodes)}; edges = []
    for raw in edges_raw:
        pair = _sequence(raw, "inputs.edges[]")
        if len(pair) != 2 or pair[0] not in index or pair[1] not in index: raise ComputeError("edges must reference declared nodes")
        edges.append((index[pair[0]], index[pair[1]]))
    directed = bool(inputs.get("directed", True)); graph = ig.Graph(n=len(nodes), edges=edges, directed=directed); pagerank = graph.pagerank(directed=directed); betweenness = graph.betweenness(directed=directed)
    ranking = sorted(({"node": nodes[i], "pagerank": float(pagerank[i]), "betweenness": float(betweenness[i])} for i in range(len(nodes))), key=lambda row: (-row["pagerank"], -row["betweenness"], row["node"]))
    return {"mode": "igraph_link_analysis", "node_count": graph.vcount(), "edge_count": graph.ecount(), "ranking": ranking, "components": [sorted(nodes[i] for i in component) for component in graph.components(mode="weak")], "engine": {"python-igraph": _package("python-igraph")}}


def problog_evidence_probability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("problog")
    from problog import get_evaluatable
    from problog.program import PrologString
    facts = _sequence(inputs.get("facts"), "inputs.facts")
    if not 1 <= len(facts) <= 100: raise ComputeError("facts must contain 1 to 100 entries")
    lines = []; safe_names = []
    for index, raw in enumerate(facts):
        row = _mapping(raw, f"inputs.facts[{index}]"); name = _text(row.get("name"), f"inputs.facts[{index}].name", 60)
        if not name.replace("_", "").isalnum() or not name[0].isalpha(): raise ComputeError("fact names must be alphanumeric identifiers")
        probability = _finite(row.get("probability"), f"inputs.facts[{index}].probability")
        if not 0 <= probability <= 1: raise ComputeError("fact probabilities must be between 0 and 1")
        safe = name.casefold(); safe_names.append(safe); lines.append(f"{probability}::{safe}.")
    lines.extend([f"all_evidence :- {','.join(safe_names)}.", "query(all_evidence)."])
    result = get_evaluatable().create_from(PrologString("\n".join(lines))).evaluate()
    return {"mode": "problog_evidence_probability", "joint_probability": float(next(iter(result.values()))), "fact_count": len(facts), "engine": {"problog": _package("problog")}, "ticket_supplied_logic_program_allowed": False, "independence_assumption": True}


def issue_tree_coverage(inputs: Mapping[str, Any]) -> dict[str, Any]:
    root = _text(inputs.get("root"), "inputs.root"); branches = _sequence(inputs.get("branches"), "inputs.branches"); seen = set(); total_weight = covered_weight = 0.0; rows = []
    for index, raw in enumerate(branches):
        row = _mapping(raw, f"inputs.branches[{index}]"); name = _text(row.get("name"), f"inputs.branches[{index}].name"); weight = _finite(row.get("weight", 1.0), f"inputs.branches[{index}].weight"); evidence_count = _integer(row.get("evidence_count", 0), f"inputs.branches[{index}].evidence_count", 0, 10_000)
        if name in seen or weight < 0: raise ComputeError("issue-tree branches must be unique with non-negative weights")
        seen.add(name); total_weight += weight; covered_weight += weight if evidence_count > 0 else 0.0; rows.append({"branch": name, "weight": weight, "evidence_count": evidence_count, "covered": evidence_count > 0})
    if not 2 <= len(rows) <= 100 or total_weight <= 0: raise ComputeError("issue tree requires 2 to 100 positive-total branches")
    return {"mode": "issue_tree_coverage", "root": root, "branches": rows, "weighted_coverage": covered_weight / total_weight, "uncovered_branches": [row["branch"] for row in rows if not row["covered"]], "method": "MECE-oriented coverage audit; semantic overlap still requires human review"}


def value_driver_tree(inputs: Mapping[str, Any]) -> dict[str, Any]:
    base_value = _finite(inputs.get("base_value"), "inputs.base_value"); drivers = _sequence(inputs.get("drivers"), "inputs.drivers"); rows = []; total_change = 0.0
    for index, raw in enumerate(drivers):
        row = _mapping(raw, f"inputs.drivers[{index}]"); name = _text(row.get("name"), f"inputs.drivers[{index}].name"); change = _finite(row.get("change"), f"inputs.drivers[{index}].change"); multiplier = _finite(row.get("multiplier", 1.0), f"inputs.drivers[{index}].multiplier"); contribution = change * multiplier; total_change += contribution; rows.append({"driver": name, "change": change, "multiplier": multiplier, "value_contribution": contribution})
    if not 1 <= len(rows) <= 100: raise ComputeError("drivers must contain 1 to 100 entries")
    rows.sort(key=lambda row: (-abs(row["value_contribution"]), row["driver"]))
    return {"mode": "value_driver_tree", "base_value": base_value, "projected_value": base_value + total_change, "total_change": total_change, "drivers": rows}


def source_reliability_matrix(inputs: Mapping[str, Any]) -> dict[str, Any]:
    sources = _sequence(inputs.get("sources"), "inputs.sources"); rows = []
    for index, raw in enumerate(sources):
        row = _mapping(raw, f"inputs.sources[{index}]"); name = _text(row.get("name"), f"inputs.sources[{index}].name"); reliability = _finite(row.get("reliability"), f"inputs.sources[{index}].reliability"); access = _finite(row.get("access"), f"inputs.sources[{index}].access"); corroboration = _finite(row.get("corroboration"), f"inputs.sources[{index}].corroboration"); recency = _finite(row.get("recency"), f"inputs.sources[{index}].recency")
        if any(not 0 <= value <= 1 for value in (reliability, access, corroboration, recency)): raise ComputeError("source dimensions must be between 0 and 1")
        rows.append({"source": name, "score": 0.35 * reliability + 0.2 * access + 0.3 * corroboration + 0.15 * recency, "dimensions": {"reliability": reliability, "access": access, "corroboration": corroboration, "recency": recency}})
    if not 1 <= len(rows) <= 500: raise ComputeError("sources must contain 1 to 500 entries")
    rows.sort(key=lambda row: (-row["score"], row["source"]))
    return {"mode": "source_reliability_matrix", "ranking": rows, "single_source_decision_allowed": False}


def claim_evidence_contradiction(inputs: Mapping[str, Any]) -> dict[str, Any]:
    claims = _names(inputs.get("claims"), "inputs.claims", 1, 200); evidence = _sequence(inputs.get("evidence"), "inputs.evidence"); matrix = defaultdict(lambda: {"support": 0.0, "contradict": 0.0, "neutral": 0.0})
    for index, raw in enumerate(evidence):
        row = _mapping(raw, f"inputs.evidence[{index}]"); claim = _text(row.get("claim"), f"inputs.evidence[{index}].claim"); stance = str(row.get("stance") or ""); weight = _finite(row.get("weight", 1.0), f"inputs.evidence[{index}].weight")
        if claim not in claims or stance not in {"support", "contradict", "neutral"} or weight < 0: raise ComputeError("evidence claim/stance/weight is invalid")
        matrix[claim][stance] += weight
    rows = []
    for claim in claims:
        values = matrix[claim]; total = sum(values.values()); conflict = min(values["support"], values["contradict"]) / max(total, 1e-12); rows.append({"claim": claim, **values, "conflict_ratio": conflict, "unresolved": total == 0 or conflict > 0.2})
    return {"mode": "claim_evidence_contradiction", "claims": rows, "unresolved_claims": [r["claim"] for r in rows if r["unresolved"]]}


def event_timeline_collision(inputs: Mapping[str, Any]) -> dict[str, Any]:
    events = _sequence(inputs.get("events"), "inputs.events")
    if not 2 <= len(events) <= MAX_EVENTS: raise ComputeError(f"events must contain 2 to {MAX_EVENTS} entries")
    normalized = []
    for index, raw in enumerate(events):
        row = _mapping(raw, f"inputs.events[{index}]"); event = {"id": _text(row.get("id"), f"inputs.events[{index}].id"), "entity": _text(row.get("entity"), f"inputs.events[{index}].entity"), "start": _finite(row.get("start"), f"inputs.events[{index}].start"), "end": _finite(row.get("end"), f"inputs.events[{index}].end"), "location": str(row.get("location") or "").strip()}
        if event["start"] > event["end"]: raise ComputeError("event start cannot exceed end")
        normalized.append(event)
    collisions = []
    for i, left in enumerate(normalized):
        for right in normalized[i + 1:]:
            if max(left["start"], right["start"]) <= min(left["end"], right["end"]) and left["entity"] == right["entity"] and left["location"] and right["location"] and left["location"] != right["location"]: collisions.append({"left": left["id"], "right": right["id"], "reason": "same-entity-overlapping-different-location"})
    return {"mode": "event_timeline_collision", "collision_count": len(collisions), "collisions": collisions}


def red_team_challenge_matrix(inputs: Mapping[str, Any]) -> dict[str, Any]:
    assumptions = _sequence(inputs.get("assumptions"), "inputs.assumptions"); rows = []
    for index, raw in enumerate(assumptions):
        row = _mapping(raw, f"inputs.assumptions[{index}]"); name = _text(row.get("name"), f"inputs.assumptions[{index}].name"); impact = _finite(row.get("impact"), f"inputs.assumptions[{index}].impact"); uncertainty = _finite(row.get("uncertainty"), f"inputs.assumptions[{index}].uncertainty"); reversibility = _finite(row.get("reversibility", 0.5), f"inputs.assumptions[{index}].reversibility")
        if any(not 0 <= value <= 1 for value in (impact, uncertainty, reversibility)): raise ComputeError("red-team dimensions must be between 0 and 1")
        rows.append({"assumption": name, "challenge_priority": impact * uncertainty * (1.0 - 0.5 * reversibility), "impact": impact, "uncertainty": uncertainty, "reversibility": reversibility})
    if not 1 <= len(rows) <= 200: raise ComputeError("assumptions must contain 1 to 200 entries")
    rows.sort(key=lambda row: (-row["challenge_priority"], row["assumption"]))
    return {"mode": "red_team_challenge_matrix", "ranking": rows, "highest_priority": rows[0]["assumption"]}


def net_assessment_balance(inputs: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = _sequence(inputs.get("dimensions"), "inputs.dimensions"); rows = []; net = 0.0
    for index, raw in enumerate(dimensions):
        row = _mapping(raw, f"inputs.dimensions[{index}]"); name = _text(row.get("name"), f"inputs.dimensions[{index}].name"); own = _finite(row.get("own"), f"inputs.dimensions[{index}].own"); competitor = _finite(row.get("competitor"), f"inputs.dimensions[{index}].competitor"); weight = _finite(row.get("weight", 1.0), f"inputs.dimensions[{index}].weight")
        if weight < 0: raise ComputeError("weights must be non-negative")
        balance = (own - competitor) * weight; net += balance; rows.append({"dimension": name, "own": own, "competitor": competitor, "weight": weight, "weighted_balance": balance})
    if not 1 <= len(rows) <= 100: raise ComputeError("dimensions must contain 1 to 100 entries")
    rows.sort(key=lambda row: (-abs(row["weighted_balance"]), row["dimension"]))
    return {"mode": "net_assessment_balance", "net_balance": net, "dimensions": rows, "forecast_claim": False}


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "open_spiel_policy_evaluation": open_spiel_policy_evaluation,
    "pygambit_pure_equilibria": pygambit_pure_equilibria,
    "axelrod_strategy_tournament": axelrod_strategy_tournament,
    "negmas_bilateral_bargaining": negmas_bilateral_bargaining,
    "scml_supply_chain_competition": scml_supply_chain_competition,
    "pyblp_price_counterfactual": pyblp_price_counterfactual,
    "pymc_marketing_budget_allocation": pymc_marketing_budget_allocation,
    "biogeme_choice_share": biogeme_choice_share,
    "pyagrum_bayesian_evidence": pyagrum_bayesian_evidence,
    "scikit_criteria_method_agreement": scikit_criteria_method_agreement,
    "clingo_rule_action_set": clingo_rule_action_set,
    "z3_constraint_counterexample": z3_constraint_counterexample,
    "hark_household_policy_response": hark_household_policy_response,
    "taxcalc_policy_counterfactual": taxcalc_policy_counterfactual,
    "policyengine_transfer_counterfactual": policyengine_transfer_counterfactual,
    "splink_entity_resolution": splink_entity_resolution,
    "rapidfuzz_record_collision": rapidfuzz_record_collision,
    "datasketch_set_similarity": datasketch_set_similarity,
    "rdflib_claim_evidence_graph": rdflib_claim_evidence_graph,
    "pyshacl_graph_validation": pyshacl_graph_validation,
    "owlready2_ontology_summary": owlready2_ontology_summary,
    "igraph_link_analysis": igraph_link_analysis,
    "problog_evidence_probability": problog_evidence_probability,
    "issue_tree_coverage": issue_tree_coverage,
    "value_driver_tree": value_driver_tree,
    "source_reliability_matrix": source_reliability_matrix,
    "claim_evidence_contradiction": claim_evidence_contradiction,
    "event_timeline_collision": event_timeline_collision,
    "red_team_challenge_matrix": red_team_challenge_matrix,
    "net_assessment_balance": net_assessment_balance,
}

SUPPORTED_MODES = tuple(sorted(HANDLERS))


def strategic_policy_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handler = HANDLERS.get(mode)
    if handler is None:
        raise ComputeError(f"unsupported strategic policy mode: {mode}")
    result = handler(inputs)
    result["offline_execution"] = True
    result["external_data_fetches"] = 0
    result["model_calls"] = 0
    result["decision_support_only"] = True
    return result


OPERATIONS = {"strategic_policy_analysis": strategic_policy_analysis}
