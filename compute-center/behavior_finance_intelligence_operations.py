#!/usr/bin/env python3
"""Final governed behavioral, quantitative-finance, and intelligence modes.

All handlers use bounded structured inputs and execute offline. They do not
collect live data, diagnose individuals, target persons, run autonomous agents,
place trades, call remote models, or accept ticket-supplied code.
"""
from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError

MAX_ROWS = 5_000
MAX_COLUMNS = 100
MAX_STEPS = 10_000
MAX_EVENTS = 10_000
MAX_ITEMS = 1_000


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
    if len(result) != len(set(result)):
        raise ComputeError(f"{name} entries must be unique")
    return result


def _vector(value: Any, name: str, minimum: int, maximum: int) -> np.ndarray:
    rows = _sequence(value, name)
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} values")
    vector = np.asarray([_finite(item, f"{name}[]") for item in rows], dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ComputeError(f"{name} must be finite")
    return vector


def _probability_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    vector = _vector(value, name, 1, MAX_ROWS)
    if length is not None and vector.size != length:
        raise ComputeError(f"{name} length must be {length}")
    if np.any(vector < 0) or vector.sum() <= 0:
        raise ComputeError(f"{name} must be non-negative with positive total")
    return vector / vector.sum()


def _matrix(value: Any, name: str, minimum_rows: int = 2, maximum_rows: int = MAX_ROWS) -> np.ndarray:
    rows = _sequence(value, name)
    if not minimum_rows <= len(rows) <= maximum_rows:
        raise ComputeError(f"{name} row count is outside the governed range")
    converted = []
    width = None
    for index, row in enumerate(rows):
        vector = _vector(row, f"{name}[{index}]", 1, MAX_COLUMNS)
        width = width or vector.size
        if vector.size != width:
            raise ComputeError(f"{name} rows must have equal length")
        converted.append(vector)
    matrix = np.asarray(converted, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] > MAX_COLUMNS:
        raise ComputeError(f"{name} has invalid dimensions")
    return matrix


def _package(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ComputeError(f"required capability pack is not installed: {distribution}") from exc


def _iso_date(value: Any, name: str) -> date:
    text = _text(value, name, 10)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ComputeError(f"{name} must be YYYY-MM-DD") from exc
    if not 1901 <= parsed.year <= 2199:
        raise ComputeError(f"{name} year is outside the governed range")
    return parsed


def quantlib_option_greeks(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("QuantLib")
    import QuantLib as ql

    observed = _iso_date(inputs.get("evaluation_date"), "inputs.evaluation_date")
    today = ql.Date(observed.day, observed.month, observed.year)
    ql.Settings.instance().evaluationDate = today
    maturity_days = _integer(inputs.get("maturity_days"), "inputs.maturity_days", 1, 36_500)
    spot_value = _finite(inputs.get("spot"), "inputs.spot")
    strike = _finite(inputs.get("strike"), "inputs.strike")
    risk_free = _finite(inputs.get("risk_free_rate", 0.03), "inputs.risk_free_rate")
    dividend = _finite(inputs.get("dividend_yield", 0.0), "inputs.dividend_yield")
    volatility_value = _finite(inputs.get("volatility"), "inputs.volatility")
    option_type = str(inputs.get("option_type") or "call").strip().lower()
    if spot_value <= 0 or strike <= 0 or not -1 <= risk_free <= 2 or not -1 <= dividend <= 2:
        raise ComputeError("spot/strike/rates are outside governed ranges")
    if not 0 < volatility_value <= 5 or option_type not in {"call", "put"}:
        raise ComputeError("volatility or option_type is invalid")

    maturity = today + maturity_days
    payoff_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(payoff_type, strike),
        ql.EuropeanExercise(maturity),
    )
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(spot_value))
    risk_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free, day_count))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, day_count))
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), volatility_value, day_count)
    )
    process = ql.BlackScholesMertonProcess(spot, dividend_curve, risk_curve, volatility)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    metrics = {
        "npv": float(option.NPV()),
        "delta": float(option.delta()),
        "gamma": float(option.gamma()),
        "vega": float(option.vega()),
        "theta": float(option.theta()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ComputeError("QuantLib returned non-finite option metrics")
    return {
        "mode": "quantlib_option_greeks",
        "option_type": option_type,
        "evaluation_date": observed.isoformat(),
        "maturity_days": maturity_days,
        **metrics,
        "engine": {"QuantLib": _package("QuantLib")},
        "live_market_data_used": False,
        "brokerage_execution_used": False,
    }


def quantlib_bond_duration(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("QuantLib")
    import QuantLib as ql

    observed = _iso_date(inputs.get("evaluation_date"), "inputs.evaluation_date")
    today = ql.Date(observed.day, observed.month, observed.year)
    ql.Settings.instance().evaluationDate = today
    maturity_years = _integer(inputs.get("maturity_years"), "inputs.maturity_years", 1, 100)
    settlement_days = _integer(inputs.get("settlement_days", 2), "inputs.settlement_days", 0, 10)
    face_value = _finite(inputs.get("face_value", 100.0), "inputs.face_value")
    coupon_rate = _finite(inputs.get("coupon_rate"), "inputs.coupon_rate")
    market_yield = _finite(inputs.get("market_yield"), "inputs.market_yield")
    if face_value <= 0 or not 0 <= coupon_rate <= 1 or not -0.1 <= market_yield <= 1:
        raise ComputeError("bond inputs are outside governed ranges")

    maturity = ql.Date(observed.day, observed.month, observed.year + maturity_years)
    day_count = ql.ActualActual(ql.ActualActual.Bond)
    schedule = ql.Schedule(
        today,
        maturity,
        ql.Period(ql.Annual),
        ql.NullCalendar(),
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Backward,
        False,
    )
    bond = ql.FixedRateBond(settlement_days, face_value, schedule, [coupon_rate], day_count)
    interest_rate = ql.InterestRate(market_yield, day_count, ql.Compounded, ql.Annual)
    clean_price = float(ql.BondFunctions.cleanPrice(bond, interest_rate))
    duration = float(ql.BondFunctions.duration(bond, interest_rate, ql.Duration.Modified))
    convexity = float(ql.BondFunctions.convexity(bond, interest_rate))
    metrics = (clean_price, duration, convexity)
    if not all(math.isfinite(value) for value in metrics):
        raise ComputeError("QuantLib returned non-finite bond metrics")
    return {
        "mode": "quantlib_bond_duration",
        "evaluation_date": observed.isoformat(),
        "maturity_years": maturity_years,
        "clean_price_per_100": clean_price,
        "modified_duration": duration,
        "convexity": convexity,
        "engine": {"QuantLib": _package("QuantLib")},
        "live_market_data_used": False,
        "brokerage_execution_used": False,
    }


def active_inference_policy_choice(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("inferactively-pymdp")
    from pymdp import utils
    from pymdp.agent import Agent

    likelihood = _matrix(inputs.get("observation_likelihood"), "inputs.observation_likelihood", 2, 20)
    if likelihood.shape[0] > 20 or likelihood.shape[1] > 20:
        raise ComputeError("observation likelihood is outside governed dimensions")
    column_sums = likelihood.sum(axis=0)
    if np.any(likelihood < 0) or not np.allclose(column_sums, 1.0, atol=1e-8):
        raise ComputeError("observation_likelihood columns must be probability vectors")

    raw_transitions = _sequence(inputs.get("transition_matrices"), "inputs.transition_matrices")
    if not 1 <= len(raw_transitions) <= 20:
        raise ComputeError("transition_matrices must contain 1 to 20 actions")
    transitions = []
    for index, raw in enumerate(raw_transitions):
        matrix = _matrix(raw, f"inputs.transition_matrices[{index}]", 2, 20)
        if matrix.shape != (likelihood.shape[1], likelihood.shape[1]):
            raise ComputeError("transition matrices must match hidden-state dimensions")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
            raise ComputeError("transition matrix columns must be probability vectors")
        transitions.append(matrix)
    preferences = _vector(inputs.get("preferences"), "inputs.preferences", likelihood.shape[0], likelihood.shape[0])
    prior = _probability_vector(inputs.get("prior"), "inputs.prior", likelihood.shape[1])
    observation = _integer(
        inputs.get("observation"),
        "inputs.observation",
        0,
        likelihood.shape[0] - 1,
    )

    from jax import numpy as jnp

    A = [jnp.asarray(likelihood, dtype=jnp.float32)]
    B = [jnp.asarray(np.stack(transitions, axis=2), dtype=jnp.float32)]
    C = [jnp.asarray(preferences, dtype=jnp.float32)]
    D = [jnp.asarray(prior, dtype=jnp.float32)]
    agent = Agent(A=A, B=B, C=C, D=D, policy_len=1, batch_size=1)
    posterior_states, inference_info = agent.infer_states(
        [jnp.asarray([observation], dtype=jnp.int32)],
        empirical_prior=agent.D,
        return_info=True,
    )
    policy_probabilities, negative_expected_free_energy = agent.infer_policies(posterior_states)
    q_pi = np.asarray(policy_probabilities, dtype=float).reshape(agent.batch_size, -1)[0]
    chosen_policy = int(np.argmax(q_pi))
    policy_array = np.asarray(agent.policies.policy_arr, dtype=int)
    chosen_action = int(policy_array[chosen_policy, 0, 0])
    posterior = np.asarray(posterior_states[0], dtype=float)[0, -1]
    negative_efe = np.asarray(negative_expected_free_energy, dtype=float).reshape(agent.batch_size, -1)[0]
    vfe = np.asarray(inference_info["vfe"], dtype=float).reshape(-1)
    return {
        "mode": "active_inference_policy_choice",
        "observation": observation,
        "chosen_policy_index": chosen_policy,
        "chosen_action": chosen_action,
        "posterior_states": posterior.tolist(),
        "policy_probabilities": q_pi.tolist(),
        "negative_expected_free_energy": negative_efe.tolist(),
        "variational_free_energy": vfe.tolist(),
        "engine": {"inferactively-pymdp": _package("inferactively-pymdp")},
        "policy_horizon": 1,
        "autonomous_loop_used": False,
        "individual_diagnosis_allowed": False,
    }


def pyod_anomaly_screen(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("pyod")
    records = _matrix(inputs.get("records"), "inputs.records", 5, MAX_ROWS)
    contamination = _finite(inputs.get("contamination", 0.05), "inputs.contamination")
    detector_name = str(inputs.get("detector") or "ecod").strip().lower()
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**31 - 1)
    if not 0.001 <= contamination <= 0.5:
        raise ComputeError("contamination must be between 0.001 and 0.5")

    if detector_name == "ecod":
        from pyod.models.ecod import ECOD

        detector = ECOD(contamination=contamination)
    elif detector_name == "iforest":
        from pyod.models.iforest import IForest

        detector = IForest(
            n_estimators=100,
            contamination=contamination,
            random_state=seed,
            n_jobs=1,
        )
    elif detector_name == "knn":
        from pyod.models.knn import KNN

        neighbours = _integer(
            inputs.get("n_neighbors", min(5, records.shape[0] - 1)),
            "inputs.n_neighbors",
            1,
            min(100, records.shape[0] - 1),
        )
        detector = KNN(contamination=contamination, n_neighbors=neighbours, n_jobs=1)
    else:
        raise ComputeError("detector must be one of: ecod, iforest, knn")

    detector.fit(records)
    scores = np.asarray(detector.decision_scores_, dtype=float).reshape(-1)
    labels = np.asarray(detector.labels_, dtype=int).reshape(-1)
    if scores.size != records.shape[0] or labels.size != records.shape[0]:
        raise ComputeError("PyOD returned an unexpected result shape")
    ranking = np.argsort(-scores, kind="stable")
    top_limit = min(100, records.shape[0])
    top = [
        {"row_index": int(index), "score": float(scores[index]), "label": int(labels[index])}
        for index in ranking[:top_limit]
    ]
    return {
        "mode": "pyod_anomaly_screen",
        "detector": detector_name,
        "row_count": int(records.shape[0]),
        "column_count": int(records.shape[1]),
        "anomaly_count": int(labels.sum()),
        "top_rows": top,
        "engine": {"pyod": _package("pyod")},
        "allowlisted_detectors": ["ecod", "iforest", "knn"],
        "agentic_engine_used": False,
        "mcp_used": False,
        "automatic_enforcement_allowed": False,
    }


def market_basket_association_rules(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _package("mlxtend")
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    raw_transactions = _sequence(inputs.get("transactions"), "inputs.transactions")
    if not 2 <= len(raw_transactions) <= MAX_ROWS:
        raise ComputeError(f"transactions must contain 2 to {MAX_ROWS} rows")
    transactions: list[list[str]] = []
    universe: set[str] = set()
    for index, raw in enumerate(raw_transactions):
        items = _names(raw, f"inputs.transactions[{index}]", 1, 100)
        universe.update(items)
        transactions.append(items)
    if len(universe) > MAX_ITEMS:
        raise ComputeError(f"transaction universe exceeds {MAX_ITEMS} items")
    min_support = _finite(inputs.get("min_support", 0.1), "inputs.min_support")
    min_confidence = _finite(inputs.get("min_confidence", 0.5), "inputs.min_confidence")
    max_len = _integer(inputs.get("max_len", 5), "inputs.max_len", 1, 20)
    if not 0 < min_support <= 1 or not 0 <= min_confidence <= 1:
        raise ComputeError("support/confidence thresholds are invalid")

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    frame = pd.DataFrame(encoded, columns=encoder.columns_)
    frequent = apriori(frame, min_support=min_support, use_colnames=True, max_len=max_len)
    if frequent.empty:
        rules = pd.DataFrame()
    else:
        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=min_confidence,
        )
    frequent_rows = [
        {
            "items": sorted(str(item) for item in row.itemsets),
            "support": float(row.support),
        }
        for row in frequent.sort_values(["support"], ascending=False).itertuples(index=False)
    ]
    rule_rows = []
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
        for row in rules.head(1_000).itertuples(index=False):
            rule_rows.append(
                {
                    "antecedents": sorted(str(item) for item in row.antecedents),
                    "consequents": sorted(str(item) for item in row.consequents),
                    "support": float(row.support),
                    "confidence": float(row.confidence),
                    "lift": float(row.lift),
                }
            )
    return {
        "mode": "market_basket_association_rules",
        "transaction_count": len(transactions),
        "item_count": len(universe),
        "frequent_itemsets": frequent_rows[:2_000],
        "rules": rule_rows,
        "rules_truncated": len(rules) > 1_000 if not rules.empty else False,
        "engine": {"mlxtend": _package("mlxtend")},
        "causal_claim_allowed": False,
    }


def replicator_dynamics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    payoff = _matrix(inputs.get("payoff_matrix"), "inputs.payoff_matrix", 2, 50)
    if payoff.shape[0] != payoff.shape[1]:
        raise ComputeError("payoff_matrix must be square")
    population = _probability_vector(inputs.get("initial_population"), "inputs.initial_population", payoff.shape[0])
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    dt = _finite(inputs.get("dt", 0.05), "inputs.dt")
    if not 0 < dt <= 1:
        raise ComputeError("dt must be between 0 and 1")
    trajectory = [population.tolist()]
    for _ in range(steps):
        fitness = payoff @ population
        mean_fitness = float(population @ fitness)
        centered = np.clip(fitness - mean_fitness, -50.0, 50.0)
        population = population * np.exp(dt * centered)
        total = float(population.sum())
        if total <= 0 or not math.isfinite(total):
            raise ComputeError("replicator dynamics became numerically invalid")
        population /= total
        trajectory.append(population.tolist())
    return {
        "mode": "replicator_dynamics",
        "strategy_count": int(payoff.shape[0]),
        "steps": steps,
        "final_population": population.tolist(),
        "dominant_strategy_index": int(np.argmax(population)),
        "trajectory": trajectory,
        "method": "bounded multiplicative replicator update",
        "external_engine_used": False,
    }


def finite_population_fixation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    population_size = _integer(inputs.get("population_size"), "inputs.population_size", 2, 1_000_000)
    initial_mutants = _integer(
        inputs.get("initial_mutants", 1),
        "inputs.initial_mutants",
        1,
        population_size - 1,
    )
    relative_fitness = _finite(inputs.get("relative_fitness"), "inputs.relative_fitness")
    if not 0 < relative_fitness <= 100:
        raise ComputeError("relative_fitness must be between 0 and 100")
    if abs(relative_fitness - 1.0) < 1e-12:
        probability = initial_mutants / population_size
    else:
        numerator = 1.0 - relative_fitness ** (-initial_mutants)
        denominator = 1.0 - relative_fitness ** (-population_size)
        probability = numerator / denominator
    probability = float(np.clip(probability, 0.0, 1.0))
    return {
        "mode": "finite_population_fixation",
        "population_size": population_size,
        "initial_mutants": initial_mutants,
        "relative_fitness": relative_fitness,
        "fixation_probability": probability,
        "neutral_probability": initial_mutants / population_size,
        "method": "Moran-process closed form under constant relative fitness",
    }


def prospect_theory_choice(inputs: Mapping[str, Any]) -> dict[str, Any]:
    options = _sequence(inputs.get("options"), "inputs.options")
    if not 2 <= len(options) <= 100:
        raise ComputeError("options must contain 2 to 100 entries")
    reference = _finite(inputs.get("reference_point", 0.0), "inputs.reference_point")
    gain_curvature = _finite(inputs.get("gain_curvature", 0.88), "inputs.gain_curvature")
    loss_curvature = _finite(inputs.get("loss_curvature", 0.88), "inputs.loss_curvature")
    loss_aversion = _finite(inputs.get("loss_aversion", 2.25), "inputs.loss_aversion")
    probability_weighting = _finite(inputs.get("probability_weighting", 0.65), "inputs.probability_weighting")
    if not 0 < gain_curvature <= 1 or not 0 < loss_curvature <= 1:
        raise ComputeError("curvature parameters must be between 0 and 1")
    if not 1 <= loss_aversion <= 20 or not 0 < probability_weighting <= 2:
        raise ComputeError("loss_aversion/probability_weighting is outside governed ranges")

    rows = []
    for index, raw in enumerate(options):
        option = _mapping(raw, f"inputs.options[{index}]")
        name = _text(option.get("name"), f"inputs.options[{index}].name")
        outcomes = _vector(option.get("outcomes"), f"inputs.options[{index}].outcomes", 1, 1_000)
        probabilities = _probability_vector(
            option.get("probabilities"),
            f"inputs.options[{index}].probabilities",
            outcomes.size,
        )
        transformed = []
        for outcome in outcomes:
            deviation = float(outcome - reference)
            if deviation >= 0:
                transformed.append(deviation**gain_curvature)
            else:
                transformed.append(-loss_aversion * ((-deviation) ** loss_curvature))
        weights = np.exp(-((-np.log(np.clip(probabilities, 1e-15, 1.0))) ** probability_weighting))
        weights = weights / weights.sum()
        score = float(weights @ np.asarray(transformed, dtype=float))
        rows.append(
            {
                "option": name,
                "prospect_score": score,
                "transformed_probabilities": weights.tolist(),
                "expected_value": float(probabilities @ outcomes),
            }
        )
    rows.sort(key=lambda row: (-row["prospect_score"], row["option"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {
        "mode": "prospect_theory_choice",
        "ranking": rows,
        "selected_option": rows[0]["option"],
        "reference_point": reference,
        "method_note": "bounded prospect-value approximation for scenario comparison, not individual diagnosis",
        "individual_prediction_allowed": False,
    }


def collective_action_threshold(inputs: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = _vector(inputs.get("thresholds"), "inputs.thresholds", 2, MAX_ROWS)
    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ComputeError("thresholds must be between 0 and 1")
    raw_initial = _sequence(inputs.get("initial_adopters"), "inputs.initial_adopters")
    if len(raw_initial) != thresholds.size or any(not isinstance(value, bool) for value in raw_initial):
        raise ComputeError("initial_adopters must be a boolean array matching thresholds")
    adopters = np.asarray(raw_initial, dtype=bool)
    external_support = _finite(inputs.get("external_support", 0.0), "inputs.external_support")
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    if not 0 <= external_support <= 1:
        raise ComputeError("external_support must be between 0 and 1")
    trajectory = [float(adopters.mean())]
    for _ in range(steps):
        signal = min(1.0, float(adopters.mean()) + external_support)
        updated = adopters | (thresholds <= signal)
        trajectory.append(float(updated.mean()))
        if np.array_equal(updated, adopters):
            adopters = updated
            break
        adopters = updated
    return {
        "mode": "collective_action_threshold",
        "population": int(thresholds.size),
        "initial_fraction": trajectory[0],
        "final_fraction": trajectory[-1],
        "cascade_size": int(adopters.sum()),
        "trajectory": trajectory,
        "method": "deterministic threshold-cascade scenario",
        "real_group_prediction_allowed": False,
    }


def rumor_correction_dynamics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rumor = _finite(inputs.get("initial_rumor"), "inputs.initial_rumor")
    correction = _finite(inputs.get("initial_correction"), "inputs.initial_correction")
    beta_rumor = _finite(inputs.get("rumor_spread_rate"), "inputs.rumor_spread_rate")
    beta_correction = _finite(inputs.get("correction_spread_rate"), "inputs.correction_spread_rate")
    conversion = _finite(inputs.get("correction_conversion_rate"), "inputs.correction_conversion_rate")
    forgetting = _finite(inputs.get("rumor_forgetting_rate", 0.0), "inputs.rumor_forgetting_rate")
    steps = _integer(inputs.get("steps", 100), "inputs.steps", 1, MAX_STEPS)
    dt = _finite(inputs.get("dt", 0.05), "inputs.dt")
    if rumor < 0 or correction < 0 or rumor + correction > 1:
        raise ComputeError("initial rumor/correction fractions are invalid")
    if any(value < 0 or value > 20 for value in (beta_rumor, beta_correction, conversion, forgetting)):
        raise ComputeError("dynamics rates must be between 0 and 20")
    if not 0 < dt <= 1:
        raise ComputeError("dt must be between 0 and 1")
    trajectory = []
    uninformed = 1.0 - rumor - correction
    for step in range(steps + 1):
        trajectory.append(
            {
                "step": step,
                "uninformed": uninformed,
                "rumor": rumor,
                "correction": correction,
            }
        )
        if step == steps:
            break
        new_rumor = beta_rumor * uninformed * rumor
        new_correction = beta_correction * uninformed * correction
        converted = conversion * rumor * correction
        forgotten = forgetting * rumor
        rumor = max(0.0, rumor + dt * (new_rumor - converted - forgotten))
        correction = max(0.0, correction + dt * (new_correction + converted))
        total = rumor + correction
        if total > 1:
            rumor /= total
            correction /= total
        uninformed = max(0.0, 1.0 - rumor - correction)
    return {
        "mode": "rumor_correction_dynamics",
        "steps": steps,
        "final": trajectory[-1],
        "peak_rumor": max(row["rumor"] for row in trajectory),
        "trajectory": trajectory,
        "method": "bounded mean-field competing-diffusion scenario",
        "persuasion_targeting_allowed": False,
        "automatic_information_operation_allowed": False,
    }


def trust_reputation_update(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actors = _names(inputs.get("actors"), "inputs.actors", 2, 500)
    trust = _matrix(inputs.get("initial_trust"), "inputs.initial_trust", len(actors), len(actors))
    if trust.shape != (len(actors), len(actors)) or np.any((trust < 0) | (trust > 1)):
        raise ComputeError("initial_trust must be an actor-by-actor matrix between 0 and 1")
    learning_rate = _finite(inputs.get("learning_rate", 0.2), "inputs.learning_rate")
    if not 0 < learning_rate <= 1:
        raise ComputeError("learning_rate must be between 0 and 1")
    events = _sequence(inputs.get("events"), "inputs.events")
    if not 1 <= len(events) <= MAX_EVENTS:
        raise ComputeError(f"events must contain 1 to {MAX_EVENTS} rows")
    index = {actor: position for position, actor in enumerate(actors)}
    for event_index, raw in enumerate(events):
        event = _mapping(raw, f"inputs.events[{event_index}]")
        source = _text(event.get("source"), f"inputs.events[{event_index}].source")
        target = _text(event.get("target"), f"inputs.events[{event_index}].target")
        outcome = _finite(event.get("outcome"), f"inputs.events[{event_index}].outcome")
        weight = _finite(event.get("weight", 1.0), f"inputs.events[{event_index}].weight")
        if source not in index or target not in index or not 0 <= outcome <= 1 or not 0 < weight <= 1:
            raise ComputeError("trust event references or values are invalid")
        i, j = index[source], index[target]
        alpha = learning_rate * weight
        trust[i, j] = (1.0 - alpha) * trust[i, j] + alpha * outcome
    reputation = trust.mean(axis=0)
    ranking = sorted(
        (
            {"actor": actors[position], "reputation": float(reputation[position])}
            for position in range(len(actors))
        ),
        key=lambda row: (-row["reputation"], row["actor"]),
    )
    return {
        "mode": "trust_reputation_update",
        "actors": actors,
        "updated_trust": trust.tolist(),
        "reputation_ranking": ranking,
        "event_count": len(events),
        "method": "bounded exponentially weighted trust update",
        "individual_scoring_for_enforcement_allowed": False,
    }


def group_consensus_pressure(inputs: Mapping[str, Any]) -> dict[str, Any]:
    opinions = _vector(inputs.get("initial_opinions"), "inputs.initial_opinions", 2, MAX_ROWS)
    confidence = _vector(inputs.get("private_confidence"), "inputs.private_confidence", opinions.size, opinions.size)
    conformity = _vector(inputs.get("conformity"), "inputs.conformity", opinions.size, opinions.size)
    if np.any((opinions < -1) | (opinions > 1)):
        raise ComputeError("initial_opinions must be between -1 and 1")
    if np.any((confidence < 0) | (confidence > 1)) or np.any((conformity < 0) | (conformity > 1)):
        raise ComputeError("private_confidence and conformity must be between 0 and 1")
    leader_index = _integer(inputs.get("leader_index", 0), "inputs.leader_index", 0, opinions.size - 1)
    leader_weight = _finite(inputs.get("leader_weight", 1.0), "inputs.leader_weight")
    steps = _integer(inputs.get("steps", 50), "inputs.steps", 1, MAX_STEPS)
    if not 0 <= leader_weight <= 20:
        raise ComputeError("leader_weight must be between 0 and 20")
    private_anchor = opinions.copy()
    trajectory = [
        {"step": 0, "mean": float(opinions.mean()), "variance": float(opinions.var())}
    ]
    for step in range(1, steps + 1):
        weights = np.ones(opinions.size, dtype=float)
        weights[leader_index] = leader_weight
        group_signal = float(np.average(opinions, weights=weights))
        private_component = confidence * private_anchor + (1.0 - confidence) * opinions
        opinions = np.clip(
            (1.0 - conformity) * private_component + conformity * group_signal,
            -1.0,
            1.0,
        )
        trajectory.append(
            {"step": step, "mean": float(opinions.mean()), "variance": float(opinions.var())}
        )
    dissent_threshold = _finite(inputs.get("dissent_threshold", 0.25), "inputs.dissent_threshold")
    if not 0 <= dissent_threshold <= 2:
        raise ComputeError("dissent_threshold must be between 0 and 2")
    dissent = int(np.sum(np.abs(opinions - opinions.mean()) >= dissent_threshold))
    return {
        "mode": "group_consensus_pressure",
        "population": int(opinions.size),
        "final_opinions": opinions.tolist(),
        "final_mean": float(opinions.mean()),
        "final_variance": float(opinions.var()),
        "dissent_count": dissent,
        "variance_reduction": trajectory[0]["variance"] - trajectory[-1]["variance"],
        "trajectory": trajectory,
        "method": "bounded conformity-and-private-confidence scenario",
        "psychological_diagnosis_allowed": False,
        "real_group_prediction_allowed": False,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "quantlib_option_greeks": quantlib_option_greeks,
    "quantlib_bond_duration": quantlib_bond_duration,
    "active_inference_policy_choice": active_inference_policy_choice,
    "pyod_anomaly_screen": pyod_anomaly_screen,
    "market_basket_association_rules": market_basket_association_rules,
    "replicator_dynamics": replicator_dynamics,
    "finite_population_fixation": finite_population_fixation,
    "prospect_theory_choice": prospect_theory_choice,
    "collective_action_threshold": collective_action_threshold,
    "rumor_correction_dynamics": rumor_correction_dynamics,
    "trust_reputation_update": trust_reputation_update,
    "group_consensus_pressure": group_consensus_pressure,
}

SUPPORTED_MODES = tuple(sorted(HANDLERS))
