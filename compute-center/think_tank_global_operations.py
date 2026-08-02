#!/usr/bin/env python3
"""Bounded offline modes selected through Exa/Tavily global discovery and live compatibility tests."""
from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, integer, mapping, matrix, package, probability, sequence, vector

MAX_ROWS = 5_000
MAX_COLUMNS = 30
MAX_EVENTS = 10_000
MAX_LINKS = 5_000
MAX_OUTPUT_ROWS = 1_000


def _positive(value: Any, name: str, maximum: float = 1e12) -> float:
    result = finite(value, name)
    if not 0 < result <= maximum:
        raise ComputeError(f"{name} must be greater than zero and at most {maximum}")
    return result


def _safe_text(value: Any, name: str, maximum: int = 100) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        raise ComputeError(f"{name} must be a non-empty safe string of at most {maximum} characters")
    return text


def _bounded_matrix(value: Any, name: str, *, rows: int = MAX_ROWS, columns: int = MAX_COLUMNS) -> np.ndarray:
    return matrix(value, name, min_rows=1, max_rows=rows, min_columns=1, max_columns=columns)


def openturns_reliability_probability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("openturns")
    import openturns as ot

    mean = finite(inputs.get("mean", 0.0), "inputs.mean")
    standard_deviation = _positive(inputs.get("standard_deviation", 1.0), "inputs.standard_deviation")
    threshold = finite(inputs.get("threshold"), "inputs.threshold")
    tail = str(inputs.get("tail") or "lower").lower()
    if tail not in {"lower", "upper"}:
        raise ComputeError("inputs.tail must be lower or upper")
    distribution = ot.Normal(mean, standard_deviation)
    cdf = float(distribution.computeCDF(threshold))
    event_probability = cdf if tail == "lower" else 1.0 - cdf
    reliability = 1.0 - event_probability
    return {
        "mode": "openturns_reliability_probability",
        "distribution": "normal",
        "mean": mean,
        "standard_deviation": standard_deviation,
        "threshold": threshold,
        "failure_tail": tail,
        "failure_probability": event_probability,
        "reliability_probability": reliability,
        "reliability_index_standard_normal": float((threshold - mean) / standard_deviation),
        "engine": {"openturns": package("openturns")},
    }


def control_step_response(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("control")
    import control

    numerator = vector(inputs.get("numerator"), "inputs.numerator", minimum=1, maximum=10)
    denominator = vector(inputs.get("denominator"), "inputs.denominator", minimum=2, maximum=10)
    if abs(float(denominator[0])) < 1e-15:
        raise ComputeError("inputs.denominator leading coefficient must be non-zero")
    time_end = _positive(inputs.get("time_end", 10.0), "inputs.time_end", 100_000)
    points = integer(inputs.get("points", 101), "inputs.points", 10, 1_000)
    system = control.tf(numerator.tolist(), denominator.tolist())
    time_grid = np.linspace(0.0, time_end, points)
    time_values, response = control.step_response(system, T=time_grid)
    response = np.asarray(response, dtype=float).reshape(-1)
    if response.size != points or not np.all(np.isfinite(response)):
        raise ComputeError("control system produced a non-finite step response")
    final_value = float(response[-1])
    peak_index = int(np.argmax(response))
    peak_value = float(response[peak_index])
    overshoot_percent = 0.0
    if abs(final_value) > 1e-12:
        overshoot_percent = max(0.0, (peak_value - final_value) / abs(final_value) * 100.0)
    return {
        "mode": "control_step_response",
        "time": np.asarray(time_values, dtype=float).tolist(),
        "response": response.tolist(),
        "final_value": final_value,
        "peak_value": peak_value,
        "peak_time": float(time_values[peak_index]),
        "overshoot_percent": overshoot_percent,
        "stable_numeric_response": bool(np.max(np.abs(response)) < 1e12),
        "engine": {"control": package("control")},
    }


def pm4py_directly_follows(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pm4py")
    import pandas as pd
    import pm4py

    raw_cases = sequence(inputs.get("cases"), "inputs.cases")
    if not 1 <= len(raw_cases) <= 2_000:
        raise ComputeError("inputs.cases must contain 1 to 2000 cases")
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for case_index, raw_case in enumerate(raw_cases):
        case = mapping(raw_case, f"inputs.cases[{case_index}]")
        case_id = _safe_text(case.get("case_id"), f"inputs.cases[{case_index}].case_id", 80)
        activities = sequence(case.get("activities"), f"inputs.cases[{case_index}].activities")
        if not 1 <= len(activities) <= 200:
            raise ComputeError("each process case must contain 1 to 200 activities")
        for event_index, raw_activity in enumerate(activities):
            rows.append({
                "case:concept:name": case_id,
                "concept:name": _safe_text(raw_activity, "activity", 100),
                "time:timestamp": start + pd.Timedelta(days=case_index, seconds=event_index),
            })
            if len(rows) > MAX_EVENTS:
                raise ComputeError(f"process log may not exceed {MAX_EVENTS} events")
    frame = pd.DataFrame(rows)
    frame = pm4py.format_dataframe(
        frame,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    dfg, starts, ends = pm4py.discover_dfg(frame)
    edges = [
        {"source": str(edge[0]), "target": str(edge[1]), "count": int(count)}
        for edge, count in sorted(dfg.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
    ]
    return {
        "mode": "pm4py_directly_follows",
        "case_count": len(raw_cases),
        "event_count": len(rows),
        "activity_count": int(frame["concept:name"].nunique()),
        "directly_follows_edges": edges,
        "start_activities": {str(key): int(value) for key, value in sorted(starts.items())},
        "end_activities": {str(key): int(value) for key, value in sorted(ends.items())},
        "engine": {"pm4py": package("pm4py")},
    }


def smt_rbf_surrogate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("smt")
    from smt.surrogate_models import RBF

    train_x = _bounded_matrix(inputs.get("train_x"), "inputs.train_x", rows=1_000, columns=20)
    train_y = vector(inputs.get("train_y"), "inputs.train_y", minimum=2, maximum=1_000)
    predict_x = _bounded_matrix(inputs.get("predict_x"), "inputs.predict_x", rows=1_000, columns=20)
    if train_x.shape[0] != train_y.size or train_x.shape[1] != predict_x.shape[1]:
        raise ComputeError("train_x/train_y rows and train_x/predict_x columns must align")
    if train_x.shape[0] < train_x.shape[1] + 1:
        raise ComputeError("surrogate training requires at least feature_count + 1 observations")
    d0 = _positive(inputs.get("d0", 0.2), "inputs.d0", 1e6)
    regularization = finite(inputs.get("regularization", 1e-10), "inputs.regularization")
    if not 0 <= regularization <= 1:
        raise ComputeError("inputs.regularization must be between 0 and 1")
    model = RBF(d0=d0, reg=regularization, print_global=False)
    model.set_training_values(train_x, train_y.reshape(-1, 1))
    model.train()
    predictions = np.asarray(model.predict_values(predict_x), dtype=float).reshape(-1)
    fitted = np.asarray(model.predict_values(train_x), dtype=float).reshape(-1)
    if not np.all(np.isfinite(predictions)):
        raise ComputeError("surrogate predictions are non-finite")
    return {
        "mode": "smt_rbf_surrogate",
        "training_rows": int(train_x.shape[0]),
        "feature_count": int(train_x.shape[1]),
        "prediction_rows": int(predict_x.shape[0]),
        "predictions": predictions.tolist(),
        "training_rmse": float(np.sqrt(np.mean((fitted - train_y) ** 2))),
        "engine": {"smt": package("smt")},
    }


def econml_linear_dml(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("econml")
    from econml.dml import LinearDML
    from sklearn.linear_model import LinearRegression

    x = _bounded_matrix(inputs.get("x"), "inputs.x", rows=MAX_ROWS, columns=MAX_COLUMNS)
    treatment = vector(inputs.get("treatment"), "inputs.treatment", minimum=20, maximum=MAX_ROWS)
    outcome = vector(inputs.get("outcome"), "inputs.outcome", minimum=20, maximum=MAX_ROWS)
    if x.shape[0] != treatment.size or treatment.size != outcome.size:
        raise ComputeError("x, treatment and outcome must have equal row counts")
    folds = integer(inputs.get("folds", 3), "inputs.folds", 2, min(10, x.shape[0] // 4))
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    model = LinearDML(
        model_y=LinearRegression(),
        model_t=LinearRegression(),
        cv=folds,
        random_state=seed,
    )
    model.fit(outcome, treatment, X=x)
    effects = np.asarray(model.effect(x), dtype=float).reshape(-1)
    if not np.all(np.isfinite(effects)):
        raise ComputeError("EconML returned non-finite treatment effects")
    return {
        "mode": "econml_linear_dml",
        "observations": int(x.shape[0]),
        "features": int(x.shape[1]),
        "folds": folds,
        "average_treatment_effect": float(np.mean(effects)),
        "effect_standard_deviation": float(np.std(effects, ddof=1)) if effects.size > 1 else 0.0,
        "effect_minimum": float(np.min(effects)),
        "effect_maximum": float(np.max(effects)),
        "identification_warning": "causal interpretation requires unconfoundedness, overlap and correct nuisance-model assumptions",
        "engine": {"econml": package("econml")},
    }


def spreg_spatial_lag(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("spreg")
    package("libpysal")
    from libpysal.weights import W
    from spreg import ML_Lag

    y = vector(inputs.get("y"), "inputs.y", minimum=20, maximum=2_000)
    x = _bounded_matrix(inputs.get("x"), "inputs.x", rows=2_000, columns=20)
    if x.shape[0] != y.size:
        raise ComputeError("x and y must have equal row counts")
    raw_neighbors = mapping(inputs.get("neighbors"), "inputs.neighbors")
    neighbors: dict[int, list[int]] = {}
    weights: dict[int, list[float]] = {}
    for index in range(y.size):
        raw = raw_neighbors.get(str(index), raw_neighbors.get(index))
        if raw is None:
            raise ComputeError("neighbors must define every observation index")
        items = [int(value) for value in sequence(raw, f"inputs.neighbors[{index}]")]
        if not items or len(items) > 100 or any(item < 0 or item >= y.size or item == index for item in items):
            raise ComputeError("neighbor indices are invalid")
        neighbors[index] = items
        weights[index] = [1.0] * len(items)
    spatial_weights = W(neighbors, weights, silence_warnings=True)
    spatial_weights.transform = "r"
    model = ML_Lag(y.reshape(-1, 1), x, w=spatial_weights, method="ord")
    coefficients = np.asarray(model.betas, dtype=float).reshape(-1)
    if not np.all(np.isfinite(coefficients)) or not math.isfinite(float(model.rho)):
        raise ComputeError("spatial lag model produced non-finite estimates")
    return {
        "mode": "spreg_spatial_lag",
        "observations": int(y.size),
        "features": int(x.shape[1]),
        "spatial_lag_coefficient": float(model.rho),
        "coefficients": coefficients.tolist(),
        "log_likelihood": float(model.logll),
        "pseudo_r_squared": float(model.pr2),
        "engine": {"spreg": package("spreg"), "libpysal": package("libpysal")},
    }


def arch_garch_forecast(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("arch")
    from arch import arch_model

    returns = vector(inputs.get("returns"), "inputs.returns", minimum=50, maximum=20_000)
    p = integer(inputs.get("p", 1), "inputs.p", 1, 3)
    q = integer(inputs.get("q", 1), "inputs.q", 1, 3)
    horizon = integer(inputs.get("horizon", 5), "inputs.horizon", 1, 30)
    distribution = str(inputs.get("distribution") or "normal").lower()
    if distribution not in {"normal", "t"}:
        raise ComputeError("inputs.distribution must be normal or t")
    fitted = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=p,
        q=q,
        dist=distribution,
        rescale=False,
    ).fit(disp="off", show_warning=False)
    variance = np.asarray(fitted.forecast(horizon=horizon, reindex=False).variance.values[-1], dtype=float)
    if variance.shape != (horizon,) or np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ComputeError("GARCH forecast variance is invalid")
    return {
        "mode": "arch_garch_forecast",
        "observations": int(returns.size),
        "p": p,
        "q": q,
        "distribution": distribution,
        "conditional_variance_forecast": variance.tolist(),
        "conditional_volatility_forecast": np.sqrt(variance).tolist(),
        "parameters": {str(key): float(value) for key, value in fitted.params.items()},
        "log_likelihood": float(fitted.loglikelihood),
        "convergence_flag": int(fitted.convergence_flag),
        "engine": {"arch": package("arch")},
    }


def mapie_conformal_interval(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("MAPIE")
    from mapie.regression import CrossConformalRegressor
    from sklearn.linear_model import LinearRegression

    train_x = _bounded_matrix(inputs.get("train_x"), "inputs.train_x", rows=MAX_ROWS, columns=MAX_COLUMNS)
    train_y = vector(inputs.get("train_y"), "inputs.train_y", minimum=20, maximum=MAX_ROWS)
    predict_x = _bounded_matrix(inputs.get("predict_x"), "inputs.predict_x", rows=1_000, columns=MAX_COLUMNS)
    if train_x.shape[0] != train_y.size or train_x.shape[1] != predict_x.shape[1]:
        raise ComputeError("training rows and prediction feature dimensions must align")
    confidence = probability(inputs.get("confidence", 0.9), "inputs.confidence")
    if not 0.5 <= confidence < 1.0:
        raise ComputeError("inputs.confidence must be at least 0.5 and less than 1")
    cv = integer(inputs.get("cv", 5), "inputs.cv", 2, min(10, train_x.shape[0] // 4))
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    model = CrossConformalRegressor(
        estimator=LinearRegression(),
        confidence_level=confidence,
        cv=cv,
        random_state=seed,
    )
    model.fit_conformalize(train_x, train_y)
    point, intervals = model.predict_interval(predict_x)
    point = np.asarray(point, dtype=float).reshape(-1)
    intervals = np.asarray(intervals, dtype=float)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    if not (np.all(np.isfinite(point)) and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ComputeError("conformal prediction returned non-finite values")
    return {
        "mode": "mapie_conformal_interval",
        "confidence": confidence,
        "cv": cv,
        "predictions": point.tolist(),
        "lower_bounds": lower.tolist(),
        "upper_bounds": upper.tolist(),
        "mean_interval_width": float(np.mean(upper - lower)),
        "coverage_claim": "finite-sample marginal coverage under exchangeability assumptions",
        "engine": {"mapie": package("MAPIE")},
    }


def pydoe3_latin_hypercube(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pyDOE3")
    from pyDOE3 import lhs

    factors = integer(inputs.get("factors"), "inputs.factors", 1, 30)
    samples = integer(inputs.get("samples"), "inputs.samples", factors + 1, 1_000)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    criterion = str(inputs.get("criterion") or "maximin").lower()
    if criterion not in {"center", "maximin", "centermaximin", "correlation"}:
        raise ComputeError("inputs.criterion is not allowlisted")
    design = np.asarray(lhs(factors, samples=samples, criterion=criterion, random_state=seed), dtype=float)
    if design.shape != (samples, factors) or not np.all((design >= 0) & (design <= 1)):
        raise ComputeError("Latin hypercube design is invalid")
    return {
        "mode": "pydoe3_latin_hypercube",
        "factors": factors,
        "samples": samples,
        "criterion": criterion,
        "design": design.tolist(),
        "column_means": np.mean(design, axis=0).tolist(),
        "engine": {"pyDOE3": package("pyDOE3")},
    }


def lmfit_exponential_calibration(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("lmfit")
    import lmfit

    x = vector(inputs.get("x"), "inputs.x", minimum=5, maximum=MAX_ROWS)
    y = vector(inputs.get("y"), "inputs.y", minimum=5, maximum=MAX_ROWS)
    if x.size != y.size:
        raise ComputeError("x and y must have equal lengths")
    amplitude = finite(inputs.get("initial_amplitude", 1.0), "inputs.initial_amplitude")
    decay = _positive(inputs.get("initial_decay", 0.1), "inputs.initial_decay", 1e6)
    offset = finite(inputs.get("initial_offset", 0.0), "inputs.initial_offset")
    model = lmfit.Model(lambda x, amplitude, decay, offset: amplitude * np.exp(-decay * x) + offset)
    result = model.fit(y, x=x, amplitude=amplitude, decay=decay, offset=offset)
    if not result.success:
        raise ComputeError("lmfit calibration did not converge")
    parameters = {
        name: {
            "value": float(parameter.value),
            "standard_error": None if parameter.stderr is None else float(parameter.stderr),
        }
        for name, parameter in result.params.items()
    }
    residual = np.asarray(result.residual, dtype=float)
    return {
        "mode": "lmfit_exponential_calibration",
        "observations": int(x.size),
        "parameters": parameters,
        "chi_square": float(result.chisqr),
        "reduced_chi_square": float(result.redchi),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "engine": {"lmfit": package("lmfit")},
    }


def skgstat_variogram(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("scikit-gstat")
    from skgstat import Variogram

    coordinates = matrix(
        inputs.get("coordinates"),
        "inputs.coordinates",
        min_rows=5,
        max_rows=5_000,
        min_columns=2,
        max_columns=3,
    )
    values = vector(inputs.get("values"), "inputs.values", minimum=5, maximum=5_000)
    if coordinates.shape[0] != values.size:
        raise ComputeError("coordinates and values must have equal row counts")
    model_name = str(inputs.get("model") or "spherical").lower()
    if model_name not in {"spherical", "exponential", "gaussian", "matern", "stable", "cubic"}:
        raise ComputeError("inputs.model is not allowlisted")
    n_lags = integer(inputs.get("n_lags", 10), "inputs.n_lags", 3, min(50, values.size - 1))
    variogram = Variogram(coordinates, values, model=model_name, n_lags=n_lags, normalize=False)
    parameters = np.asarray(variogram.parameters, dtype=float)
    experimental = np.asarray(variogram.experimental, dtype=float)
    bins = np.asarray(variogram.bins, dtype=float)
    parameter_finite = bool(np.all(np.isfinite(parameters)))
    valid_lags = np.isfinite(experimental) & np.isfinite(bins)
    dropped_lags = int(np.size(valid_lags) - np.count_nonzero(valid_lags))
    experimental = experimental[valid_lags]
    bins = bins[valid_lags]
    return {
        "mode": "skgstat_variogram",
        "observations": int(values.size),
        "dimensions": int(coordinates.shape[1]),
        "model": model_name,
        "parameters": parameters.tolist() if parameter_finite else [],
        "fit_status": "fitted" if parameter_finite else "non-identifiable-from-input",
        "dropped_non_finite_lags": dropped_lags,
        "lag_bins": bins.tolist(),
        "experimental_semivariance": experimental.tolist(),
        "engine": {"scikit-gstat": package("scikit-gstat")},
    }


def rsome_robust_allocation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("rsome")
    from rsome import lpg_solver as solver
    from rsome import ro

    scenario_returns = matrix(
        inputs.get("scenario_returns"),
        "inputs.scenario_returns",
        min_rows=2,
        max_rows=500,
        min_columns=2,
        max_columns=50,
    )
    names_raw = inputs.get("asset_names")
    names = [str(value) for value in sequence(names_raw, "inputs.asset_names")] if names_raw is not None else [f"asset_{i}" for i in range(scenario_returns.shape[1])]
    if len(names) != scenario_returns.shape[1] or len(set(names)) != len(names) or any(not name or len(name) > 80 for name in names):
        raise ComputeError("asset_names must be unique and match scenario columns")
    model = ro.Model()
    weights = model.dvar(scenario_returns.shape[1])
    floor = model.dvar()
    model.max(floor)
    model.st(weights >= 0, weights.sum() == 1)
    for row in scenario_returns:
        model.st(row @ weights >= floor)
    model.solve(solver, display=False)
    solution = np.asarray(weights.get(), dtype=float).reshape(-1)
    if solution.size != scenario_returns.shape[1] or not np.all(np.isfinite(solution)):
        raise ComputeError("RSOME returned an invalid allocation")
    realized = scenario_returns @ solution
    return {
        "mode": "rsome_robust_allocation",
        "asset_names": names,
        "weights": solution.tolist(),
        "scenario_returns": realized.tolist(),
        "worst_case_return": float(np.min(realized)),
        "mean_return": float(np.mean(realized)),
        "scenario_count": int(scenario_returns.shape[0]),
        "engine": {"rsome": package("rsome"), "solver": "scipy-linprog"},
    }


def aequilibrae_shortest_path(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("aequilibrae")
    import pandas as pd
    from aequilibrae.paths import Graph

    raw_links = sequence(inputs.get("links"), "inputs.links")
    if not 1 <= len(raw_links) <= MAX_LINKS:
        raise ComputeError(f"inputs.links must contain 1 to {MAX_LINKS} directed links")
    rows = []
    nodes: set[int] = set()
    for index, raw_link in enumerate(raw_links, start=1):
        link = mapping(raw_link, f"inputs.links[{index - 1}]")
        a_node = integer(link.get("a_node"), "a_node", 1, 2_000_000_000)
        b_node = integer(link.get("b_node"), "b_node", 1, 2_000_000_000)
        if a_node == b_node:
            raise ComputeError("network links cannot be self-loops")
        cost = _positive(link.get("cost"), "cost", 1e12)
        rows.append({"link_id": index, "a_node": a_node, "b_node": b_node, "direction": 1, "cost": cost})
        nodes.update({a_node, b_node})
    origin = integer(inputs.get("origin"), "inputs.origin", 1, 2_000_000_000)
    destination = integer(inputs.get("destination"), "inputs.destination", 1, 2_000_000_000)
    if origin not in nodes or destination not in nodes or origin == destination:
        raise ComputeError("origin and destination must be distinct nodes present in the network")
    graph = Graph()
    graph.network = pd.DataFrame(rows)
    graph.prepare_graph(np.asarray(sorted(nodes), dtype=np.int64))
    graph.set_graph("cost")
    graph.set_skimming(["cost"])
    graph.set_blocked_centroid_flows(False)
    result = graph.compute_path(origin, destination)
    path_nodes = np.asarray(result.path_nodes, dtype=np.int64)
    path_links = np.asarray(result.path, dtype=np.int64)
    if path_nodes.size < 2 or int(path_nodes[0]) != origin or int(path_nodes[-1]) != destination:
        raise ComputeError("no valid directed path was found")
    link_cost = {row["link_id"]: row["cost"] for row in rows}
    total_cost = float(sum(link_cost[int(link_id)] for link_id in path_links))
    return {
        "mode": "aequilibrae_shortest_path",
        "origin": origin,
        "destination": destination,
        "path_nodes": path_nodes.tolist(),
        "path_links": path_links.tolist(),
        "total_cost": total_cost,
        "link_count": int(path_links.size),
        "network_link_count": len(rows),
        "engine": {"aequilibrae": package("aequilibrae")},
    }


def epydemix_sir_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("epydemix")
    from epydemix import EpiModel

    population = integer(inputs.get("population", 100_000), "inputs.population", 100, 10_000_000)
    infected = integer(inputs.get("initial_infected", 100), "inputs.initial_infected", 1, population - 1)
    beta = _positive(inputs.get("transmission_rate", 0.3), "inputs.transmission_rate", 10.0)
    gamma = _positive(inputs.get("recovery_rate", 0.1), "inputs.recovery_rate", 10.0)
    days = integer(inputs.get("days", 60), "inputs.days", 2, 365)
    simulations = integer(inputs.get("simulations", 20), "inputs.simulations", 1, 200)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    model = EpiModel(
        compartments=["S", "I", "R"],
        parameters={"beta": beta, "gamma": gamma},
        use_default_population=True,
        default_population_size=population,
    )
    model.add_transition(source="S", target="I", kind="mediated", params=("beta", "I"))
    model.add_transition(source="I", target="R", kind="spontaneous", params="gamma")
    initial_conditions = {
        "S": np.asarray([population - infected], dtype=int),
        "I": np.asarray([infected], dtype=int),
        "R": np.asarray([0], dtype=int),
    }
    start_date = date(2026, 1, 1)
    end_date = start_date + timedelta(days=days - 1)
    results = model.run_simulations(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        initial_conditions_dict=initial_conditions,
        Nsim=simulations,
        dt=1.0,
        rng=np.random.default_rng(seed),
    )
    infected_paths = []
    recovered_final = []
    for trajectory in results.trajectories:
        compartments = trajectory.compartments
        if isinstance(compartments, Mapping):
            if "I" not in compartments or "R" not in compartments:
                raise ComputeError("Epydemix returned incomplete compartment data")
            infected_array = np.asarray(compartments["I"], dtype=float)
            recovered_array = np.asarray(compartments["R"], dtype=float)
            if infected_array.ndim > 1:
                infected_array = infected_array.sum(axis=tuple(range(1, infected_array.ndim)))
            if recovered_array.ndim > 1:
                recovered_array = recovered_array.sum(axis=tuple(range(1, recovered_array.ndim)))
            infected_paths.append(infected_array.reshape(-1))
            recovered_final.append(float(recovered_array.reshape(-1)[-1]))
            continue
        array = np.asarray(compartments, dtype=float)
        index = dict(trajectory.compartment_idx)
        if array.ndim == 3:
            array = array.sum(axis=1)
        if array.ndim != 2 or "I" not in index or "R" not in index:
            raise ComputeError("Epydemix returned an unexpected trajectory shape")
        infected_paths.append(array[:, int(index["I"])])
        recovered_final.append(float(array[-1, int(index["R"])]))
    infected_stack = np.asarray(infected_paths, dtype=float)
    if infected_stack.ndim != 2 or not np.all(np.isfinite(infected_stack)):
        raise ComputeError("Epydemix returned non-finite trajectories")
    return {
        "mode": "epydemix_sir_simulation",
        "population": population,
        "days": days,
        "simulations": simulations,
        "transmission_rate": beta,
        "recovery_rate": gamma,
        "infected_median": np.quantile(infected_stack, 0.5, axis=0).tolist(),
        "infected_p05": np.quantile(infected_stack, 0.05, axis=0).tolist(),
        "infected_p95": np.quantile(infected_stack, 0.95, axis=0).tolist(),
        "peak_infected_median": float(np.max(np.quantile(infected_stack, 0.5, axis=0))),
        "final_recovered_mean": float(np.mean(recovered_final)),
        "engine": {"epydemix": package("epydemix")},
        "external_population_data_used": False,
    }


def pysd_stock_flow_scenario(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pysd")
    import pysd

    initial_stock = finite(inputs.get("initial_stock", 100.0), "inputs.initial_stock")
    constant_inflow = finite(inputs.get("constant_inflow", 10.0), "inputs.constant_inflow")
    decay_rate = finite(inputs.get("decay_rate", 0.05), "inputs.decay_rate")
    if decay_rate < 0 or decay_rate > 100:
        raise ComputeError("inputs.decay_rate must be between 0 and 100")
    final_time = _positive(inputs.get("final_time", 20.0), "inputs.final_time", 10_000)
    time_step = _positive(inputs.get("time_step", 0.25), "inputs.time_step", final_time)
    if final_time / time_step > MAX_OUTPUT_ROWS:
        raise ComputeError(f"system dynamics output may not exceed {MAX_OUTPUT_ROWS} rows")
    model_text = f"""Stock = INTEG (Inflow - Outflow, Initial Stock)
~ units
~ |
Inflow = Constant Inflow
~ units/time
~ |
Outflow = Decay Rate * Stock
~ units/time
~ |
Initial Stock = {initial_stock:.17g}
~ units
~ |
Constant Inflow = {constant_inflow:.17g}
~ units/time
~ |
Decay Rate = {decay_rate:.17g}
~ 1/time
~ |
FINAL TIME = {final_time:.17g}
~ time
~ |
INITIAL TIME = 0
~ time
~ |
SAVEPER = TIME STEP
~ time
~ |
TIME STEP = {time_step:.17g}
~ time
~ |
"""
    with tempfile.TemporaryDirectory(prefix="pysd-bounded-") as directory:
        model_path = Path(directory) / "bounded_stock_flow.mdl"
        model_path.write_text(model_text, encoding="utf-8")
        model = pysd.read_vensim(model_path)
        frame = model.run(return_columns=["Stock", "Inflow", "Outflow"])
    if frame.empty or len(frame) > MAX_OUTPUT_ROWS:
        raise ComputeError("PySD produced an empty or oversized result")
    values = frame.astype(float)
    if not np.all(np.isfinite(values.to_numpy())):
        raise ComputeError("PySD produced non-finite results")
    return {
        "mode": "pysd_stock_flow_scenario",
        "rows": int(len(values)),
        "time": [float(value) for value in values.index.to_numpy()],
        "stock": values["Stock"].tolist(),
        "inflow": values["Inflow"].tolist(),
        "outflow": values["Outflow"].tolist(),
        "final_stock": float(values["Stock"].iloc[-1]),
        "engine": {"pysd": package("pysd")},
        "model_surface": "fixed single-stock inflow-decay template",
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "openturns_reliability_probability": openturns_reliability_probability,
    "control_step_response": control_step_response,
    "pm4py_directly_follows": pm4py_directly_follows,
    "smt_rbf_surrogate": smt_rbf_surrogate,
    "econml_linear_dml": econml_linear_dml,
    "spreg_spatial_lag": spreg_spatial_lag,
    "arch_garch_forecast": arch_garch_forecast,
    "mapie_conformal_interval": mapie_conformal_interval,
    "pydoe3_latin_hypercube": pydoe3_latin_hypercube,
    "lmfit_exponential_calibration": lmfit_exponential_calibration,
    "skgstat_variogram": skgstat_variogram,
    "rsome_robust_allocation": rsome_robust_allocation,
    "aequilibrae_shortest_path": aequilibrae_shortest_path,
    "epydemix_sir_simulation": epydemix_sir_simulation,
    "pysd_stock_flow_scenario": pysd_stock_flow_scenario,
}
