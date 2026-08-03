#!/usr/bin/env python3
"""Governed causal-discovery, drift-detection and robust-finance modes.

All handlers consume bounded structured data, execute offline, make zero model
calls, reject ticket-supplied code and never place trades. Causal-discovery
outputs are hypotheses under declared assumptions, not proof of causation.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from scipy.stats import kurtosis, norm, skew

from compute_runner import ComputeError

MAX_ROWS = 5_000
MAX_COLUMNS = 30
MAX_BOOTSTRAPS = 100
MAX_LAG = 20
MAX_TRIALS = 10_000


def _package(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ComputeError(f"required capability pack is not installed: {distribution}") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ComputeError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ComputeError(f"{name} must be <= {maximum}")
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


def _matrix(value: Any, name: str, minimum_rows: int = 20, maximum_rows: int = MAX_ROWS) -> np.ndarray:
    rows = _sequence(value, name)
    if not minimum_rows <= len(rows) <= maximum_rows:
        raise ComputeError(f"{name} must contain {minimum_rows} to {maximum_rows} rows")
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or not 2 <= array.shape[1] <= MAX_COLUMNS or not np.all(np.isfinite(array)):
        raise ComputeError(f"{name} must be a finite rectangular matrix with 2 to {MAX_COLUMNS} columns")
    return array


def _names(value: Any, count: int) -> list[str]:
    if value is None:
        return [f"x{index}" for index in range(count)]
    rows = _sequence(value, "inputs.variable_names")
    names = [str(item).strip() for item in rows]
    if len(names) != count or any(not item or len(item) > 80 for item in names) or len(set(names)) != count:
        raise ComputeError("variable_names must be unique non-empty names matching the column count")
    return names


def _graph_edges(graph: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left in range(graph.shape[0]):
        for right in range(left + 1, graph.shape[0]):
            lr = int(graph[left, right])
            rl = int(graph[right, left])
            if lr == 0 and rl == 0:
                continue
            if lr == -1 and rl == 1:
                relation = "directed"
                source, target = names[left], names[right]
            elif lr == 1 and rl == -1:
                relation = "directed"
                source, target = names[right], names[left]
            elif lr == -1 and rl == -1:
                relation = "undirected"
                source, target = names[left], names[right]
            elif lr == 1 and rl == 1:
                relation = "bidirected"
                source, target = names[left], names[right]
            else:
                relation = "partially_oriented"
                source, target = names[left], names[right]
            edges.append({
                "source": source,
                "target": target,
                "relation": relation,
                "raw_endpoints": [lr, rl],
            })
    return edges


def causal_pc_discovery(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("causal-learn")
    from causallearn.search.ConstraintBased.PC import pc

    data = _matrix(inputs.get("data"), "inputs.data", minimum_rows=30)
    names = _names(inputs.get("variable_names"), data.shape[1])
    alpha = _finite(inputs.get("alpha", 0.05), "inputs.alpha", 0.0001, 0.5)
    result = pc(data, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False, verbose=False)
    graph = np.asarray(result.G.graph, dtype=int)
    return {
        "mode": "causal_pc_discovery",
        "engine": {"causal-learn": version, "algorithm": "stable-PC", "independence_test": "fisher-z"},
        "observations": int(data.shape[0]),
        "variables": names,
        "alpha": alpha,
        "edges": _graph_edges(graph, names),
        "adjacency_endpoints": graph.tolist(),
        "claim_status": "causal-structure-hypothesis-under-causal-sufficiency-and-faithfulness",
        "network_used": False,
    }


def causal_fci_discovery(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("causal-learn")
    from causallearn.search.ConstraintBased.FCI import fci

    data = _matrix(inputs.get("data"), "inputs.data", minimum_rows=30)
    names = _names(inputs.get("variable_names"), data.shape[1])
    alpha = _finite(inputs.get("alpha", 0.05), "inputs.alpha", 0.0001, 0.5)
    depth = _integer(inputs.get("depth", -1), "inputs.depth", -1, 10)
    graph_object, _ = fci(
        data,
        independence_test_method="fisherz",
        alpha=alpha,
        depth=depth,
        max_path_length=_integer(inputs.get("max_path_length", -1), "inputs.max_path_length", -1, 20),
        verbose=False,
        show_progress=False,
    )
    graph = np.asarray(graph_object.graph, dtype=int)
    return {
        "mode": "causal_fci_discovery",
        "engine": {"causal-learn": version, "algorithm": "FCI", "independence_test": "fisher-z"},
        "observations": int(data.shape[0]),
        "variables": names,
        "alpha": alpha,
        "edges": _graph_edges(graph, names),
        "adjacency_endpoints": graph.tolist(),
        "claim_status": "partial-ancestral-graph-hypothesis-allowing-latent-confounding",
        "network_used": False,
    }


def causal_graph_stability_bootstrap(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("causal-learn")
    from causallearn.search.ConstraintBased.PC import pc

    data = _matrix(inputs.get("data"), "inputs.data", minimum_rows=40, maximum_rows=2_000)
    names = _names(inputs.get("variable_names"), data.shape[1])
    alpha = _finite(inputs.get("alpha", 0.05), "inputs.alpha", 0.0001, 0.5)
    bootstraps = _integer(inputs.get("bootstraps", 30), "inputs.bootstraps", 5, MAX_BOOTSTRAPS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    threshold = _finite(inputs.get("stability_threshold", 0.7), "inputs.stability_threshold", 0.5, 1.0)
    rng = np.random.default_rng(seed)
    counts = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    for _ in range(bootstraps):
        sample = data[rng.integers(0, data.shape[0], size=data.shape[0])]
        graph = np.asarray(pc(sample, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False, verbose=False).G.graph)
        skeleton = (graph != 0) | (graph.T != 0)
        counts += skeleton.astype(int)
    rows = []
    for left in range(data.shape[1]):
        for right in range(left + 1, data.shape[1]):
            frequency = float(counts[left, right] / bootstraps)
            if frequency >= threshold:
                rows.append({"left": names[left], "right": names[right], "selection_frequency": frequency})
    rows.sort(key=lambda row: (-row["selection_frequency"], row["left"], row["right"]))
    return {
        "mode": "causal_graph_stability_bootstrap",
        "engine": {"causal-learn": version, "algorithm": "stable-PC-bootstrap"},
        "bootstraps": bootstraps,
        "seed": seed,
        "stability_threshold": threshold,
        "stable_skeleton_edges": rows,
        "claim_status": "resampling-stability-screen-not-causal-proof",
        "network_used": False,
    }


def tigramite_pcmci_discovery(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("tigramite")
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    data = _matrix(inputs.get("data"), "inputs.data", minimum_rows=50)
    names = _names(inputs.get("variable_names"), data.shape[1])
    tau_max = _integer(inputs.get("tau_max", 3), "inputs.tau_max", 1, MAX_LAG)
    alpha = _finite(inputs.get("alpha", 0.05), "inputs.alpha", 0.0001, 0.5)
    pcmci = PCMCI(dataframe=pp.DataFrame(data, var_names=names), cond_ind_test=ParCorr(), verbosity=0)
    result = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    p_matrix = np.asarray(result["p_matrix"], dtype=float)
    val_matrix = np.asarray(result["val_matrix"], dtype=float)
    links = []
    for source in range(data.shape[1]):
        for target in range(data.shape[1]):
            for lag in range(1, tau_max + 1):
                p_value = float(p_matrix[source, target, lag])
                if p_value <= alpha:
                    links.append({
                        "source": names[source],
                        "target": names[target],
                        "lag": -lag,
                        "strength": float(val_matrix[source, target, lag]),
                        "p_value": p_value,
                    })
    links.sort(key=lambda row: (row["p_value"], -abs(row["strength"]), row["source"], row["target"]))
    return {
        "mode": "tigramite_pcmci_discovery",
        "engine": {"tigramite": version, "algorithm": "PCMCI", "independence_test": "partial-correlation"},
        "observations": int(data.shape[0]),
        "variables": names,
        "tau_max": tau_max,
        "alpha": alpha,
        "significant_lagged_links": links,
        "claim_status": "time-series-causal-graph-hypothesis-under-stationarity-and-no-unmodeled-confounding",
        "network_used": False,
    }


def evidently_data_drift(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("evidently")
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from scipy.stats import ks_2samp

    reference = _matrix(inputs.get("reference"), "inputs.reference", minimum_rows=20)
    current = _matrix(inputs.get("current"), "inputs.current", minimum_rows=20)
    if reference.shape[1] != current.shape[1]:
        raise ComputeError("reference and current must have equal column counts")
    names = _names(inputs.get("variable_names"), reference.shape[1])
    drift_share = _finite(inputs.get("drift_share", 0.5), "inputs.drift_share", 0.05, 1.0)
    report = Report([DataDriftPreset(drift_share=drift_share)])
    evaluation = report.run(
        current_data=pd.DataFrame(current, columns=names),
        reference_data=pd.DataFrame(reference, columns=names),
    )
    payload = json.loads(evaluation.json())
    alpha = _finite(inputs.get("screening_alpha", 0.05), "inputs.screening_alpha", 0.0001, 0.5)
    columns = []
    for index, name in enumerate(names):
        statistic, p_value = ks_2samp(reference[:, index], current[:, index], method="auto")
        columns.append({"column": name, "ks_statistic": float(statistic), "p_value": float(p_value), "screened_drift": bool(p_value <= alpha)})
    drifted = sum(row["screened_drift"] for row in columns)
    return {
        "mode": "evidently_data_drift",
        "engine": {"evidently": version, "preset": "DataDriftPreset"},
        "reference_rows": int(reference.shape[0]),
        "current_rows": int(current.shape[0]),
        "drifted_columns_screen": drifted,
        "drift_share_screen": float(drifted / len(columns)),
        "columns": columns,
        "evidently_report_generated": bool(payload),
        "evidently_report_metric_count": len(payload.get("metrics", [])) if isinstance(payload, Mapping) else None,
        "network_used": False,
    }


def river_adwin_drift(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("river")
    from river import drift

    values = np.asarray(_sequence(inputs.get("values"), "inputs.values"), dtype=float)
    if values.ndim != 1 or not 20 <= values.size <= 50_000 or not np.all(np.isfinite(values)):
        raise ComputeError("values must contain 20 to 50000 finite observations")
    delta = _finite(inputs.get("delta", 0.002), "inputs.delta", 1e-8, 0.5)
    detector = drift.ADWIN(delta=delta)
    detected = []
    for index, value in enumerate(values):
        detector.update(float(value))
        if detector.drift_detected:
            detected.append(index)
    return {
        "mode": "river_adwin_drift",
        "engine": {"river": version, "detector": "ADWIN"},
        "observations": int(values.size),
        "delta": delta,
        "drift_indices": detected,
        "drift_count": len(detected),
        "network_used": False,
    }


def skfolio_walk_forward_portfolio(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("skfolio")
    import pandas as pd
    from skfolio.optimization import MeanRisk

    returns = _matrix(inputs.get("returns"), "inputs.returns", minimum_rows=80)
    train_size = _integer(inputs.get("train_size", 60), "inputs.train_size", 40, returns.shape[0] - 10)
    test_size = _integer(inputs.get("test_size", 20), "inputs.test_size", 1, min(250, returns.shape[0] - train_size))
    max_weight = _finite(inputs.get("max_weight", 1.0), "inputs.max_weight", 1.0 / returns.shape[1], 1.0)
    cost_bps = _finite(inputs.get("transaction_cost_bps", 10.0), "inputs.transaction_cost_bps", 0.0, 1_000.0)
    windows = []
    net_returns: list[float] = []
    previous = np.zeros(returns.shape[1], dtype=float)
    start = train_size
    while start < returns.shape[0]:
        end = min(start + test_size, returns.shape[0])
        train = pd.DataFrame(returns[start - train_size : start])
        model = MeanRisk(min_weights=0.0, max_weights=max_weight)
        model.fit(train)
        weights = np.asarray(model.weights_, dtype=float)
        if weights.shape != (returns.shape[1],) or not np.all(np.isfinite(weights)):
            raise ComputeError("skfolio returned invalid weights")
        turnover = float(np.sum(np.abs(weights - previous)))
        fold = np.asarray(returns[start:end] @ weights, dtype=float)
        cost = turnover * cost_bps / 10_000.0
        if fold.size:
            fold[0] -= cost
        net_returns.extend(fold.tolist())
        windows.append({"train_start": start - train_size, "train_end": start, "test_end": end, "weights": weights.tolist(), "turnover": turnover, "cost_return": cost})
        previous = weights
        start = end
    net = np.asarray(net_returns, dtype=float)
    if net.size == 0:
        raise ComputeError("walk-forward configuration produced no out-of-sample observations")
    wealth = np.cumprod(1.0 + net)
    drawdown = 1.0 - wealth / np.maximum.accumulate(wealth)
    return {
        "mode": "skfolio_walk_forward_portfolio",
        "engine": {"skfolio": version, "optimizer": "MeanRisk-minimum-variance"},
        "fold_count": len(windows),
        "out_of_sample_observations": int(net.size),
        "cumulative_return_net": float(wealth[-1] - 1.0),
        "mean_return": float(np.mean(net)),
        "volatility": float(np.std(net, ddof=1)) if net.size > 1 else 0.0,
        "maximum_drawdown": float(np.max(drawdown)),
        "total_turnover": float(sum(row["turnover"] for row in windows)),
        "windows": windows,
        "lookahead_bias_control": "each test fold uses only the immediately preceding training window",
        "brokerage_execution": False,
        "network_used": False,
    }


def black_litterman_allocation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    version = _package("PyPortfolioOpt")
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.efficient_frontier import EfficientFrontier

    covariance = np.asarray(_sequence(inputs.get("covariance"), "inputs.covariance"), dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not 2 <= covariance.shape[0] <= 50 or not np.all(np.isfinite(covariance)):
        raise ComputeError("covariance must be a finite square matrix of size 2 to 50")
    assets = covariance.shape[0]
    prior = np.asarray(_sequence(inputs.get("prior_returns"), "inputs.prior_returns"), dtype=float)
    views = np.asarray(_sequence(inputs.get("view_returns"), "inputs.view_returns"), dtype=float)
    pick = np.asarray(_sequence(inputs.get("pick_matrix"), "inputs.pick_matrix"), dtype=float)
    if prior.shape != (assets,) or views.ndim != 1 or pick.shape != (views.size, assets):
        raise ComputeError("prior_returns, view_returns and pick_matrix dimensions are inconsistent")
    tau = _finite(inputs.get("tau", 0.05), "inputs.tau", 0.0001, 1.0)
    model = BlackLittermanModel(covariance, pi=prior, P=pick, Q=views, tau=tau)
    posterior_returns = model.bl_returns()
    posterior_covariance = model.bl_cov()
    frontier = EfficientFrontier(posterior_returns, posterior_covariance, weight_bounds=(0.0, 1.0))
    frontier.min_volatility()
    weights = frontier.clean_weights()
    return {
        "mode": "black_litterman_allocation",
        "engine": {"PyPortfolioOpt": version, "model": "Black-Litterman"},
        "posterior_returns": {str(key): float(value) for key, value in posterior_returns.items()},
        "weights": {str(key): float(value) for key, value in weights.items()},
        "tau": tau,
        "view_count": int(views.size),
        "brokerage_execution": False,
        "network_used": False,
    }


def deflated_sharpe_gate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    returns = np.asarray(_sequence(inputs.get("returns"), "inputs.returns"), dtype=float)
    if returns.ndim != 1 or not 30 <= returns.size <= 100_000 or not np.all(np.isfinite(returns)):
        raise ComputeError("returns must contain 30 to 100000 finite observations")
    trials = _integer(inputs.get("strategy_trials", 1), "inputs.strategy_trials", 1, MAX_TRIALS)
    observed = _finite(inputs.get("observed_sharpe", float(np.mean(returns) / max(np.std(returns, ddof=1), 1e-12))), "inputs.observed_sharpe")
    trial_std = _finite(inputs.get("trial_sharpe_standard_deviation", 0.0), "inputs.trial_sharpe_standard_deviation", 0.0)
    gamma = 0.5772156649015329
    if trials == 1 or trial_std == 0:
        expected_maximum = 0.0
    else:
        expected_maximum = trial_std * ((1 - gamma) * norm.ppf(1 - 1 / trials) + gamma * norm.ppf(1 - 1 / (trials * math.e)))
    sample_skew = float(skew(returns, bias=False))
    sample_kurtosis = float(kurtosis(returns, fisher=False, bias=False))
    denominator = math.sqrt(max(1e-12, 1 - sample_skew * observed + ((sample_kurtosis - 1) / 4) * observed**2))
    statistic = (observed - expected_maximum) * math.sqrt(returns.size - 1) / denominator
    probability = float(norm.cdf(statistic))
    threshold = _finite(inputs.get("minimum_probability", 0.95), "inputs.minimum_probability", 0.5, 0.999999)
    return {
        "mode": "deflated_sharpe_gate",
        "observations": int(returns.size),
        "observed_sharpe": observed,
        "expected_maximum_sharpe_under_multiple_testing": expected_maximum,
        "deflated_sharpe_probability": probability,
        "minimum_probability": threshold,
        "passed": bool(probability >= threshold),
        "strategy_trials": trials,
        "sample_skewness": sample_skew,
        "sample_kurtosis": sample_kurtosis,
        "interpretation": "multiple-testing and non-normality screen; not evidence of future profitability",
        "brokerage_execution": False,
    }


def transaction_cost_capacity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    returns = _matrix(inputs.get("asset_returns"), "inputs.asset_returns", minimum_rows=20)
    weights = np.asarray(_sequence(inputs.get("weights"), "inputs.weights"), dtype=float)
    if weights.shape != returns.shape or not np.all(np.isfinite(weights)):
        raise ComputeError("weights must be a finite matrix with the same shape as asset_returns")
    if np.any(np.abs(weights).sum(axis=1) > 5.0 + 1e-9):
        raise ComputeError("gross leverage may not exceed 5")
    capital = _finite(inputs.get("capital"), "inputs.capital", 1.0)
    average_daily_volume = np.asarray(_sequence(inputs.get("average_daily_volume"), "inputs.average_daily_volume"), dtype=float)
    if average_daily_volume.shape != (returns.shape[1],) or np.any(average_daily_volume <= 0) or not np.all(np.isfinite(average_daily_volume)):
        raise ComputeError("average_daily_volume must contain one positive value per asset")
    commission_bps = _finite(inputs.get("commission_bps", 2.0), "inputs.commission_bps", 0.0, 1_000.0)
    spread_bps = _finite(inputs.get("spread_bps", 5.0), "inputs.spread_bps", 0.0, 2_000.0)
    impact_coefficient = _finite(inputs.get("impact_coefficient", 0.1), "inputs.impact_coefficient", 0.0, 10.0)
    previous = np.zeros(returns.shape[1], dtype=float)
    gross_rows = []
    net_rows = []
    costs = []
    participation_rows = []
    for index in range(returns.shape[0]):
        current = weights[index]
        traded_notional = np.abs(current - previous) * capital
        participation = traded_notional / average_daily_volume
        linear_cost = traded_notional.sum() * (commission_bps + 0.5 * spread_bps) / 10_000.0
        impact_cost = float(np.sum(traded_notional * impact_coefficient * np.sqrt(np.maximum(participation, 0.0))))
        total_cost_return = (linear_cost + impact_cost) / capital
        gross_return = float(current @ returns[index])
        gross_rows.append(gross_return)
        net_rows.append(gross_return - total_cost_return)
        costs.append(total_cost_return)
        participation_rows.append(float(np.max(participation)))
        previous = current
    gross = np.asarray(gross_rows)
    net = np.asarray(net_rows)
    gross_wealth = np.cumprod(1.0 + gross)
    net_wealth = np.cumprod(1.0 + net)
    max_participation = _finite(inputs.get("maximum_participation", 0.1), "inputs.maximum_participation", 0.0001, 1.0)
    return {
        "mode": "transaction_cost_capacity",
        "observations": int(returns.shape[0]),
        "gross_cumulative_return": float(gross_wealth[-1] - 1.0),
        "net_cumulative_return": float(net_wealth[-1] - 1.0),
        "total_cost_return": float(np.sum(costs)),
        "maximum_observed_participation": float(max(participation_rows)),
        "maximum_participation_limit": max_participation,
        "participation_breach_count": int(sum(value > max_participation for value in participation_rows)),
        "capacity_status": "BREACH" if any(value > max_participation for value in participation_rows) else "PASS",
        "impact_model": "square-root participation impact plus commission and half-spread",
        "brokerage_execution": False,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "causal_pc_discovery": causal_pc_discovery,
    "causal_fci_discovery": causal_fci_discovery,
    "causal_graph_stability_bootstrap": causal_graph_stability_bootstrap,
    "tigramite_pcmci_discovery": tigramite_pcmci_discovery,
    "evidently_data_drift": evidently_data_drift,
    "river_adwin_drift": river_adwin_drift,
    "skfolio_walk_forward_portfolio": skfolio_walk_forward_portfolio,
    "black_litterman_allocation": black_litterman_allocation,
    "deflated_sharpe_gate": deflated_sharpe_gate,
    "transaction_cost_capacity": transaction_cost_capacity,
}

from personal_finance_operations import HANDLERS as PERSONAL_FINANCE_HANDLERS

if set(HANDLERS) & set(PERSONAL_FINANCE_HANDLERS):
    raise RuntimeError("personal-finance modes conflict with institutional expansion modes")
HANDLERS.update(PERSONAL_FINANCE_HANDLERS)
