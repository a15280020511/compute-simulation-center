#!/usr/bin/env python3
"""Bounded offline sector-model operations.

Every mode is fixed, repository-controlled and network-denied. Heavy dependencies
are imported lazily after the Capability Manager has selected one pinned bundle.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np

from compute_runner import ComputeError

MAX_ROWS = 5_000
MAX_COLUMNS = 30
MAX_GAME_ACTIONS = 20
MAX_SCENARIOS = 500
MAX_GRID_POINTS = 2_500
MODES = {
    "doubleml_plr",
    "quantecon_markov_chain",
    "nash_bimatrix_equilibria",
    "ema_robust_regret",
    "pypsa_linear_power_flow",
    "pandapower_ac_power_flow",
    "wntr_hydraulic_snapshot",
    "pywr_resource_allocation",
    "gstools_random_field",
    "pykrige_interpolation",
    "brightway_matrix_lca",
}
EXPECTED = {
    "doubleml": "0.11.3",
    "quantecon": "0.11.4",
    "nashpy": "0.0.43",
    "ema-workbench": "3.0.0",
    "pypsa": "1.2.4",
    "pandapower": "3.5.4",
    "wntr": "1.5.0",
    "pywr": "1.31.1",
    "gstools": "1.7.0",
    "pykrige": "1.7.3",
    "brightway25": "1.1.1",
}


def _require_version(distribution: str) -> str:
    try:
        observed = version(distribution)
    except Exception as exc:
        raise ComputeError(f"required sector-model package is not installed: {distribution}") from exc
    expected = EXPECTED[distribution]
    if observed != expected:
        raise ComputeError(f"{distribution} version must be exactly {expected}; observed {observed}")
    return observed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
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
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _vector(value: Any, name: str, *, minimum: int = 2, maximum: int = MAX_ROWS) -> np.ndarray:
    raw = _sequence(value, name)
    if not minimum <= len(raw) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} values")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ComputeError(f"{name} must be a finite one-dimensional array")
    return array


def _matrix(value: Any, name: str, *, max_rows: int = MAX_ROWS, max_columns: int = MAX_COLUMNS) -> np.ndarray:
    raw = _sequence(value, name)
    if not 1 <= len(raw) <= max_rows:
        raise ComputeError(f"{name} row count is invalid")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2 or not 1 <= array.shape[1] <= max_columns or not np.all(np.isfinite(array)):
        raise ComputeError(f"{name} must be a finite rectangular matrix with at most {max_columns} columns")
    return array


def _doubleml(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("doubleml")
    try:
        from doubleml import DoubleMLData, DoubleMLPLR
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise ComputeError("DoubleML dependencies are unavailable") from exc
    x = _matrix(inputs.get("x"), "inputs.x")
    y = _vector(inputs.get("y"), "inputs.y")
    d = _vector(inputs.get("treatment"), "inputs.treatment")
    if x.shape[0] != y.size or y.size != d.size:
        raise ComputeError("x, y and treatment must have equal row counts")
    if y.size < 8:
        raise ComputeError("doubleml_plr requires at least 8 observations")
    folds = _integer(inputs.get("folds", 2), "inputs.folds", 2, min(5, y.size // 2))
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    data = DoubleMLData.from_arrays(x, y, d)
    binary = bool(np.all((d == 0) | (d == 1)))
    ml_l = LinearRegression()
    ml_m = LogisticRegression(max_iter=2_000, random_state=seed) if binary else LinearRegression()
    model = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=folds, draw_sample_splitting=False)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    sample_splits = [(train, test) for train, test in splitter.split(x)]
    model.set_sample_splitting(sample_splits)
    model.fit()
    confidence = _finite(inputs.get("confidence", 0.95), "inputs.confidence", minimum=0.5, maximum=0.999)
    interval = model.confint(level=confidence)
    return {
        "mode": "doubleml_plr",
        "engine": "DoubleML",
        "engine_version": observed,
        "observations": int(y.size),
        "features": int(x.shape[1]),
        "folds": folds,
        "binary_treatment": binary,
        "effect": float(model.coef[0]),
        "standard_error": float(model.se[0]),
        "p_value": float(model.pval[0]),
        "confidence": confidence,
        "confidence_interval": [float(interval.iloc[0, 0]), float(interval.iloc[0, 1])],
        "claim_status": "identified_under_partial-linear-and-cross-fitting-assumptions",
    }


def _quantecon(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("quantecon")
    try:
        from quantecon import MarkovChain
    except ImportError as exc:
        raise ComputeError("QuantEcon is unavailable") from exc
    matrix = _matrix(inputs.get("transition_matrix"), "inputs.transition_matrix", max_rows=50, max_columns=50)
    if matrix.shape[0] != matrix.shape[1]:
        raise ComputeError("transition_matrix must be square")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ComputeError("transition_matrix rows must be non-negative and sum to one")
    initial = np.asarray(inputs.get("initial_distribution", [1.0 / matrix.shape[0]] * matrix.shape[0]), dtype=float)
    if initial.shape != (matrix.shape[0],) or np.any(initial < 0) or not np.isclose(initial.sum(), 1.0):
        raise ComputeError("initial_distribution must match state count and sum to one")
    steps = _integer(inputs.get("steps", 50), "inputs.steps", 1, 10_000)
    chain = MarkovChain(matrix)
    distribution = initial.copy()
    for _ in range(steps):
        distribution = distribution @ matrix
    stationary = np.asarray(chain.stationary_distributions, dtype=float)
    return {
        "mode": "quantecon_markov_chain",
        "engine": "QuantEcon",
        "engine_version": observed,
        "state_count": int(matrix.shape[0]),
        "steps": steps,
        "distribution_after_steps": distribution.tolist(),
        "stationary_distributions": stationary.tolist(),
        "stationary_distribution_count": int(stationary.shape[0]),
    }


def _nash(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("nashpy")
    try:
        import nashpy as nash
    except ImportError as exc:
        raise ComputeError("Nashpy is unavailable") from exc
    red = _matrix(inputs.get("row_payoffs"), "inputs.row_payoffs", max_rows=MAX_GAME_ACTIONS, max_columns=MAX_GAME_ACTIONS)
    blue = _matrix(inputs.get("column_payoffs"), "inputs.column_payoffs", max_rows=MAX_GAME_ACTIONS, max_columns=MAX_GAME_ACTIONS)
    if red.shape != blue.shape:
        raise ComputeError("payoff matrices must have identical shapes")
    game = nash.Game(red, blue)
    equilibria = []
    for row_strategy, column_strategy in game.support_enumeration():
        equilibria.append({
            "row_strategy": np.asarray(row_strategy, dtype=float).tolist(),
            "column_strategy": np.asarray(column_strategy, dtype=float).tolist(),
            "row_payoff": float(np.asarray(row_strategy) @ red @ np.asarray(column_strategy)),
            "column_payoff": float(np.asarray(row_strategy) @ blue @ np.asarray(column_strategy)),
        })
        if len(equilibria) >= 50:
            break
    return {
        "mode": "nash_bimatrix_equilibria",
        "engine": "Nashpy",
        "engine_version": observed,
        "shape": list(red.shape),
        "equilibrium_count": len(equilibria),
        "equilibria": equilibria,
        "truncated": len(equilibria) >= 50,
    }


def _ema_regret(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("ema-workbench")
    try:
        from ema_workbench import ScalarOutcome
    except ImportError as exc:
        raise ComputeError("EMA Workbench is unavailable") from exc
    outcomes = _matrix(inputs.get("outcomes"), "inputs.outcomes", max_rows=50, max_columns=MAX_SCENARIOS)
    names = list(inputs.get("alternative_names") or [f"alternative_{i}" for i in range(outcomes.shape[0])])
    if len(names) != outcomes.shape[0] or len(set(map(str, names))) != len(names):
        raise ComputeError("alternative_names must be unique and match outcome rows")
    maximize = bool(inputs.get("maximize", True))
    kind = ScalarOutcome.MAXIMIZE if maximize else ScalarOutcome.MINIMIZE
    outcome_definition = ScalarOutcome("decision_metric", kind=kind)
    benchmark = np.max(outcomes, axis=0) if maximize else np.min(outcomes, axis=0)
    regret = benchmark - outcomes if maximize else outcomes - benchmark
    worst = np.max(regret, axis=1)
    mean = np.mean(regret, axis=1)
    order = np.lexsort((mean, worst))
    rows = [{
        "alternative": str(names[index]),
        "worst_case_regret": float(worst[index]),
        "mean_regret": float(mean[index]),
        "rank": int(position + 1),
    } for position, index in enumerate(order)]
    return {
        "mode": "ema_robust_regret",
        "engine": "EMA Workbench compatible robust-decision screen",
        "engine_version": observed,
        "outcome_kind": "maximize" if outcome_definition.kind == ScalarOutcome.MAXIMIZE else "minimize",
        "scenario_count": int(outcomes.shape[1]),
        "alternative_count": int(outcomes.shape[0]),
        "recommended_alternative": rows[0]["alternative"],
        "ranking": rows,
    }


def _pypsa(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("pypsa")
    try:
        import pypsa
    except ImportError as exc:
        raise ComputeError("PyPSA is unavailable") from exc
    load = _finite(inputs.get("load_mw"), "inputs.load_mw", minimum=0.000001, maximum=100_000)
    voltage = _finite(inputs.get("voltage_kv", 110.0), "inputs.voltage_kv", minimum=0.1, maximum=1_200)
    reactance = _finite(inputs.get("line_reactance", 0.1), "inputs.line_reactance", minimum=0.000001, maximum=100)
    resistance = _finite(inputs.get("line_resistance", 0.01), "inputs.line_resistance", minimum=0, maximum=100)
    network = pypsa.Network()
    network.set_snapshots(["now"])
    network.add("Bus", "source", v_nom=voltage)
    network.add("Bus", "demand", v_nom=voltage)
    network.add("Line", "line", bus0="source", bus1="demand", x=reactance, r=resistance, s_nom=max(load * 2, 1))
    network.add("Generator", "generator", bus="source", p_set=load, control="Slack")
    network.add("Load", "load", bus="demand", p_set=load)
    network.lpf()
    p0 = float(network.lines_t.p0.loc["now", "line"])
    p1 = float(network.lines_t.p1.loc["now", "line"])
    angles = {str(name): float(value) for name, value in network.buses_t.v_ang.loc["now"].items()}
    return {
        "mode": "pypsa_linear_power_flow",
        "engine": "PyPSA",
        "engine_version": observed,
        "load_mw": load,
        "line_flow_from_mw": p0,
        "line_flow_to_mw": p1,
        "linear_loss_mw": p0 + p1,
        "bus_voltage_angles_radians": angles,
    }


def _pandapower(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("pandapower")
    try:
        import pandapower as pp
    except ImportError as exc:
        raise ComputeError("pandapower is unavailable") from exc
    voltage = _finite(inputs.get("voltage_kv", 20.0), "inputs.voltage_kv", minimum=0.1, maximum=1_200)
    load_mw = _finite(inputs.get("load_mw"), "inputs.load_mw", minimum=0.000001, maximum=100_000)
    load_mvar = _finite(inputs.get("load_mvar", 0.0), "inputs.load_mvar", minimum=-100_000, maximum=100_000)
    length = _finite(inputs.get("line_length_km", 1.0), "inputs.line_length_km", minimum=0.0001, maximum=10_000)
    net = pp.create_empty_network(sn_mva=max(1.0, load_mw * 2))
    source = pp.create_bus(net, vn_kv=voltage, name="source")
    demand = pp.create_bus(net, vn_kv=voltage, name="demand")
    pp.create_ext_grid(net, source, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, source, demand, length_km=length,
        r_ohm_per_km=_finite(inputs.get("r_ohm_per_km", 0.2), "inputs.r_ohm_per_km", minimum=0.000001, maximum=100),
        x_ohm_per_km=_finite(inputs.get("x_ohm_per_km", 0.1), "inputs.x_ohm_per_km", minimum=0.000001, maximum=100),
        c_nf_per_km=_finite(inputs.get("c_nf_per_km", 10.0), "inputs.c_nf_per_km", minimum=0, maximum=100_000),
        max_i_ka=_finite(inputs.get("max_i_ka", 1.0), "inputs.max_i_ka", minimum=0.0001, maximum=100),
        name="line",
    )
    pp.create_load(net, demand, p_mw=load_mw, q_mvar=load_mvar)
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="flat", max_iteration=30)
    return {
        "mode": "pandapower_ac_power_flow",
        "engine": "pandapower",
        "engine_version": observed,
        "converged": bool(net.converged),
        "source_bus": net.res_bus.loc[source].to_dict(),
        "demand_bus": net.res_bus.loc[demand].to_dict(),
        "line": net.res_line.loc[0].to_dict(),
    }


def _wntr(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("wntr")
    try:
        import wntr
    except ImportError as exc:
        raise ComputeError("WNTR is unavailable") from exc
    reservoir_head = _finite(inputs.get("reservoir_head_m", 100.0), "inputs.reservoir_head_m", minimum=1, maximum=10_000)
    elevation = _finite(inputs.get("junction_elevation_m", 10.0), "inputs.junction_elevation_m", minimum=-1_000, maximum=10_000)
    demand = _finite(inputs.get("demand_m3_s", 0.01), "inputs.demand_m3_s", minimum=0, maximum=1_000)
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("reservoir", base_head=reservoir_head)
    wn.add_junction("junction", base_demand=demand, elevation=elevation)
    wn.add_pipe(
        "pipe", "reservoir", "junction",
        length=_finite(inputs.get("pipe_length_m", 1_000.0), "inputs.pipe_length_m", minimum=0.1, maximum=1_000_000),
        diameter=_finite(inputs.get("pipe_diameter_m", 0.3), "inputs.pipe_diameter_m", minimum=0.001, maximum=20),
        roughness=_finite(inputs.get("pipe_roughness", 100.0), "inputs.pipe_roughness", minimum=1, maximum=1_000),
        minor_loss=0.0,
        initial_status="OPEN",
    )
    wn.options.time.duration = 0
    result = wntr.sim.WNTRSimulator(wn).run_sim(convergence_error=True)
    head = float(result.node["head"].iloc[0]["junction"])
    pressure = float(result.node["pressure"].iloc[0]["junction"])
    flow = float(result.link["flowrate"].iloc[0]["pipe"])
    return {
        "mode": "wntr_hydraulic_snapshot",
        "engine": "WNTR",
        "engine_version": observed,
        "junction_head_m": head,
        "junction_pressure_m": pressure,
        "pipe_flow_m3_s": flow,
        "requested_demand_m3_s": demand,
    }


def _pywr(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("pywr")
    try:
        from pywr.core import Model
        from pywr.nodes import Input, Output
    except ImportError as exc:
        raise ComputeError("Pywr is unavailable") from exc
    supply_capacity = _finite(inputs.get("supply_capacity"), "inputs.supply_capacity", minimum=0, maximum=1_000_000_000)
    demand_capacity = _finite(inputs.get("demand_capacity"), "inputs.demand_capacity", minimum=0, maximum=1_000_000_000)
    supply_cost = _finite(inputs.get("supply_cost", 0.0), "inputs.supply_cost", minimum=-1_000_000, maximum=1_000_000)
    demand_value = _finite(inputs.get("demand_value", 1.0), "inputs.demand_value", minimum=0, maximum=1_000_000)
    model = Model()
    supply = Input(model, name="supply", max_flow=supply_capacity, cost=supply_cost)
    demand = Output(model, name="demand", max_flow=demand_capacity, cost=-demand_value)
    supply.connect(demand)
    model.setup()
    model.step()
    allocated = float(np.asarray(demand.flow).reshape(-1)[0])
    model.finish()
    return {
        "mode": "pywr_resource_allocation",
        "engine": "Pywr",
        "engine_version": observed,
        "supply_capacity": supply_capacity,
        "demand_capacity": demand_capacity,
        "allocated_flow": allocated,
        "unmet_demand": max(0.0, demand_capacity - allocated),
        "objective_contribution": float((demand_value - supply_cost) * allocated),
    }


def _gstools(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("gstools")
    try:
        import gstools as gs
    except ImportError as exc:
        raise ComputeError("GSTools is unavailable") from exc
    nx = _integer(inputs.get("nx", 20), "inputs.nx", 2, 50)
    ny = _integer(inputs.get("ny", 20), "inputs.ny", 2, 50)
    if nx * ny > MAX_GRID_POINTS:
        raise ComputeError(f"grid cannot exceed {MAX_GRID_POINTS} cells")
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    variance = _finite(inputs.get("variance", 1.0), "inputs.variance", minimum=0.000001, maximum=1_000_000)
    length_scale = _finite(inputs.get("length_scale", 10.0), "inputs.length_scale", minimum=0.000001, maximum=1_000_000)
    mean = _finite(inputs.get("mean", 0.0), "inputs.mean", minimum=-1_000_000, maximum=1_000_000)
    model = gs.Gaussian(dim=2, var=variance, len_scale=length_scale)
    field = np.asarray(gs.SRF(model, mean=mean, seed=seed).structured([np.arange(nx), np.arange(ny)]), dtype=float)
    return {
        "mode": "gstools_random_field",
        "engine": "GSTools",
        "engine_version": observed,
        "seed": seed,
        "shape": list(field.shape),
        "minimum": float(np.min(field)),
        "maximum": float(np.max(field)),
        "mean": float(np.mean(field)),
        "standard_deviation": float(np.std(field)),
        "field": field.tolist(),
    }


def _pykrige(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("pykrige")
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise ComputeError("PyKrige is unavailable") from exc
    x = _vector(inputs.get("x"), "inputs.x", minimum=3, maximum=2_000)
    y = _vector(inputs.get("y"), "inputs.y", minimum=3, maximum=2_000)
    z = _vector(inputs.get("z"), "inputs.z", minimum=3, maximum=2_000)
    px = _vector(inputs.get("predict_x"), "inputs.predict_x", minimum=1, maximum=2_000)
    py = _vector(inputs.get("predict_y"), "inputs.predict_y", minimum=1, maximum=2_000)
    if not (x.size == y.size == z.size) or px.size != py.size:
        raise ComputeError("observation coordinates/values and prediction coordinates must align")
    variogram = str(inputs.get("variogram_model") or "linear")
    if variogram not in {"linear", "power", "gaussian", "spherical", "exponential"}:
        raise ComputeError("unsupported variogram_model")
    model = OrdinaryKriging(x, y, z, variogram_model=variogram, verbose=False, enable_plotting=False)
    estimates, variances = model.execute("points", px, py)
    return {
        "mode": "pykrige_interpolation",
        "engine": "PyKrige",
        "engine_version": observed,
        "observation_count": int(x.size),
        "prediction_count": int(px.size),
        "variogram_model": variogram,
        "predictions": np.asarray(estimates, dtype=float).tolist(),
        "kriging_variances": np.asarray(variances, dtype=float).tolist(),
    }


def _brightway(inputs: Mapping[str, Any]) -> dict[str, Any]:
    observed = _require_version("brightway25")
    try:
        import bw2calc  # noqa: F401
        import bw2data  # noqa: F401
    except ImportError as exc:
        raise ComputeError("Brightway 2.5 backend is unavailable") from exc
    technology = _matrix(inputs.get("technology_matrix"), "inputs.technology_matrix", max_rows=100, max_columns=100)
    biosphere = _matrix(inputs.get("biosphere_matrix"), "inputs.biosphere_matrix", max_rows=100, max_columns=100)
    demand = _vector(inputs.get("demand"), "inputs.demand", minimum=1, maximum=100)
    characterization = _vector(inputs.get("characterization_factors"), "inputs.characterization_factors", minimum=1, maximum=100)
    if technology.shape[0] != technology.shape[1] or technology.shape[0] != demand.size:
        raise ComputeError("technology_matrix must be square and match demand")
    if biosphere.shape[1] != demand.size or biosphere.shape[0] != characterization.size:
        raise ComputeError("biosphere_matrix dimensions must match activities and characterization factors")
    condition = float(np.linalg.cond(technology))
    if not math.isfinite(condition) or condition > 1e12:
        raise ComputeError("technology_matrix is singular or ill-conditioned")
    supply = np.linalg.solve(technology, demand)
    inventory = biosphere @ supply
    score = float(characterization @ inventory)
    contributions = characterization[:, None] * biosphere * supply[None, :]
    return {
        "mode": "brightway_matrix_lca",
        "engine": "Brightway 2.5 compatible matrix LCA",
        "engine_version": observed,
        "activity_count": int(demand.size),
        "biosphere_flow_count": int(characterization.size),
        "technology_condition_number": condition,
        "supply_array": supply.tolist(),
        "inventory": inventory.tolist(),
        "impact_score": score,
        "activity_contributions": np.sum(contributions, axis=0).tolist(),
        "restriction": "bounded matrix calculation only; external Brightway databases are not fetched or imported at runtime",
    }


def sector_model_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode not in MODES:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(MODES))}")
    handlers = {
        "doubleml_plr": _doubleml,
        "quantecon_markov_chain": _quantecon,
        "nash_bimatrix_equilibria": _nash,
        "ema_robust_regret": _ema_regret,
        "pypsa_linear_power_flow": _pypsa,
        "pandapower_ac_power_flow": _pandapower,
        "wntr_hydraulic_snapshot": _wntr,
        "pywr_resource_allocation": _pywr,
        "gstools_random_field": _gstools,
        "pykrige_interpolation": _pykrige,
        "brightway_matrix_lca": _brightway,
    }
    result = handlers[mode](_mapping(inputs, "inputs"))
    result["network_used"] = False
    result["arbitrary_code_used"] = False
    result["maturity"] = "controlled-preview"
    return result


OPERATIONS = {"sector_model_analysis": sector_model_analysis}
