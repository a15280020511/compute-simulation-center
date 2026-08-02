#!/usr/bin/env python3
"""Allowlisted institutional infrastructure, climate, epidemiology and finance operations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from institutional_common import engine, integer, jsonable, matrix, vector
from think_tank_common import finite, mapping, sequence


def energy_system_dispatch(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pypsa", "highspy")
    import pandas as pd
    import pypsa

    load = vector(inputs.get("load"), "inputs.load", minimum=1, maximum=1_000)
    raw_generators = sequence(inputs.get("generators"), "inputs.generators")
    if not 1 <= len(raw_generators) <= 100:
        raise ComputeError("generators must contain 1 to 100 entries")
    network = pypsa.Network()
    snapshots = pd.RangeIndex(load.size, name="snapshot")
    network.set_snapshots(snapshots)
    network.add("Bus", "bus")
    network.add("Load", "load", bus="bus", p_set=load)
    names = []
    for index, raw in enumerate(raw_generators):
        row = mapping(raw, f"inputs.generators[{index}]")
        name = str(row.get("name") or f"g{index}")
        if name in names:
            raise ComputeError("generator names must be unique")
        capacity = finite(row.get("capacity"), f"inputs.generators[{index}].capacity")
        marginal_cost = finite(row.get("marginal_cost"), f"inputs.generators[{index}].marginal_cost")
        if capacity <= 0 or marginal_cost < 0:
            raise ComputeError("generator capacity must be positive and marginal_cost non-negative")
        availability_raw = row.get("availability")
        if availability_raw is None:
            availability = np.ones(load.size)
        else:
            availability = vector(
                availability_raw,
                f"inputs.generators[{index}].availability",
                minimum=load.size,
                maximum=load.size,
            )
            if np.any((availability < 0) | (availability > 1)):
                raise ComputeError("generator availability must be between 0 and 1")
        network.add(
            "Generator",
            name,
            bus="bus",
            p_nom=capacity,
            marginal_cost=marginal_cost,
            p_max_pu=availability,
        )
        names.append(name)
    status, condition = network.optimize(solver_name="highs", log_to_console=False)
    if str(status).lower() not in {"ok", "warning"}:
        raise ComputeError(f"energy dispatch optimization failed: {status}/{condition}")
    dispatch = {
        name: [float(value) for value in network.generators_t.p[name].to_numpy()]
        for name in names
    }
    marginal_price = [float(value) for value in network.buses_t.marginal_price["bus"].to_numpy()]
    return {
        "mode": "energy_system_dispatch",
        "status": str(status),
        "condition": str(condition),
        "snapshots": int(load.size),
        "objective": float(network.objective),
        "dispatch": dispatch,
        "marginal_price": marginal_price,
        "unserved_energy": float(max(0.0, np.sum(load) - sum(np.sum(values) for values in dispatch.values()))),
        "engine": engine("pypsa", "highspy"),
    }


def power_flow_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pandapower")
    import pandapower as pp

    load_mw = finite(inputs.get("load_mw"), "inputs.load_mw")
    load_mvar = finite(inputs.get("load_mvar", 0.0), "inputs.load_mvar")
    line_length_km = finite(inputs.get("line_length_km", 1.0), "inputs.line_length_km")
    voltage_kv = finite(inputs.get("voltage_kv", 20.0), "inputs.voltage_kv")
    if load_mw < 0 or line_length_km <= 0 or voltage_kv <= 0:
        raise ComputeError("load_mw must be non-negative and line/voltage values positive")
    network = pp.create_empty_network()
    source = pp.create_bus(network, vn_kv=voltage_kv, name="source")
    demand = pp.create_bus(network, vn_kv=voltage_kv, name="demand")
    pp.create_ext_grid(network, bus=source, vm_pu=1.0)
    pp.create_line_from_parameters(
        network,
        from_bus=source,
        to_bus=demand,
        length_km=line_length_km,
        r_ohm_per_km=finite(inputs.get("r_ohm_per_km", 0.2), "inputs.r_ohm_per_km"),
        x_ohm_per_km=finite(inputs.get("x_ohm_per_km", 0.1), "inputs.x_ohm_per_km"),
        c_nf_per_km=finite(inputs.get("c_nf_per_km", 10.0), "inputs.c_nf_per_km"),
        max_i_ka=finite(inputs.get("max_i_ka", 0.4), "inputs.max_i_ka"),
    )
    pp.create_load(network, bus=demand, p_mw=load_mw, q_mvar=load_mvar)
    pp.runpp(network, algorithm="nr", calculate_voltage_angles=True, init="auto")
    return {
        "mode": "power_flow_analysis",
        "converged": bool(network.converged),
        "bus_voltage_pu": [float(value) for value in network.res_bus.vm_pu.to_numpy()],
        "bus_angle_degree": [float(value) for value in network.res_bus.va_degree.to_numpy()],
        "line_loading_percent": [float(value) for value in network.res_line.loading_percent.to_numpy()],
        "line_losses_mw": [float(value) for value in network.res_line.pl_mw.to_numpy()],
        "engine": engine("pandapower"),
    }


def water_network_resilience(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("wntr")
    import wntr

    demand = finite(inputs.get("demand_m3s", 0.01), "inputs.demand_m3s")
    duration_hours = integer(inputs.get("duration_hours", 24), "inputs.duration_hours", 1, 168)
    if demand < 0:
        raise ComputeError("demand_m3s must be non-negative")
    network = wntr.network.WaterNetworkModel()
    network.options.time.duration = duration_hours * 3600
    network.options.time.hydraulic_timestep = 3600
    network.options.time.report_timestep = 3600
    network.options.hydraulic.demand_model = "PDD"
    network.options.hydraulic.required_pressure = finite(inputs.get("required_pressure", 20.0), "inputs.required_pressure")
    network.options.hydraulic.minimum_pressure = finite(inputs.get("minimum_pressure", 0.0), "inputs.minimum_pressure")
    network.add_reservoir("R1", base_head=finite(inputs.get("reservoir_head", 80.0), "inputs.reservoir_head"))
    network.add_junction("J1", base_demand=demand, elevation=finite(inputs.get("elevation_1", 10.0), "inputs.elevation_1"))
    network.add_junction("J2", base_demand=demand, elevation=finite(inputs.get("elevation_2", 12.0), "inputs.elevation_2"))
    diameter = finite(inputs.get("diameter_m", 0.3), "inputs.diameter_m")
    roughness = finite(inputs.get("roughness", 100.0), "inputs.roughness")
    network.add_pipe("P1", "R1", "J1", length=1000.0, diameter=diameter, roughness=roughness, minor_loss=0.0)
    network.add_pipe("P2", "J1", "J2", length=1000.0, diameter=diameter, roughness=roughness, minor_loss=0.0)
    results = wntr.sim.WNTRSimulator(network).run_sim()
    pressure = results.node["pressure"][["J1", "J2"]]
    demand_result = results.node["demand"][["J1", "J2"]]
    required = float(network.options.hydraulic.required_pressure)
    service = np.clip(pressure.to_numpy(dtype=float) / required, 0.0, 1.0)
    return {
        "mode": "water_network_resilience",
        "duration_hours": duration_hours,
        "minimum_pressure": {column: float(pressure[column].min()) for column in pressure.columns},
        "mean_pressure": {column: float(pressure[column].mean()) for column in pressure.columns},
        "delivered_demand_m3s": {column: float(demand_result[column].mean()) for column in demand_result.columns},
        "mean_pressure_service_ratio": float(np.mean(service)),
        "engine": engine("wntr"),
    }


def climate_threshold_index(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("xclim", "xarray")
    import pandas as pd
    import xarray as xr
    from xclim import indices

    temperatures = vector(inputs.get("daily_max_temperature_c"), "inputs.daily_max_temperature_c", minimum=30, maximum=100_000)
    threshold = finite(inputs.get("threshold_c", 35.0), "inputs.threshold_c")
    dates = pd.date_range(str(inputs.get("start_date") or "2000-01-01"), periods=temperatures.size, freq="D")
    array = xr.DataArray(temperatures, dims=("time",), coords={"time": dates}, attrs={"units": "degC"})
    counts = indices.tx_days_above(array, thresh=f"{threshold} degC", freq="YS")
    maximum = array.resample(time="YS").max()
    return {
        "mode": "climate_threshold_index",
        "threshold_c": threshold,
        "annual_days_above_threshold": {
            str(pd.Timestamp(time).year): int(round(float(value)))
            for time, value in zip(counts.time.values, counts.values, strict=True)
        },
        "annual_maximum_temperature_c": {
            str(pd.Timestamp(time).year): float(value)
            for time, value in zip(maximum.time.values, maximum.values, strict=True)
        },
        "engine": engine("xclim", "xarray"),
    }


def epidemic_scenario(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("starsim")
    import starsim as ss

    population = integer(inputs.get("population", 1_000), "inputs.population", 100, 100_000)
    duration = integer(inputs.get("duration_days", 120), "inputs.duration_days", 10, 1_000)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    beta = finite(inputs.get("beta", 0.05), "inputs.beta")
    infectious_duration = finite(inputs.get("infectious_duration", 7.0), "inputs.infectious_duration")
    initial_prevalence = finite(inputs.get("initial_prevalence", 0.01), "inputs.initial_prevalence")
    if beta <= 0 or infectious_duration <= 0 or not 0 < initial_prevalence < 1:
        raise ComputeError("epidemic parameters are invalid")
    disease = ss.SIR(beta=beta, dur_inf=infectious_duration, init_prev=initial_prevalence)
    simulation = ss.Sim(
        n_agents=population,
        diseases=disease,
        networks="random",
        rand_seed=seed,
        dur=duration,
        verbose=False,
    )
    simulation.run(verbose=0)
    infected = np.asarray(simulation.results.sir.n_infected, dtype=float)
    cumulative = np.asarray(simulation.results.sir.cum_infections, dtype=float)
    return {
        "mode": "epidemic_scenario",
        "population": population,
        "duration_days": duration,
        "peak_infected": float(np.max(infected)),
        "peak_day": int(np.argmax(infected)),
        "cumulative_infections": float(cumulative[-1]),
        "attack_rate": float(cumulative[-1] / population),
        "engine": engine("starsim"),
    }


def european_option_pricing(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("QuantLib")
    import QuantLib as ql

    option_type = str(inputs.get("option_type") or "call").lower()
    if option_type not in {"call", "put"}:
        raise ComputeError("option_type must be call or put")
    spot = finite(inputs.get("spot"), "inputs.spot")
    strike = finite(inputs.get("strike"), "inputs.strike")
    volatility = finite(inputs.get("volatility"), "inputs.volatility")
    risk_free_rate = finite(inputs.get("risk_free_rate"), "inputs.risk_free_rate")
    dividend_yield = finite(inputs.get("dividend_yield", 0.0), "inputs.dividend_yield")
    maturity_days = integer(inputs.get("maturity_days"), "inputs.maturity_days", 1, 3650)
    if spot <= 0 or strike <= 0 or volatility <= 0:
        raise ComputeError("spot, strike and volatility must be positive")
    calendar = ql.NullCalendar()
    today = ql.Date(1, 1, 2020)
    ql.Settings.instance().evaluationDate = today
    maturity = today + maturity_days
    day_count = ql.Actual365Fixed()
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == "call" else ql.Option.Put, strike)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    risk_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, day_count))
    vol_surface = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, volatility, day_count))
    process = ql.BlackScholesMertonProcess(spot_handle, dividend_curve, risk_curve, vol_surface)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {
        "mode": "european_option_pricing",
        "option_type": option_type,
        "net_present_value": float(option.NPV()),
        "delta": float(option.delta()),
        "gamma": float(option.gamma()),
        "vega": float(option.vega()),
        "theta": float(option.theta()),
        "rho": float(option.rho()),
        "engine": engine("QuantLib"),
    }


def copula_dependence_fit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pyvinecopulib")
    import pyvinecopulib as pv

    data = matrix(inputs.get("uniform_data"), "inputs.uniform_data", min_rows=30, max_rows=50_000, min_columns=2, max_columns=2)
    if np.any((data <= 0) | (data >= 1)):
        raise ComputeError("uniform_data values must be strictly between 0 and 1")
    model = pv.Bicop.from_data(data)
    parameters = np.asarray(model.parameters, dtype=float)
    return {
        "mode": "copula_dependence_fit",
        "family": str(model.family),
        "rotation": int(model.rotation),
        "parameters": jsonable(parameters),
        "kendalls_tau": float(model.tau),
        "log_likelihood": float(model.loglik(data)),
        "aic": float(model.aic(data)),
        "bic": float(model.bic(data)),
        "engine": engine("pyvinecopulib"),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "energy_system_dispatch": energy_system_dispatch,
    "power_flow_analysis": power_flow_analysis,
    "water_network_resilience": water_network_resilience,
    "climate_threshold_index": climate_threshold_index,
    "epidemic_scenario": epidemic_scenario,
    "european_option_pricing": european_option_pricing,
    "copula_dependence_fit": copula_dependence_fit,
}
