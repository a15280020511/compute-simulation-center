#!/usr/bin/env python3
"""Allowlisted institutional decision, spatial and urban-analysis operations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from institutional_common import engine, integer, jsonable, matrix, strings, vector
from think_tank_common import finite, mapping, sequence


def deep_uncertainty_exploration(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("ema-workbench")
    from ema_workbench import Model, RealParameter, ScalarOutcome, perform_experiments

    raw_parameters = sequence(inputs.get("parameters"), "inputs.parameters")
    if not 1 <= len(raw_parameters) <= 20:
        raise ComputeError("parameters must contain 1 to 20 entries")
    definitions: list[tuple[str, float, float, float]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_parameters):
        row = mapping(raw, f"inputs.parameters[{index}]")
        name = str(row.get("name") or "")
        if not name or name in seen or name[0].isdigit() or not name.replace("_", "").isalnum():
            raise ComputeError("parameter names must be unique safe identifiers")
        lower = finite(row.get("minimum"), f"inputs.parameters[{index}].minimum")
        upper = finite(row.get("maximum"), f"inputs.parameters[{index}].maximum")
        coefficient = finite(row.get("coefficient"), f"inputs.parameters[{index}].coefficient")
        if lower >= upper:
            raise ComputeError("parameter minimum must be below maximum")
        definitions.append((name, lower, upper, coefficient))
        seen.add(name)
    intercept = finite(inputs.get("intercept", 0.0), "inputs.intercept")
    scenarios = integer(inputs.get("scenarios", 100), "inputs.scenarios", 10, 5_000)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)

    def fixed_model(**kwargs: float) -> dict[str, float]:
        value = intercept + sum(coefficient * float(kwargs[name]) for name, _, _, coefficient in definitions)
        interaction = 0.0
        if len(definitions) >= 2:
            first, second = definitions[0][0], definitions[1][0]
            interaction = 0.01 * float(kwargs[first]) * float(kwargs[second])
        return {"outcome": value + interaction}

    model = Model("bounded_linear_scenario_model", function=fixed_model)
    model.uncertainties = [RealParameter(name, lower, upper) for name, lower, upper, _ in definitions]
    model.outcomes = [ScalarOutcome("outcome")]
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        experiments, outcomes = perform_experiments(model, scenarios=scenarios, reporting_interval=max(1, scenarios + 1))
    finally:
        np.random.set_state(state)
    values = np.asarray(outcomes["outcome"], dtype=float)
    top_index = int(np.argmax(values))
    bottom_index = int(np.argmin(values))
    parameter_rows = experiments[[name for name, _, _, _ in definitions]]
    return {
        "mode": "deep_uncertainty_exploration",
        "scenario_count": int(values.size),
        "outcome_summary": {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        },
        "best_scenario": {
            "parameters": jsonable(parameter_rows.iloc[top_index].to_dict()),
            "outcome": float(values[top_index]),
        },
        "worst_scenario": {
            "parameters": jsonable(parameter_rows.iloc[bottom_index].to_dict()),
            "outcome": float(values[bottom_index]),
        },
        "engine": engine("ema-workbench"),
    }


def comprehensive_mcda(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pymcdm")
    from pymcdm.methods import PROMETHEE_II, TOPSIS, VIKOR

    decision_matrix = matrix(inputs.get("decision_matrix"), "inputs.decision_matrix", min_rows=2, max_rows=1_000, min_columns=2, max_columns=50)
    weights = vector(inputs.get("weights"), "inputs.weights", minimum=decision_matrix.shape[1], maximum=decision_matrix.shape[1])
    criteria_types = np.asarray(
        [int(item) for item in sequence(inputs.get("criteria_types"), "inputs.criteria_types")],
        dtype=int,
    )
    if criteria_types.size != decision_matrix.shape[1] or any(item not in {-1, 1} for item in criteria_types):
        raise ComputeError("criteria_types must contain one -1 or 1 per criterion")
    if np.any(weights < 0) or float(np.sum(weights)) <= 0:
        raise ComputeError("weights must be non-negative with positive sum")
    weights = weights / np.sum(weights)
    method_name = str(inputs.get("method") or "topsis").lower()
    methods = {
        "topsis": (TOPSIS(), True),
        "vikor": (VIKOR(), False),
        "promethee_ii": (PROMETHEE_II("usual"), True),
    }
    if method_name not in methods:
        raise ComputeError("method must be topsis, vikor, or promethee_ii")
    method, higher_is_better = methods[method_name]
    preferences = np.asarray(method(decision_matrix, weights, criteria_types), dtype=float).reshape(-1)
    order = np.argsort(preferences)
    if higher_is_better:
        order = order[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return {
        "mode": "comprehensive_mcda",
        "method": method_name,
        "higher_is_better": higher_is_better,
        "preferences": jsonable(preferences),
        "ranks": jsonable(ranks),
        "best_alternative": int(order[0]),
        "engine": engine("pymcdm"),
    }


def matrix_game_equilibrium(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("nashpy")
    import nashpy as nash

    row_payoffs = matrix(inputs.get("row_payoffs"), "inputs.row_payoffs", min_rows=2, max_rows=20, min_columns=2, max_columns=20)
    column_payoffs = matrix(
        inputs.get("column_payoffs"),
        "inputs.column_payoffs",
        min_rows=row_payoffs.shape[0],
        max_rows=row_payoffs.shape[0],
        min_columns=row_payoffs.shape[1],
        max_columns=row_payoffs.shape[1],
    )
    if row_payoffs.shape != column_payoffs.shape:
        raise ComputeError("payoff matrices must have identical shapes")
    game = nash.Game(row_payoffs, column_payoffs)
    equilibria = list(game.support_enumeration())
    if not equilibria:
        equilibria = list(game.vertex_enumeration())
    rows = []
    for row_strategy, column_strategy in equilibria[:100]:
        row_strategy = np.asarray(row_strategy, dtype=float)
        column_strategy = np.asarray(column_strategy, dtype=float)
        payoffs = game[row_strategy, column_strategy]
        rows.append(
            {
                "row_strategy": jsonable(row_strategy),
                "column_strategy": jsonable(column_strategy),
                "row_payoff": float(payoffs[0]),
                "column_payoff": float(payoffs[1]),
            }
        )
    return {
        "mode": "matrix_game_equilibrium",
        "shape": list(row_payoffs.shape),
        "equilibrium_count": len(equilibria),
        "equilibria": rows,
        "engine": engine("nashpy"),
    }


def geospatial_join(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("geopandas")
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, box

    raw_points = sequence(inputs.get("points"), "inputs.points")
    raw_regions = sequence(inputs.get("regions"), "inputs.regions")
    if not 1 <= len(raw_points) <= 20_000 or not 1 <= len(raw_regions) <= 5_000:
        raise ComputeError("points or regions exceed limit")
    points = []
    for index, raw in enumerate(raw_points):
        row = mapping(raw, f"inputs.points[{index}]")
        points.append(
            {
                "point_id": str(row.get("id", index)),
                "geometry": Point(
                    finite(row.get("x"), f"inputs.points[{index}].x"),
                    finite(row.get("y"), f"inputs.points[{index}].y"),
                ),
            }
        )
    regions = []
    for index, raw in enumerate(raw_regions):
        row = mapping(raw, f"inputs.regions[{index}]")
        minx = finite(row.get("minx"), f"inputs.regions[{index}].minx")
        miny = finite(row.get("miny"), f"inputs.regions[{index}].miny")
        maxx = finite(row.get("maxx"), f"inputs.regions[{index}].maxx")
        maxy = finite(row.get("maxy"), f"inputs.regions[{index}].maxy")
        if minx >= maxx or miny >= maxy:
            raise ComputeError("region bounds are invalid")
        regions.append({"region_id": str(row.get("id", index)), "geometry": box(minx, miny, maxx, maxy)})
    crs = str(inputs.get("crs") or "EPSG:4326")
    point_frame = gpd.GeoDataFrame(pd.DataFrame(points), geometry="geometry", crs=crs)
    region_frame = gpd.GeoDataFrame(pd.DataFrame(regions), geometry="geometry", crs=crs)
    joined = gpd.sjoin(point_frame, region_frame, how="left", predicate="within")
    matches = [
        {
            "point_id": str(row.point_id),
            "region_id": None if row.region_id != row.region_id else str(row.region_id),
        }
        for row in joined.itertuples()
    ]
    return {
        "mode": "geospatial_join",
        "point_count": len(points),
        "region_count": len(regions),
        "matched_count": sum(item["region_id"] is not None for item in matches),
        "matches": matches,
        "engine": engine("geopandas"),
    }


def geographically_weighted_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("mgwr")
    from mgwr.gwr import GWR

    coordinates = matrix(inputs.get("coordinates"), "inputs.coordinates", min_rows=20, max_rows=5_000, min_columns=2, max_columns=2)
    x = matrix(inputs.get("x"), "inputs.x", min_rows=coordinates.shape[0], max_rows=coordinates.shape[0], max_columns=20)
    y = vector(inputs.get("y"), "inputs.y", minimum=coordinates.shape[0], maximum=coordinates.shape[0])
    if x.shape[0] != coordinates.shape[0]:
        raise ComputeError("coordinates, x and y must align")
    bandwidth = finite(inputs.get("bandwidth"), "inputs.bandwidth")
    if bandwidth <= 0:
        raise ComputeError("bandwidth must be positive")
    fit = GWR(
        [tuple(row) for row in coordinates],
        y.reshape(-1, 1),
        x,
        bandwidth,
        fixed=True,
        kernel="bisquare",
        constant=True,
        spherical=False,
    ).fit()
    return {
        "mode": "geographically_weighted_regression",
        "observations": int(x.shape[0]),
        "bandwidth": bandwidth,
        "aic": float(fit.aic),
        "aicc": float(fit.aicc),
        "r2": float(fit.R2),
        "local_parameters": jsonable(np.asarray(fit.params, dtype=float)),
        "engine": engine("mgwr"),
    }


def urban_morphology_metrics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("momepy", "geopandas")
    import geopandas as gpd
    import momepy
    from shapely.geometry import box

    raw_buildings = sequence(inputs.get("buildings"), "inputs.buildings")
    if not 1 <= len(raw_buildings) <= 20_000:
        raise ComputeError("buildings must contain 1 to 20000 rectangles")
    ids = []
    geometries = []
    for index, raw in enumerate(raw_buildings):
        row = mapping(raw, f"inputs.buildings[{index}]")
        minx = finite(row.get("minx"), f"inputs.buildings[{index}].minx")
        miny = finite(row.get("miny"), f"inputs.buildings[{index}].miny")
        maxx = finite(row.get("maxx"), f"inputs.buildings[{index}].maxx")
        maxy = finite(row.get("maxy"), f"inputs.buildings[{index}].maxy")
        if minx >= maxx or miny >= maxy:
            raise ComputeError("building bounds are invalid")
        ids.append(str(row.get("id", index)))
        geometries.append(box(minx, miny, maxx, maxy))
    frame = gpd.GeoDataFrame({"building_id": ids}, geometry=geometries, crs=str(inputs.get("crs") or "EPSG:3857"))
    compactness = np.asarray(momepy.circular_compactness(frame.geometry), dtype=float)
    areas = np.asarray(frame.geometry.area, dtype=float)
    perimeters = np.asarray(frame.geometry.length, dtype=float)
    return {
        "mode": "urban_morphology_metrics",
        "building_count": len(ids),
        "metrics": [
            {
                "building_id": ids[index],
                "area": float(areas[index]),
                "perimeter": float(perimeters[index]),
                "circular_compactness": float(compactness[index]),
            }
            for index in range(len(ids))
        ],
        "engine": engine("momepy", "geopandas"),
    }


def spatial_lag_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("spreg", "libpysal")
    from libpysal.weights import W
    from spreg import ML_Lag

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20, max_rows=5_000, max_columns=20)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    raw_neighbors = mapping(inputs.get("neighbors"), "inputs.neighbors")
    neighbors: dict[int, list[int]] = {}
    for key, raw in raw_neighbors.items():
        index = int(key)
        values = [int(item) for item in sequence(raw, f"inputs.neighbors[{key}]")]
        if index < 0 or index >= x.shape[0] or any(item < 0 or item >= x.shape[0] or item == index for item in values):
            raise ComputeError("neighbor indices are invalid")
        neighbors[index] = values
    if set(neighbors) != set(range(x.shape[0])):
        raise ComputeError("neighbors must define every observation")
    weights = W(neighbors, silence_warnings=True)
    weights.transform = "r"
    fit = ML_Lag(y.reshape(-1, 1), x, w=weights, method="full", name_y="y")
    betas = np.asarray(fit.betas, dtype=float).reshape(-1)
    return {
        "mode": "spatial_lag_regression",
        "observations": int(x.shape[0]),
        "rho": float(fit.rho),
        "coefficients": jsonable(betas),
        "log_likelihood": float(fit.logll),
        "aic": float(fit.aic),
        "engine": engine("spreg", "libpysal"),
    }


def facility_location(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("spopt", "pulp")
    import pulp
    from spopt.locate import PMedian

    costs = matrix(inputs.get("cost_matrix"), "inputs.cost_matrix", min_rows=2, max_rows=1_000, min_columns=2, max_columns=500)
    weights = vector(inputs.get("demand_weights"), "inputs.demand_weights", minimum=costs.shape[0], maximum=costs.shape[0])
    p_facilities = integer(inputs.get("p_facilities"), "inputs.p_facilities", 1, costs.shape[1])
    if np.any(costs < 0) or np.any(weights < 0):
        raise ComputeError("costs and demand weights must be non-negative")
    model = PMedian.from_cost_matrix(costs, weights, p_facilities=p_facilities)
    solved = model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
    selected = [index for index, clients in enumerate(solved.fac2cli) if clients]
    assignments = [
        {"facility": int(facility), "clients": [int(client) for client in clients]}
        for facility, clients in enumerate(solved.fac2cli)
        if clients
    ]
    objective = float(pulp.value(solved.problem.objective))
    return {
        "mode": "facility_location",
        "selected_facilities": selected,
        "assignments": assignments,
        "objective": objective,
        "engine": engine("spopt", "pulp"),
    }


def trajectory_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("movingpandas", "geopandas")
    import geopandas as gpd
    import movingpandas as mpd
    import pandas as pd
    from shapely.geometry import Point

    raw_points = sequence(inputs.get("points"), "inputs.points")
    if not 2 <= len(raw_points) <= 50_000:
        raise ComputeError("points must contain 2 to 50000 rows")
    rows = []
    for index, raw in enumerate(raw_points):
        row = mapping(raw, f"inputs.points[{index}]")
        timestamp = pd.Timestamp(str(row.get("timestamp") or ""))
        if timestamp is pd.NaT:
            raise ComputeError("trajectory timestamp is invalid")
        rows.append(
            {
                "trajectory_id": str(row.get("trajectory_id") or "default"),
                "timestamp": timestamp,
                "geometry": Point(
                    finite(row.get("x"), f"inputs.points[{index}].x"),
                    finite(row.get("y"), f"inputs.points[{index}].y"),
                ),
            }
        )
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=str(inputs.get("crs") or "EPSG:4326"))
    results = []
    for trajectory_id, group in frame.groupby("trajectory_id", sort=True):
        group = group.sort_values("timestamp").set_index("timestamp")
        if len(group) < 2:
            continue
        trajectory = mpd.Trajectory(group, trajectory_id)
        results.append(
            {
                "trajectory_id": str(trajectory_id),
                "point_count": int(len(group)),
                "length": float(trajectory.get_length()),
                "duration_seconds": float(trajectory.get_duration().total_seconds()),
                "direction_degrees": float(trajectory.get_direction()),
            }
        )
    if not results:
        raise ComputeError("every trajectory has fewer than two points")
    return {
        "mode": "trajectory_analysis",
        "trajectory_count": len(results),
        "trajectories": results,
        "engine": engine("movingpandas", "geopandas"),
    }


def spatial_segregation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("segregation", "geopandas")
    import geopandas as gpd
    from segregation.singlegroup import Dissim
    from shapely.geometry import box

    raw_areas = sequence(inputs.get("areas"), "inputs.areas")
    if not 2 <= len(raw_areas) <= 10_000:
        raise ComputeError("areas must contain 2 to 10000 rows")
    rows = []
    geometries = []
    for index, raw in enumerate(raw_areas):
        row = mapping(raw, f"inputs.areas[{index}]")
        group_population = integer(row.get("group_population"), f"inputs.areas[{index}].group_population", 0, 10**9)
        total_population = integer(row.get("total_population"), f"inputs.areas[{index}].total_population", 1, 10**9)
        if group_population > total_population:
            raise ComputeError("group_population cannot exceed total_population")
        rows.append({"area_id": str(row.get("id", index)), "group_population": group_population, "total_population": total_population})
        minx = finite(row.get("minx"), f"inputs.areas[{index}].minx")
        miny = finite(row.get("miny"), f"inputs.areas[{index}].miny")
        maxx = finite(row.get("maxx"), f"inputs.areas[{index}].maxx")
        maxy = finite(row.get("maxy"), f"inputs.areas[{index}].maxy")
        geometries.append(box(minx, miny, maxx, maxy))
    frame = gpd.GeoDataFrame(rows, geometry=geometries, crs=str(inputs.get("crs") or "EPSG:3857"))
    result = Dissim(frame, group_pop_var="group_population", total_pop_var="total_population")
    return {
        "mode": "spatial_segregation",
        "dissimilarity_index": float(result.statistic),
        "area_count": len(rows),
        "group_population": int(frame["group_population"].sum()),
        "total_population": int(frame["total_population"].sum()),
        "engine": engine("segregation", "geopandas"),
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "deep_uncertainty_exploration": deep_uncertainty_exploration,
    "comprehensive_mcda": comprehensive_mcda,
    "matrix_game_equilibrium": matrix_game_equilibrium,
    "geospatial_join": geospatial_join,
    "geographically_weighted_regression": geographically_weighted_regression,
    "urban_morphology_metrics": urban_morphology_metrics,
    "spatial_lag_regression": spatial_lag_regression,
    "facility_location": facility_location,
    "trajectory_analysis": trajectory_analysis,
    "spatial_segregation": spatial_segregation,
}
