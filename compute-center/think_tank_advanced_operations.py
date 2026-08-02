#!/usr/bin/env python3
"""Hierarchical Bayesian, raster and spatial-statistical modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import MAX_GRID_CELLS, integer, mapping, matrix, package, sequence, vector


def hierarchical_bayesian_mean(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pymc")
    package("arviz")
    import arviz as az
    import pymc as pm

    values = vector(inputs.get("values"), "inputs.values", minimum=20, maximum=5_000)
    groups_raw = [str(v) for v in sequence(inputs.get("groups"), "inputs.groups")]
    if len(groups_raw) != values.size:
        raise ComputeError("groups and values must have equal length")
    names = sorted(set(groups_raw))
    if not 2 <= len(names) <= 50:
        raise ComputeError("hierarchical model requires 2 to 50 groups")
    index = np.asarray([names.index(v) for v in groups_raw], dtype=int)
    draws = integer(inputs.get("draws", 500), "inputs.draws", 100, 2_000)
    tune = integer(inputs.get("tune", 500), "inputs.tune", 100, 2_000)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    scale = max(float(np.std(values)), 1.0)
    with pm.Model(coords={"group": names}) as model:
        mu_global = pm.Normal("mu_global", mu=float(np.mean(values)), sigma=scale * 2)
        tau = pm.HalfNormal("tau", sigma=scale)
        group_mean = pm.Normal("group_mean", mu=mu_global, sigma=tau, dims="group")
        sigma = pm.HalfNormal("sigma", sigma=scale)
        pm.Normal("observed", mu=group_mean[index], sigma=sigma, observed=values)
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            random_seed=seed,
            progressbar=False,
            target_accept=0.9,
            compute_convergence_checks=True,
        )
    summary = az.summary(trace, var_names=["mu_global", "tau", "sigma", "group_mean"], round_to=None)
    diagnostics = {
        "max_r_hat": float(np.nanmax(summary["r_hat"].to_numpy())),
        "min_ess_bulk": float(np.nanmin(summary["ess_bulk"].to_numpy())),
        "divergences": int(np.sum(trace.sample_stats["diverging"].to_numpy())),
    }
    return {
        "mode": "hierarchical_bayesian_mean",
        "global_mean_posterior": float(trace.posterior["mu_global"].mean().item()),
        "group_means": {
            name: float(trace.posterior["group_mean"].sel(group=name).mean().item()) for name in names
        },
        "diagnostics": diagnostics,
        "engine": {"pymc": package("pymc"), "arviz": package("arviz")},
        "formal_use_allowed": diagnostics["max_r_hat"] <= 1.01 and diagnostics["divergences"] == 0,
    }


def raster_zonal_statistics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("xarray")
    package("rioxarray")
    package("rasterio")
    import xarray as xr
    import rioxarray  # noqa: F401

    values = matrix(inputs.get("values"), "inputs.values", max_rows=1_000, max_columns=1_000)
    zones = matrix(inputs.get("zones"), "inputs.zones", max_rows=1_000, max_columns=1_000)
    if values.shape != zones.shape or values.size > MAX_GRID_CELLS:
        raise ComputeError("values and zones grids must align and stay within cell limit")
    crs = str(inputs.get("crs") or "EPSG:4326")
    array = xr.DataArray(values, dims=("y", "x")).rio.write_crs(crs)
    rows = []
    for zone in sorted(set(int(v) for v in np.unique(zones) if int(v) >= 0)):
        subset = array.values[zones == zone]
        rows.append(
            {
                "zone": zone,
                "count": int(subset.size),
                "mean": float(np.mean(subset)),
                "minimum": float(np.min(subset)),
                "maximum": float(np.max(subset)),
                "sum": float(np.sum(subset)),
            }
        )
    return {
        "mode": "raster_zonal_statistics",
        "crs": str(array.rio.crs),
        "shape": list(values.shape),
        "zones": rows,
        "engines": {
            "xarray": package("xarray"),
            "rioxarray": package("rioxarray"),
            "rasterio": package("rasterio"),
        },
    }


def raster_change_detection(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("xarray")
    before = matrix(inputs.get("before"), "inputs.before", max_rows=1_000, max_columns=1_000)
    after = matrix(inputs.get("after"), "inputs.after", max_rows=1_000, max_columns=1_000)
    if before.shape != after.shape or before.size > MAX_GRID_CELLS:
        raise ComputeError("before and after grids must align and stay within cell limit")
    from think_tank_common import finite

    threshold = finite(inputs.get("threshold", 0.0), "inputs.threshold")
    delta = after - before
    changed = np.abs(delta) > threshold
    return {
        "mode": "raster_change_detection",
        "shape": list(delta.shape),
        "mean_change": float(np.mean(delta)),
        "absolute_change_sum": float(np.sum(np.abs(delta))),
        "changed_cells": int(np.sum(changed)),
        "changed_fraction": float(np.mean(changed)),
        "minimum_change": float(np.min(delta)),
        "maximum_change": float(np.max(delta)),
        "engine": {"xarray": package("xarray")},
    }


def spatial_autocorrelation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("libpysal")
    package("esda")
    from esda import Moran
    from libpysal.weights import W

    values = vector(inputs.get("values"), "inputs.values", minimum=5, maximum=5_000)
    neighbors_raw = mapping(inputs.get("neighbors"), "inputs.neighbors")
    neighbors: dict[int, list[int]] = {}
    for key, raw in neighbors_raw.items():
        index = int(key)
        rows = [int(v) for v in sequence(raw, f"inputs.neighbors[{key}]")]
        if index < 0 or index >= values.size or any(v < 0 or v >= values.size or v == index for v in rows):
            raise ComputeError("neighbor indices are invalid")
        neighbors[index] = rows
    if set(neighbors) != set(range(values.size)):
        raise ComputeError("neighbors must define every observation")
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    permutations = integer(inputs.get("permutations", 999), "inputs.permutations", 99, 9_999)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        moran = Moran(values, W(neighbors, silence_warnings=True), permutations=permutations)
    finally:
        np.random.set_state(state)
    return {
        "mode": "spatial_autocorrelation",
        "morans_i": float(moran.I),
        "expected_i": float(moran.EI),
        "p_value_simulation": float(moran.p_sim),
        "z_score_simulation": float(moran.z_sim),
        "permutations": permutations,
        "engines": {"libpysal": package("libpysal"), "esda": package("esda")},
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "hierarchical_bayesian_mean": hierarchical_bayesian_mean,
    "raster_zonal_statistics": raster_zonal_statistics,
    "raster_change_detection": raster_change_detection,
    "spatial_autocorrelation": spatial_autocorrelation,
}
