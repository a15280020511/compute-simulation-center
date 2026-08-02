#!/usr/bin/env python3
"""Compatibility adapters for version-sensitive specialized think-tank tools."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, matrix, package, sequence, vector


def hierarchical_forecast_reconciliation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("hierarchicalforecast")
    package("pandas")
    import pandas as pd
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, MinTrace

    series_ids = [str(v) for v in sequence(inputs.get("series_ids"), "inputs.series_ids")]
    bottom_ids = [str(v) for v in sequence(inputs.get("bottom_ids"), "inputs.bottom_ids")]
    s = matrix(inputs.get("summing_matrix"), "inputs.summing_matrix", min_rows=2, max_rows=500, min_columns=1, max_columns=499)
    base = matrix(inputs.get("base_forecasts"), "inputs.base_forecasts", min_rows=len(series_ids), max_rows=len(series_ids), min_columns=1, max_columns=365)
    if s.shape != (len(series_ids), len(bottom_ids)) or base.shape[0] != len(series_ids):
        raise ComputeError("hierarchy matrices do not align with identifiers")
    method = str(inputs.get("method") or "mint_ols")
    if method not in {"bottom_up", "mint_ols"}:
        raise ComputeError("method must be bottom_up or mint_ols")

    horizon = base.shape[1]
    yhat = pd.DataFrame([
        {"unique_id": sid, "ds": step, "base": float(base[i, step])}
        for i, sid in enumerate(series_ids)
        for step in range(horizon)
    ])
    sdf = pd.DataFrame(s, columns=bottom_ids)
    sdf.insert(0, "unique_id", series_ids)
    tags = {"all": np.asarray(series_ids), "bottom": np.asarray(bottom_ids)}
    reconciler = BottomUp() if method == "bottom_up" else MinTrace(method="ols")
    execution = "hierarchicalforecast"
    try:
        output = HierarchicalReconciliation([reconciler]).reconcile(Y_hat_df=yhat, S=sdf, tags=tags)
        candidates = [c for c in output.columns if c not in {"unique_id", "ds", "base"}]
        if not candidates:
            raise ComputeError("HierarchicalForecast returned no reconciled column")
        column = candidates[-1]
        reconciled = np.vstack([
            output.loc[output["unique_id"] == sid].sort_values("ds")[column].to_numpy(dtype=float)
            for sid in series_ids
        ])
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        execution = f"closed-form-{method}"
        bottom_index = [series_ids.index(sid) for sid in bottom_ids]
        if method == "bottom_up":
            reconciled = s @ base[bottom_index, :]
        else:
            projection = s @ np.linalg.pinv(s.T @ s) @ s.T
            reconciled = projection @ base
        fallback_error = f"{type(exc).__name__}: {exc}"
    if not np.all(np.isfinite(reconciled)):
        raise ComputeError("reconciled forecasts contain non-finite values")
    result = {
        "mode": "hierarchical_forecast_reconciliation",
        "method": method,
        "series_ids": series_ids,
        "horizon": horizon,
        "reconciled_forecasts": reconciled.tolist(),
        "coherence_residual_max": float(np.max(np.abs(reconciled - s @ reconciled[[series_ids.index(v) for v in bottom_ids], :]))),
        "execution": execution,
        "engine": {"hierarchicalforecast": package("hierarchicalforecast")},
    }
    if execution != "hierarchicalforecast":
        result["api_compatibility_note"] = fallback_error
    return result


def scenario_discovery_prim(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("ema-workbench")
    package("pandas")
    import pandas as pd
    from ema_workbench.analysis import prim

    experiments = matrix(inputs.get("experiments"), "inputs.experiments", min_rows=30, max_rows=50_000, min_columns=1, max_columns=30)
    outcomes = vector(inputs.get("outcomes"), "inputs.outcomes", minimum=experiments.shape[0], maximum=experiments.shape[0])
    threshold = finite(inputs.get("threshold"), "inputs.threshold")
    threshold_type = str(inputs.get("threshold_type") or ">")
    if threshold_type not in {">", "<"}:
        raise ComputeError("threshold_type must be > or <")
    peel_alpha = finite(inputs.get("peel_alpha", 0.05), "inputs.peel_alpha")
    if not 0 < peel_alpha < 0.5:
        raise ComputeError("peel_alpha must be between zero and 0.5")
    frame = pd.DataFrame(experiments, columns=[f"x{i}" for i in range(experiments.shape[1])])
    direction = getattr(prim, "ABOVE", 1) if threshold_type == ">" else getattr(prim, "BELOW", -1)
    algorithm = prim.Prim(frame, outcomes, threshold=threshold, threshold_type=direction, peel_alpha=peel_alpha)
    box = algorithm.find_box()
    trajectory = getattr(box, "peeling_trajectory", pd.DataFrame())
    limits = getattr(box, "limits", None)
    if isinstance(limits, pd.DataFrame):
        limits_value = limits.to_dict(orient="index")
    elif limits is None:
        limits_value = {}
    else:
        limits_value = str(limits)
    selected = np.asarray(getattr(box, "yi", []), dtype=int)
    return {
        "mode": "scenario_discovery_prim",
        "threshold": threshold,
        "threshold_type": threshold_type,
        "selected_count": int(selected.size),
        "selected_indices": selected[:10_000].tolist(),
        "peeling_trajectory": trajectory.to_dict(orient="records") if isinstance(trajectory, pd.DataFrame) else [],
        "box_limits": limits_value,
        "engine": {"ema-workbench": package("ema-workbench")},
    }


def spatial_regression(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("spreg")
    import spreg
    from libpysal.weights import W

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20, max_rows=5_000, min_columns=1, max_columns=30)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    raw = inputs.get("neighbors")
    if not isinstance(raw, Mapping):
        raise ComputeError("neighbors must be an object")
    neighbors = {int(k): [int(v) for v in values] for k, values in raw.items()}
    if set(neighbors) != set(range(x.shape[0])):
        raise ComputeError("neighbors must cover every observation index")
    weights = W(neighbors, silence_warnings=True)
    weights.transform = "r"
    model = spreg.ML_Lag(y.reshape(-1, 1), x, w=weights, name_y="y", name_x=[f"x{i}" for i in range(x.shape[1])])
    betas = np.asarray(model.betas, dtype=float).reshape(-1)
    return {
        "mode": "spatial_regression",
        "coefficients": betas.tolist(),
        "rho": float(model.rho),
        "log_likelihood": float(model.logll),
        "aic": float(model.aic),
        "bic": float(model.schwarz),
        "observations": int(x.shape[0]),
        "engine": {"spreg": package("spreg")},
    }


def energy_capacity_expansion(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pypsa")
    package("pandas")
    import pandas as pd
    import pypsa

    demand = vector(inputs.get("demand"), "inputs.demand", minimum=2, maximum=8_760)
    marginal = vector(inputs.get("marginal_costs"), "inputs.marginal_costs", minimum=1, maximum=20)
    capital = vector(inputs.get("capital_costs"), "inputs.capital_costs", minimum=marginal.size, maximum=marginal.size)
    availability = matrix(inputs.get("availability"), "inputs.availability", min_rows=demand.size, max_rows=demand.size, min_columns=marginal.size, max_columns=marginal.size)
    if np.any(demand < 0) or np.any(marginal < 0) or np.any(capital < 0) or np.any((availability < 0) | (availability > 1)):
        raise ComputeError("energy inputs are outside valid bounds")
    network = pypsa.Network()
    snapshots = pd.RangeIndex(demand.size)
    network.set_snapshots(snapshots)
    network.add("Bus", "system")
    network.add("Load", "demand", bus="system", p_set=pd.Series(demand, index=snapshots))
    for i in range(marginal.size):
        network.add(
            "Generator",
            f"g{i}",
            bus="system",
            p_nom_extendable=True,
            marginal_cost=float(marginal[i]),
            capital_cost=float(capital[i]),
            p_max_pu=pd.Series(availability[:, i], index=snapshots),
        )
    status, condition = network.optimize(solver_name="highs")
    if str(status).lower() not in {"ok", "warning"}:
        raise ComputeError(f"PyPSA optimization failed: {status}/{condition}")
    capacities = network.generators.p_nom_opt.astype(float)
    dispatch = network.generators_t.p.astype(float)
    return {
        "mode": "energy_capacity_expansion",
        "solver_status": str(status),
        "termination_condition": str(condition),
        "optimal_capacity": {str(k): float(v) for k, v in capacities.items()},
        "dispatch": {str(k): [float(v) for v in dispatch[k].to_numpy()] for k in dispatch.columns},
        "objective": float(network.objective),
        "engine": {"pypsa": package("pypsa")},
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "hierarchical_forecast_reconciliation": hierarchical_forecast_reconciliation,
    "scenario_discovery_prim": scenario_discovery_prim,
    "spatial_regression": spatial_regression,
    "energy_capacity_expansion": energy_capacity_expansion,
}
