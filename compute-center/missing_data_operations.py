#!/usr/bin/env python3
"""Bounded missing-data governance and multiple-imputation operations.

No input is overwritten. Imputed values are returned as explicitly marked estimates with
per-cell uncertainty. The module never fetches data or evaluates ticket-supplied code.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np

from compute_runner import ComputeError

MAX_ROWS = 5_000
MAX_COLUMNS = 50
MAX_MISSING_CELLS = 10_000
MAX_IMPUTATIONS = 20
EXPECTED_STATSMODELS = "0.14.6"
EXPECTED_PANDAS = "3.0.3"


def _dependencies():
    try:
        import pandas as pd
        from statsmodels.imputation.mice import MICEData
    except ImportError as exc:
        raise ComputeError("missing-data engine is not installed; install requirements-missing-data.txt") from exc
    if version("statsmodels") != EXPECTED_STATSMODELS or version("pandas") != EXPECTED_PANDAS:
        raise ComputeError(
            f"missing-data dependencies must be statsmodels=={EXPECTED_STATSMODELS} and pandas=={EXPECTED_PANDAS}"
        )
    return pd, MICEData


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _frame(inputs: Mapping[str, Any]):
    pd, _ = _dependencies()
    columns = [str(item) for item in _sequence(inputs.get("columns"), "inputs.columns")]
    if not 1 <= len(columns) <= MAX_COLUMNS or len(set(columns)) != len(columns) or any(not item for item in columns):
        raise ComputeError(f"inputs.columns must contain 1 to {MAX_COLUMNS} unique non-empty names")
    raw_rows = _sequence(inputs.get("data"), "inputs.data")
    if not 2 <= len(raw_rows) <= MAX_ROWS:
        raise ComputeError(f"inputs.data must contain 2 to {MAX_ROWS} rows")
    rows: list[list[float | None]] = []
    for i, raw in enumerate(raw_rows):
        row = _sequence(raw, f"inputs.data[{i}]")
        if len(row) != len(columns):
            raise ComputeError(f"inputs.data[{i}] must contain exactly {len(columns)} values")
        converted: list[float | None] = []
        for j, item in enumerate(row):
            if item is None:
                converted.append(None)
            else:
                converted.append(_finite(item, f"inputs.data[{i}][{j}]"))
        rows.append(converted)
    frame = pd.DataFrame(rows, columns=columns, dtype=float)
    if any(int(frame[column].notna().sum()) == 0 for column in columns):
        raise ComputeError("all-missing columns are not imputable")
    missing_cells = int(frame.isna().sum().sum())
    if missing_cells > MAX_MISSING_CELLS:
        raise ComputeError(f"missing cells cannot exceed {MAX_MISSING_CELLS}")
    return frame


def _profile(frame) -> dict[str, Any]:
    missing = frame.isna()
    rows, columns = frame.shape
    per_column = []
    for name in frame.columns:
        count = int(missing[name].sum())
        observed = frame[name].dropna().to_numpy(dtype=float)
        per_column.append(
            {
                "name": str(name),
                "missing_count": count,
                "missing_rate": count / rows,
                "observed_count": int(observed.size),
                "observed_mean": float(np.mean(observed)),
                "observed_standard_deviation": float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0,
            }
        )
    pattern_counts = missing.astype(int).astype(str).agg("".join, axis=1).value_counts().sort_index().to_dict()
    return {
        "row_count": rows,
        "column_count": columns,
        "missing_cell_count": int(missing.sum().sum()),
        "missing_cell_rate": float(missing.to_numpy().mean()),
        "complete_row_count": int((~missing.any(axis=1)).sum()),
        "complete_row_rate": float((~missing.any(axis=1)).mean()),
        "columns": per_column,
        "missingness_patterns": {str(key): int(value) for key, value in pattern_counts.items()},
    }


def _mice_samples(frame, *, imputations: int, burn_in: int, seed: int):
    _, MICEData = _dependencies()
    np.random.seed(seed)
    try:
        mice = MICEData(frame.copy(), perturbation_method="gaussian", k_pmm=5)
        for _ in range(burn_in):
            mice.update_all()
        return [mice.next_sample().copy() for _ in range(imputations)]
    except Exception as exc:
        raise ComputeError(f"MICE imputation failed: {type(exc).__name__}: {exc}") from exc


def _missingness_profile(inputs: Mapping[str, Any]) -> dict[str, Any]:
    frame = _frame(inputs)
    return {"engine": {"name": "pandas", "version": EXPECTED_PANDAS}, "mode": "missingness_profile", **_profile(frame)}


def _complete_case_bias_check(inputs: Mapping[str, Any]) -> dict[str, Any]:
    frame = _frame(inputs)
    complete = frame.dropna()
    if complete.empty:
        raise ComputeError("complete-case bias check requires at least one complete row")
    rows = []
    for name in frame.columns:
        all_observed = frame[name].dropna().to_numpy(dtype=float)
        complete_values = complete[name].to_numpy(dtype=float)
        scale = max(float(np.std(all_observed, ddof=1)) if all_observed.size > 1 else 0.0, 1e-12)
        difference = float(np.mean(complete_values) - np.mean(all_observed))
        rows.append(
            {
                "name": str(name),
                "all_observed_mean": float(np.mean(all_observed)),
                "complete_case_mean": float(np.mean(complete_values)),
                "mean_difference": difference,
                "standardized_difference": difference / scale,
            }
        )
    maximum = max(abs(row["standardized_difference"]) for row in rows)
    return {
        "engine": {"name": "numpy", "version": np.__version__},
        "mode": "complete_case_bias_check",
        "profile": _profile(frame),
        "complete_case_rows": int(len(complete)),
        "variables": rows,
        "maximum_absolute_standardized_difference": maximum,
        "bias_warning": maximum > 0.2,
    }


def _mice_multiple_imputation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    frame = _frame(inputs)
    imputations = _integer(inputs.get("imputations", 5), "inputs.imputations", 2, MAX_IMPUTATIONS)
    burn_in = _integer(inputs.get("burn_in", 5), "inputs.burn_in", 0, 50)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    missing_rows, missing_columns = np.where(frame.isna().to_numpy())
    missing_locations = [
        (int(row_index), str(frame.columns[int(column_index)]))
        for row_index, column_index in zip(missing_rows, missing_columns, strict=True)
    ]
    samples = _mice_samples(frame, imputations=imputations, burn_in=burn_in, seed=seed)
    records = []
    for row_index, column in missing_locations:
        values = [float(sample.at[row_index, column]) for sample in samples]
        records.append(
            {
                "row_index": row_index,
                "column": column,
                "observed": False,
                "method": "mice",
                "imputation_rounds": imputations,
                "imputed_values": values,
                "pooled_value": float(np.mean(values)),
                "uncertainty": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "missingness_reason": str(inputs.get("missingness_reason") or "unspecified"),
            }
        )
    pooled_columns = {}
    for column in frame.columns:
        values = np.asarray([sample[column].mean() for sample in samples], dtype=float)
        pooled_columns[str(column)] = {
            "mean_across_imputations": float(np.mean(values)),
            "between_imputation_standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        }
    return {
        "engine": {"name": "statsmodels-mice", "version": EXPECTED_STATSMODELS, "pandas": EXPECTED_PANDAS},
        "mode": "mice_multiple_imputation",
        "seed": seed,
        "imputations": imputations,
        "burn_in": burn_in,
        "profile": _profile(frame),
        "imputation_records": records,
        "pooled_columns": pooled_columns,
        "original_data_overwritten": False,
    }


def _imputation_sensitivity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    frame = _frame(inputs)
    imputations = _integer(inputs.get("imputations", 5), "inputs.imputations", 2, MAX_IMPUTATIONS)
    seed = _integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    samples = _mice_samples(frame, imputations=imputations, burn_in=5, seed=seed)
    rows = []
    for column in frame.columns:
        observed = frame[column].dropna().to_numpy(dtype=float)
        mean_fill = frame[column].fillna(float(np.mean(observed))).to_numpy(dtype=float)
        median_fill = frame[column].fillna(float(np.median(observed))).to_numpy(dtype=float)
        mice_means = np.asarray([sample[column].mean() for sample in samples], dtype=float)
        estimates = [float(np.mean(mean_fill)), float(np.mean(median_fill)), float(np.mean(mice_means))]
        rows.append(
            {
                "name": str(column),
                "mean_imputation_estimate": estimates[0],
                "median_imputation_estimate": estimates[1],
                "mice_pooled_estimate": estimates[2],
                "estimate_range": max(estimates) - min(estimates),
                "method_sensitive": (max(estimates) - min(estimates)) > 0.1 * max(float(np.std(observed)), 1e-12),
            }
        )
    return {"engine": {"name": "statsmodels-mice", "version": EXPECTED_STATSMODELS}, "mode": "imputation_sensitivity", "variables": rows, "any_method_sensitive": any(row["method_sensitive"] for row in rows)}


def _collection_priority(inputs: Mapping[str, Any]) -> dict[str, Any]:
    variables = _sequence(inputs.get("variables"), "inputs.variables")
    if not 1 <= len(variables) <= 100:
        raise ComputeError("inputs.variables must contain 1 to 100 entries")
    rows = []
    for index, raw in enumerate(variables):
        row = _mapping(raw, f"inputs.variables[{index}]")
        name = str(row.get("name") or "").strip()
        if not name:
            raise ComputeError("collection-priority variable name is required")
        missing_rate = _finite(row.get("missing_rate"), f"variables[{index}].missing_rate")
        sensitivity = _finite(row.get("sensitivity"), f"variables[{index}].sensitivity")
        decision_impact = _finite(row.get("decision_impact", 1.0), f"variables[{index}].decision_impact")
        collection_cost = _finite(row.get("collection_cost", 1.0), f"variables[{index}].collection_cost")
        if not 0 <= missing_rate <= 1 or sensitivity < 0 or decision_impact < 0 or collection_cost <= 0:
            raise ComputeError("collection-priority values are outside valid bounds")
        score = missing_rate * sensitivity * decision_impact / collection_cost
        rows.append({"name": name, "priority_score": score, "missing_rate": missing_rate, "sensitivity": sensitivity, "decision_impact": decision_impact, "collection_cost": collection_cost})
    rows.sort(key=lambda item: (-item["priority_score"], item["name"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {"engine": {"name": "deterministic-value-of-information-screen", "version": 1}, "mode": "collection_priority", "variables": rows}


def missing_data_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handlers = {
        "missingness_profile": _missingness_profile,
        "complete_case_bias_check": _complete_case_bias_check,
        "mice_multiple_imputation": _mice_multiple_imputation,
        "imputation_sensitivity": _imputation_sensitivity,
        "collection_priority": _collection_priority,
    }
    if mode not in handlers:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(handlers))}")
    return handlers[mode](inputs)


OPERATIONS = {"missing_data_analysis": missing_data_analysis}
