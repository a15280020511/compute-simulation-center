#!/usr/bin/env python3
"""Out-of-sample model structure comparison against a simple baseline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class ModelComparisonError(ValueError):
    pass


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ModelComparisonError(f"{name} must be a non-empty finite vector")
    return array


def compare_models(actual: Sequence[float], predictions: Mapping[str, Sequence[float]], *, baseline_model_id: str, complexity: Mapping[str, int] | None = None, minimum_improvement_over_baseline: float = 0.0) -> dict[str, Any]:
    observed = _vector(actual, "actual")
    if baseline_model_id not in predictions:
        raise ModelComparisonError("baseline_model_id is missing from predictions")
    complexity = complexity or {}
    rows: list[dict[str, Any]] = []
    for model_id, raw in predictions.items():
        predicted = _vector(raw, f"predictions[{model_id}]")
        if predicted.shape != observed.shape:
            raise ModelComparisonError(f"prediction length mismatch for {model_id}")
        error = predicted - observed
        rows.append({"model_id": str(model_id), "metrics": {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}, "complexity": int(complexity.get(model_id, 1))})
    baseline = next(row for row in rows if row["model_id"] == baseline_model_id)
    baseline_rmse = max(float(baseline["metrics"]["rmse"]), 1e-15)
    for row in rows:
        improvement = (baseline_rmse - float(row["metrics"]["rmse"])) / baseline_rmse
        row["improvement_over_baseline"] = float(improvement)
        row["beats_baseline"] = row["model_id"] == baseline_model_id or improvement >= minimum_improvement_over_baseline
        row["selection_score"] = float(row["metrics"]["rmse"] * (1 + 0.001 * row["complexity"]))
    eligible = [row for row in rows if row["beats_baseline"]]
    selected = min(eligible, key=lambda row: (row["selection_score"], row["complexity"], row["model_id"]))
    rows.sort(key=lambda row: (row["selection_score"], row["complexity"], row["model_id"]))
    failed = [row["model_id"] for row in rows if row["model_id"] != baseline_model_id and not row["beats_baseline"]]
    return {"schema_version": "compute-model-comparison-v1", "baseline_model_id": baseline_model_id, "selected_model_id": selected["model_id"], "minimum_improvement_over_baseline": minimum_improvement_over_baseline, "models": rows, "complex_models_not_better_than_baseline": failed, "decision_rule": "Prefer the lowest out-of-sample error among models that meet the baseline-improvement threshold; break ties by lower complexity."}
