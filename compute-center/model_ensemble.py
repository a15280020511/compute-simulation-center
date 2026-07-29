#!/usr/bin/env python3
"""Bounded model ensembles using out-of-sample evidence only."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class EnsembleError(ValueError):
    pass


def _matrix(predictions: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    if not predictions:
        raise EnsembleError("predictions must not be empty")
    names = sorted(str(name) for name in predictions)
    rows = [np.asarray(predictions[name], dtype=float) for name in names]
    if any(row.ndim != 1 or row.size == 0 or not np.all(np.isfinite(row)) for row in rows):
        raise EnsembleError("every prediction must be a finite one-dimensional vector")
    if len({row.size for row in rows}) != 1:
        raise EnsembleError("all predictions must have equal length")
    return names, np.vstack(rows)


def _normalize(weights: np.ndarray, maximum_weight: float) -> np.ndarray:
    if maximum_weight <= 0 or maximum_weight > 1:
        raise EnsembleError("maximum_weight must be within (0,1]")
    weights = np.clip(weights, 0, None)
    if float(np.sum(weights)) <= 0:
        raise EnsembleError("ensemble weights must contain positive mass")
    weights = weights / np.sum(weights)
    for _ in range(100):
        excess = np.maximum(weights - maximum_weight, 0)
        if float(np.sum(excess)) <= 1e-12:
            break
        weights = np.minimum(weights, maximum_weight)
        free = weights < maximum_weight - 1e-12
        if not np.any(free):
            raise EnsembleError("maximum_weight is infeasible for the number of models")
        weights[free] += float(np.sum(excess)) * weights[free] / np.sum(weights[free])
    weights = weights / np.sum(weights)
    if np.any(weights > maximum_weight + 1e-9):
        raise EnsembleError("could not satisfy maximum_weight")
    return weights


def ensemble_predictions(predictions: Mapping[str, Sequence[float]], *, method: str, validation_errors: Mapping[str, float] | None = None, maximum_weight: float = 0.7) -> dict[str, Any]:
    names, matrix = _matrix(predictions)
    if method == "equal_weight":
        weights = _normalize(np.ones(len(names), dtype=float), maximum_weight)
        combined = np.average(matrix, axis=0, weights=weights)
    elif method in {"validation_weighted", "inverse_error_capped"}:
        if not validation_errors or any(name not in validation_errors for name in names):
            raise EnsembleError("validation_errors are required for weighted ensembles")
        errors = np.asarray([float(validation_errors[name]) for name in names], dtype=float)
        if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
            raise EnsembleError("validation errors must be positive finite values")
        raw = 1 / errors if method == "inverse_error_capped" else np.exp(-errors / max(float(np.median(errors)), 1e-12))
        weights = _normalize(raw, maximum_weight)
        combined = np.average(matrix, axis=0, weights=weights)
    elif method == "robust_consensus":
        weights = np.repeat(1 / len(names), len(names))
        combined = np.median(matrix, axis=0)
    else:
        raise EnsembleError(f"unsupported ensemble method: {method}")
    return {"schema_version": "compute-model-ensemble-v1", "method": method, "model_ids": names, "weights": {name: float(weight) for name, weight in zip(names, weights, strict=True)}, "prediction": [float(item) for item in combined], "single_model_maximum_weight": maximum_weight}
