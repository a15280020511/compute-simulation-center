#!/usr/bin/env python3
"""Shared validation helpers for bounded think-tank capabilities."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from compute_runner import ComputeError

MAX_ROWS = 50_000
MAX_COLUMNS = 100
MAX_ASSETS = 50
MAX_SCENARIOS = 2_000
MAX_GRID_CELLS = 1_000_000
MAX_ACTORS = 50
MAX_PERIODS = 1_000


def mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def probability(value: Any, name: str) -> float:
    result = finite(value, name)
    if not 0 <= result <= 1:
        raise ComputeError(f"{name} must be between 0 and 1")
    return result


def vector(value: Any, name: str, minimum: int = 1, maximum: int = MAX_ROWS) -> np.ndarray:
    rows = sequence(value, name)
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} values")
    return np.asarray([finite(item, f"{name}[{i}]") for i, item in enumerate(rows)], dtype=float)


def matrix(
    value: Any,
    name: str,
    *,
    min_rows: int = 1,
    max_rows: int = MAX_ROWS,
    min_columns: int = 1,
    max_columns: int = MAX_COLUMNS,
) -> np.ndarray:
    rows = sequence(value, name)
    if not min_rows <= len(rows) <= max_rows:
        raise ComputeError(f"{name} must contain {min_rows} to {max_rows} rows")
    parsed: list[list[float]] = []
    width: int | None = None
    for i, raw in enumerate(rows):
        row = [finite(item, f"{name}[{i}][{j}]") for j, item in enumerate(sequence(raw, f"{name}[{i}]"))]
        if not min_columns <= len(row) <= max_columns:
            raise ComputeError(f"{name}[{i}] must contain {min_columns} to {max_columns} values")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ComputeError(f"{name} rows must have equal length")
        parsed.append(row)
    return np.asarray(parsed, dtype=float)


def package(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise ComputeError(f"required optional package is not installed: {name}") from exc


def identifiers(value: Any, name: str, maximum: int = MAX_COLUMNS) -> list[str]:
    rows = [str(item) for item in sequence(value, name)]
    invalid = (
        not rows
        or len(rows) > maximum
        or any(not item for item in rows)
        or len(set(rows)) != len(rows)
        or any(not item.replace("_", "").isalnum() or item[0].isdigit() for item in rows)
    )
    if invalid:
        raise ComputeError(f"{name} must contain 1 to {maximum} unique safe identifiers")
    return rows


def records(value: Any, name: str, maximum: int = MAX_ROWS) -> list[dict[str, Any]]:
    rows = sequence(value, name)
    if not 1 <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain 1 to {maximum} records")
    result = []
    for i, raw in enumerate(rows):
        row = mapping(raw, f"{name}[{i}]")
        if len(row) > MAX_COLUMNS:
            raise ComputeError(f"{name}[{i}] contains too many fields")
        result.append(dict(row))
    return result
