#!/usr/bin/env python3
"""Shared helpers for the Exa-discovered institutional toolkit."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from compute_runner import ComputeError
from think_tank_common import finite, integer, mapping, matrix, package, sequence, vector

MAX_TEXT_BYTES = 200_000
MAX_RECORDS = 20_000
MAX_FEATURES = 50
MAX_SERIES = 100
MAX_FORECAST_HORIZON = 365
MAX_GRAPH_TRIPLES = 50_000


def strings(value: Any, name: str, *, minimum: int = 1, maximum: int = MAX_RECORDS) -> list[str]:
    rows = sequence(value, name)
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} values")
    result = [str(item) for item in rows]
    if any(len(item) > 10_000 for item in result):
        raise ComputeError(f"{name} contains an oversized string")
    return result


def safe_names(value: Any, name: str, *, maximum: int = MAX_FEATURES) -> list[str]:
    rows = strings(value, name, maximum=maximum)
    if len(set(rows)) != len(rows):
        raise ComputeError(f"{name} must contain unique names")
    for item in rows:
        if not item or item[0].isdigit() or not item.replace("_", "").isalnum():
            raise ComputeError(f"{name} contains an unsafe identifier")
    return rows


def bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ComputeError(f"{name} must be boolean")
    return value


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ComputeError("non-finite result")
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


def engine(*names: str) -> dict[str, str]:
    return {name: package(name) for name in names}


def bounded_text(value: Any, name: str, *, maximum_bytes: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise ComputeError(f"{name} must be text")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ComputeError(f"{name} exceeds byte limit")
    return value


def equal_length(name: str, *arrays: Sequence[Any]) -> None:
    lengths = {len(item) for item in arrays}
    if len(lengths) != 1:
        raise ComputeError(f"{name} arrays must have equal length")
