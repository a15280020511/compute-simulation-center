#!/usr/bin/env python3
"""Normalize registered operation outputs into canonical JSON-compatible values.

Third-party scientific packages can return NumPy scalar and array objects even when the
numerical result is valid. Compute artifacts and hashes require native JSON values, so every
managed operation is wrapped at the registry boundary. No value is rounded or reinterpreted.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

import numpy as np

_WRAPPERS: dict[Callable[..., Any], Callable[..., Any]] = {}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=lambda item: str(item))]
    return value


def wrap_operation(handler: Callable[..., Any]) -> Callable[..., Any]:
    cached = _WRAPPERS.get(handler)
    if cached is not None:
        return cached

    @wraps(handler)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return json_safe(handler(*args, **kwargs))

    _WRAPPERS[handler] = wrapped
    return wrapped
