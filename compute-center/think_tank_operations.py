#!/usr/bin/env python3
"""Unified allowlisted registry for top-tier think-tank analysis modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from think_tank_advanced_operations import HANDLERS as ADVANCED_HANDLERS
from think_tank_business_operations import HANDLERS as BUSINESS_HANDLERS
from think_tank_data_operations import HANDLERS as DATA_HANDLERS
from think_tank_decision_operations import HANDLERS as DECISION_HANDLERS
from think_tank_econometric_operations import HANDLERS as ECONOMETRIC_HANDLERS
from think_tank_finance_operations import HANDLERS as FINANCE_HANDLERS

HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    **DATA_HANDLERS,
    **ECONOMETRIC_HANDLERS,
    **BUSINESS_HANDLERS,
    **FINANCE_HANDLERS,
    **DECISION_HANDLERS,
    **ADVANCED_HANDLERS,
}

if len(HANDLERS) != sum(
    len(group)
    for group in (
        DATA_HANDLERS,
        ECONOMETRIC_HANDLERS,
        BUSINESS_HANDLERS,
        FINANCE_HANDLERS,
        DECISION_HANDLERS,
        ADVANCED_HANDLERS,
    )
):
    raise RuntimeError("duplicate top think-tank mode registration")

SUPPORTED_MODES = tuple(sorted(HANDLERS))
