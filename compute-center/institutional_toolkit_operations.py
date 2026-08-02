#!/usr/bin/env python3
"""Aggregate Exa-discovered institutional toolkit handlers."""
from __future__ import annotations

from institutional_decision_spatial_operations import HANDLERS as DECISION_SPATIAL_HANDLERS
from institutional_economics_operations import HANDLERS as ECONOMICS_HANDLERS
from institutional_forecasting_operations import HANDLERS as FORECASTING_HANDLERS
from institutional_infrastructure_finance_operations import HANDLERS as INFRASTRUCTURE_FINANCE_HANDLERS
from institutional_knowledge_engineering_operations import HANDLERS as KNOWLEDGE_ENGINEERING_HANDLERS

HANDLERS = {
    **ECONOMICS_HANDLERS,
    **FORECASTING_HANDLERS,
    **DECISION_SPATIAL_HANDLERS,
    **INFRASTRUCTURE_FINANCE_HANDLERS,
    **KNOWLEDGE_ENGINEERING_HANDLERS,
}

if len(HANDLERS) != 41:
    raise RuntimeError(f"institutional toolkit must expose exactly 41 unique modes, found {len(HANDLERS)}")

MODES = tuple(sorted(HANDLERS))
