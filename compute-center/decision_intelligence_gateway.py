#!/usr/bin/env python3
"""Single allowlisted gateway for finance, quantitative, strategic and intelligence modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from compute_runner import ComputeError
from finance_operations import finance_decision_analysis as legacy_finance_decision_analysis
from operations_research_modes import HANDLERS as OR_HANDLERS
from professional_forecasting_operations import HANDLERS as FORECAST_HANDLERS
from quantitative_operations import HANDLERS as QUANT_HANDLERS
from strategic_intelligence_operations import HANDLERS as STRATEGIC_HANDLERS
from think_tank_operations import HANDLERS as THINK_TANK_HANDLERS

LEGACY_MODES = {
    "performance_metrics",
    "portfolio_optimization",
    "investment_projection",
    "business_unit_economics",
    "capital_budgeting",
    "strategy_backtest",
}

PRODUCTION_HANDLER_MODES = (
    set(QUANT_HANDLERS)
    | set(FORECAST_HANDLERS)
    | set(OR_HANDLERS)
    | set(STRATEGIC_HANDLERS)
)
PRODUCTION_MODES = LEGACY_MODES | PRODUCTION_HANDLER_MODES
PREVIEW_MODES = set(THINK_TANK_HANDLERS)

MODE_HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    **QUANT_HANDLERS,
    **FORECAST_HANDLERS,
    **OR_HANDLERS,
    **STRATEGIC_HANDLERS,
    **THINK_TANK_HANDLERS,
}

if PRODUCTION_MODES & PREVIEW_MODES:
    raise RuntimeError("production and controlled-preview decision modes must not overlap")

# Backward-compatible production set used by the established 22-mode gate.
SUPPORTED_MODES = tuple(sorted(PRODUCTION_MODES))
# Complete executable allowlist, including the separately governed preview extension.
ALL_SUPPORTED_MODES = tuple(sorted(PRODUCTION_MODES | PREVIEW_MODES))


def finance_decision_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode in LEGACY_MODES:
        result = legacy_finance_decision_analysis(inputs)
    else:
        handler = MODE_HANDLERS.get(mode)
        if handler is None:
            raise ComputeError(f"unsupported decision-intelligence mode: {mode}")
        result = handler(inputs)
    result.setdefault("decision_support_only", True)
    result.setdefault("no_guaranteed_profit", True)
    result.setdefault("external_data_fetches", 0)
    result.setdefault("brokerage_execution", False)
    result.setdefault("arbitrary_code_allowed", False)
    return result


OPERATIONS = {"finance_decision_analysis": finance_decision_analysis}
