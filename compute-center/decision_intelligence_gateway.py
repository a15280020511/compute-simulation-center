#!/usr/bin/env python3
"""Single allowlisted gateway for finance, quantitative, strategic and intelligence modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from assurance_operations import HANDLERS as ASSURANCE_HANDLERS
from compute_runner import ComputeError
from finance_operations import finance_decision_analysis as legacy_finance_decision_analysis
from indirect_intelligence_operations import HANDLERS as INDIRECT_INTELLIGENCE_HANDLERS
from institutional_expansion_operations import HANDLERS as INSTITUTIONAL_EXPANSION_HANDLERS
from operations_research_modes import HANDLERS as OR_HANDLERS
from professional_forecasting_operations import HANDLERS as FORECAST_HANDLERS
from quantitative_operations import HANDLERS as QUANT_HANDLERS
from strategic_intelligence_operations import HANDLERS as STRATEGIC_HANDLERS
from think_tank_operations import HANDLERS as THINK_TANK_HANDLERS
from uncertainty_factor_accuracy_operations import HANDLERS as UNCERTAINTY_FACTOR_ACCURACY_HANDLERS

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
PREVIEW_MODES = (
    set(THINK_TANK_HANDLERS)
    | set(ASSURANCE_HANDLERS)
    | set(INSTITUTIONAL_EXPANSION_HANDLERS)
    | set(UNCERTAINTY_FACTOR_ACCURACY_HANDLERS)
)

MODE_HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    **QUANT_HANDLERS,
    **FORECAST_HANDLERS,
    **OR_HANDLERS,
    **STRATEGIC_HANDLERS,
    **THINK_TANK_HANDLERS,
    **ASSURANCE_HANDLERS,
    **INSTITUTIONAL_EXPANSION_HANDLERS,
    **UNCERTAINTY_FACTOR_ACCURACY_HANDLERS,
}

# Controlled-preview overlays are intentionally excluded from PREVIEW_MODES and
# ALL_SUPPORTED_MODES so an incremental, repository-governed extension cannot
# silently mutate the static decision-gateway baseline/cardinality contract.
# Capability Manager V2 owns discovery, pinned dependencies, limits and maturity
# for these overlays; the gateway only provides their allowlisted runtime handler.
CONTROLLED_PREVIEW_OVERLAY_HANDLERS: dict[
    str, Callable[[Mapping[str, Any]], dict[str, Any]]
] = {
    **INDIRECT_INTELLIGENCE_HANDLERS,
}
CONTROLLED_PREVIEW_OVERLAY_MODES = tuple(sorted(CONTROLLED_PREVIEW_OVERLAY_HANDLERS))

if PRODUCTION_MODES & PREVIEW_MODES:
    raise RuntimeError("production and controlled-preview decision modes must not overlap")
if (PRODUCTION_MODES | PREVIEW_MODES) & set(CONTROLLED_PREVIEW_OVERLAY_HANDLERS):
    raise RuntimeError("controlled-preview overlay must not overlap the stable gateway baseline")
if len(MODE_HANDLERS) != sum(
    len(group)
    for group in (
        QUANT_HANDLERS,
        FORECAST_HANDLERS,
        OR_HANDLERS,
        STRATEGIC_HANDLERS,
        THINK_TANK_HANDLERS,
        ASSURANCE_HANDLERS,
        INSTITUTIONAL_EXPANSION_HANDLERS,
        UNCERTAINTY_FACTOR_ACCURACY_HANDLERS,
    )
):
    raise RuntimeError("duplicate decision-intelligence mode registration")

SUPPORTED_MODES = tuple(sorted(PRODUCTION_MODES))
ALL_SUPPORTED_MODES = tuple(sorted(PRODUCTION_MODES | PREVIEW_MODES))


def finance_decision_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    overlay_handler = CONTROLLED_PREVIEW_OVERLAY_HANDLERS.get(mode)
    if overlay_handler is not None:
        result = overlay_handler(inputs)
        result.setdefault("runtime_registration", "controlled-preview-overlay")
    elif mode in LEGACY_MODES:
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
