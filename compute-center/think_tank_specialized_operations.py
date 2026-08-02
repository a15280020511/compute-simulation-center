#!/usr/bin/env python3
"""Unified registry for second-wave specialized think-tank modes."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Callable
from think_tank_specialized_assurance import HANDLERS as ASSURANCE
from think_tank_specialized_compat import HANDLERS as COMPAT
from think_tank_specialized_decision import HANDLERS as DECISION
from think_tank_specialized_finance_knowledge import HANDLERS as FINANCE
from think_tank_specialized_forecast import HANDLERS as FORECAST
from think_tank_specialized_health import HANDLERS as HEALTH
from think_tank_specialized_operations_domain import HANDLERS as OPERATIONS
from think_tank_specialized_infrastructure import HANDLERS as INFRASTRUCTURE
from think_tank_specialized_policy import HANDLERS as POLICY
from think_tank_specialized_spatial import HANDLERS as SPATIAL
HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {}
for group in (POLICY,FORECAST,DECISION,SPATIAL,INFRASTRUCTURE,HEALTH,OPERATIONS,FINANCE,ASSURANCE):
    duplicate=set(HANDLERS)&set(group)
    if duplicate: raise RuntimeError(f"duplicate specialized mode: {sorted(duplicate)}")
    HANDLERS.update(group)
if not set(COMPAT).issubset(HANDLERS):
    raise RuntimeError(f"compatibility adapter references unknown mode: {sorted(set(COMPAT)-set(HANDLERS))}")
HANDLERS.update(COMPAT)
SUPPORTED_MODES=tuple(sorted(HANDLERS))
