#!/usr/bin/env python3
"""Single allowlisted gateway for legacy and social Mesa modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compute_runner import ComputeError
from mesa_operations import agent_based_simulation as legacy_agent_based_simulation
from social_behavior_operations import MODES as SOCIAL_MODES
from social_behavior_operations import social_behavior_simulation

LEGACY_MODES = {"heterogeneous_worker_choice", "network_contagion", "resource_competition"}


def agent_based_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode in LEGACY_MODES:
        return legacy_agent_based_simulation(inputs)
    if mode in SOCIAL_MODES:
        return social_behavior_simulation(inputs)
    raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(LEGACY_MODES | SOCIAL_MODES))}")


OPERATIONS = {"agent_based_simulation": agent_based_simulation}
