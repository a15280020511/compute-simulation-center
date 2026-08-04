#!/usr/bin/env python3
"""Network-denied GEKKO backend for bounded deterministic optimization."""
from __future__ import annotations

from typing import Any

import numpy as np

from compute_runner import ComputeError


def solve_nonnegative_linear_program(
    objective: np.ndarray,
    constraints: np.ndarray,
    bounds: np.ndarray,
    *,
    maximize: bool,
) -> dict[str, Any]:
    """Solve a bounded matrix-defined LP with GEKKO's local solver only."""
    from gekko import GEKKO

    model = GEKKO(remote=False)
    variables = [model.Var(lb=0.0, value=0.0) for _ in range(objective.size)]
    for row_index in range(constraints.shape[0]):
        expression = sum(
            float(constraints[row_index, column_index]) * variables[column_index]
            for column_index in range(objective.size)
        )
        model.Equation(expression <= float(bounds[row_index]))

    objective_expression = sum(
        float(objective[index]) * variables[index] for index in range(objective.size)
    )
    if maximize:
        model.Maximize(objective_expression)
    else:
        model.Minimize(objective_expression)

    model.options.SOLVER = 1
    model.options.DIAGLEVEL = 0
    try:
        model.solve(disp=False)
    except Exception as exc:
        raise ComputeError(
            f"GEKKO local optimization failed: {type(exc).__name__}: {exc}"
        ) from exc

    application_status = int(model.options.APPSTATUS)
    if application_status != 1:
        raise ComputeError(
            f"GEKKO local optimization did not converge: APPSTATUS={application_status}"
        )

    values = np.asarray([float(variable.value[0]) for variable in variables], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ComputeError("GEKKO returned non-finite decision values")
    objective_value = float(np.dot(objective, values))
    return {
        "decision": values.tolist(),
        "objective_value": objective_value,
        "termination": "optimal-local",
        "solver": "APOPT",
        "remote": False,
    }
