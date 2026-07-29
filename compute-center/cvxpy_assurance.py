#!/usr/bin/env python3
"""Fixed CVXPY models and numerical constraint-residual audit."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np

class ConstraintAssuranceError(ValueError):
    pass


def _v(value: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or a.size == 0 or a.size > 500 or not np.isfinite(a).all():
        raise ConstraintAssuranceError(f'{name} must be a non-empty finite vector')
    return a


def convex_resource_allocation(costs: Sequence[float], minimums: Sequence[float], maximums: Sequence[float], total: float) -> dict[str, Any]:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ConstraintAssuranceError('CVXPY optional dependency is not installed') from exc
    c, lo, hi = _v(costs, 'costs'), _v(minimums, 'minimums'), _v(maximums, 'maximums')
    if not (c.shape == lo.shape == hi.shape) or np.any(lo > hi) or not float(lo.sum()) <= total <= float(hi.sum()):
        raise ConstraintAssuranceError('inconsistent bounds or infeasible total')
    x = cp.Variable(c.size)
    constraints = [x >= lo, x <= hi, cp.sum(x) == float(total)]
    problem = cp.Problem(cp.Minimize(c @ x), constraints)
    if not problem.is_dcp():
        raise ConstraintAssuranceError('fixed model unexpectedly violates DCP')
    problem.solve()
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        raise ConstraintAssuranceError(f'optimization failed: {problem.status}')
    residuals = [float(np.max(np.atleast_1d(item.violation()))) for item in constraints]
    duals = [np.asarray(item.dual_value).tolist() for item in constraints]
    return {'schema_version': 'cvxpy-resource-allocation-v1', 'status': problem.status, 'objective': float(problem.value), 'allocation': np.asarray(x.value).tolist(), 'dcp': problem.is_dcp(), 'maximum_constraint_violation': max(residuals), 'constraint_residuals': residuals, 'dual_values': duals}


def constraint_residual_audit(values: Sequence[float], minimums: Sequence[float], maximums: Sequence[float], total: float | None = None, tolerance: float = 1e-7) -> dict[str, Any]:
    x, lo, hi = _v(values, 'values'), _v(minimums, 'minimums'), _v(maximums, 'maximums')
    if not (x.shape == lo.shape == hi.shape):
        raise ConstraintAssuranceError('values and bounds must have equal shape')
    lower = np.maximum(lo - x, 0.0); upper = np.maximum(x - hi, 0.0)
    equality = 0.0 if total is None else abs(float(x.sum()) - float(total))
    maximum = max(float(lower.max(initial=0.0)), float(upper.max(initial=0.0)), equality)
    return {'schema_version': 'constraint-residual-audit-v1', 'maximum_violation': maximum, 'lower_bound_violation': lower.tolist(), 'upper_bound_violation': upper.tolist(), 'sum_equality_violation': equality, 'feasible_within_tolerance': maximum <= tolerance}
