#!/usr/bin/env python3
"""Uniform hard-constraint validation with an independent post-solver recheck."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class ConstraintViolation(ValueError):
    """Raised when one or more hard constraints fail."""


def _resolve(context: Mapping[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConstraintViolation(f"constraint field not found: {path}")
        value = value[part]
    return value


def _numbers(value: Any) -> list[float]:
    values = [float(item) for item in value] if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [float(value)]
    if not values or not all(math.isfinite(item) for item in values):
        raise ConstraintViolation("constraint values must be finite numbers")
    return values


def evaluate_constraints(context: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    rows = profile.get("hard_constraints")
    if not isinstance(rows, list):
        raise ConstraintViolation("hard_constraints must be an array")
    violations: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def record(row: Mapping[str, Any], passed: bool, observed: Any, message: str) -> None:
        item = {"id": str(row.get("id") or ""), "type": str(row.get("type") or ""), "passed": passed, "observed": observed, "message": message}
        checks.append(item)
        if not passed:
            violations.append(item)

    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ConstraintViolation("hard constraint must be an object")
        kind = str(raw.get("type") or "")
        tolerance = float(raw.get("tolerance", 1e-8))
        field = str(raw.get("field") or "")
        try:
            if kind in {"bounds", "probability", "nonnegative", "finite", "integer"}:
                values = _numbers(_resolve(context, field))
                if kind == "probability":
                    passed, message = all(-tolerance <= item <= 1 + tolerance for item in values), "value must be within [0,1]"
                elif kind == "nonnegative":
                    passed, message = all(item >= -tolerance for item in values), "value must be nonnegative"
                elif kind == "finite":
                    passed, message = all(math.isfinite(item) for item in values), "value must be finite"
                elif kind == "integer":
                    passed, message = all(abs(item - round(item)) <= tolerance for item in values), "value must be integer-valued"
                else:
                    minimum = float(raw.get("minimum", -math.inf)); maximum = float(raw.get("maximum", math.inf))
                    passed, message = all(minimum - tolerance <= item <= maximum + tolerance for item in values), f"value must be within [{minimum},{maximum}]"
                record(raw, passed, values, str(raw.get("message") or message))
            elif kind == "sum_equals":
                fields = [str(item) for item in raw.get("fields") or []]
                observed = float(sum(float(_resolve(context, item)) for item in fields))
                target = float(raw["target"])
                record(raw, abs(observed - target) <= tolerance, observed, str(raw.get("message") or f"sum must equal {target}"))
            elif kind in {"less_equal", "greater_equal", "relation"}:
                left = float(_resolve(context, str(raw["left"])))
                right_raw = raw["right"]
                right = float(_resolve(context, right_raw)) if isinstance(right_raw, str) else float(right_raw)
                if kind == "less_equal":
                    passed = left <= right + tolerance
                elif kind == "greater_equal":
                    passed = left + tolerance >= right
                else:
                    operator = str(raw["operator"])
                    passed = {"<": left < right + tolerance, "<=": left <= right + tolerance, ">": left + tolerance > right, ">=": left + tolerance >= right, "==": abs(left - right) <= tolerance}[operator]
                record(raw, passed, {"left": left, "right": right}, str(raw.get("message") or "relation constraint failed"))
            else:
                raise ConstraintViolation(f"unsupported hard constraint type: {kind}")
        except (KeyError, TypeError, ValueError, ConstraintViolation) as exc:
            record(raw, False, None, str(exc))
    return {"schema_version": "compute-constraint-report-v1", "status": "PASS" if not violations else "FAIL", "hard_constraint_count": len(checks), "violation_count": len(violations), "checks": checks, "violations": violations, "independent_post_check_required": profile.get("independent_post_check") is True}


def enforce_constraints(context: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    report = evaluate_constraints(context, profile)
    if report["status"] != "PASS":
        identifiers = ", ".join(item["id"] or "<unnamed>" for item in report["violations"])
        raise ConstraintViolation(f"hard constraints failed: {identifiers}")
    return report


def independent_post_check(context: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    if profile.get("independent_post_check") is not True:
        raise ConstraintViolation("independent_post_check must be true")
    report = evaluate_constraints(context, profile)
    report["phase"] = "post_solver_independent_recheck"
    return report
