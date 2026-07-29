#!/usr/bin/env python3
"""Uniform hard-constraint validation with an independent post-solver recheck."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    if not values or not all(math.isfinite(item) for item in values):
        raise ConstraintViolation("constraint values must be finite numbers")
    return values


def _scalar_check(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    tolerance: float,
) -> tuple[bool, Any, str]:
    kind = str(row.get("type") or "")
    values = _numbers(_resolve(context, str(row.get("field") or "")))
    checks: dict[str, tuple[Callable[[float], bool], str]] = {
        "probability": (
            lambda item: -tolerance <= item <= 1 + tolerance,
            "value must be within [0,1]",
        ),
        "nonnegative": (
            lambda item: item >= -tolerance,
            "value must be nonnegative",
        ),
        "finite": (math.isfinite, "value must be finite"),
        "integer": (
            lambda item: abs(item - round(item)) <= tolerance,
            "value must be integer-valued",
        ),
    }
    if kind in checks:
        predicate, message = checks[kind]
        return all(predicate(item) for item in values), values, message
    minimum = float(row.get("minimum", -math.inf))
    maximum = float(row.get("maximum", math.inf))
    passed = all(
        minimum - tolerance <= item <= maximum + tolerance for item in values
    )
    return passed, values, f"value must be within [{minimum},{maximum}]"


def _sum_check(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    tolerance: float,
) -> tuple[bool, Any, str]:
    fields = [str(item) for item in row.get("fields") or []]
    observed = float(sum(float(_resolve(context, item)) for item in fields))
    target = float(row["target"])
    return abs(observed - target) <= tolerance, observed, f"sum must equal {target}"


def _relation_check(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    tolerance: float,
) -> tuple[bool, Any, str]:
    kind = str(row.get("type") or "")
    left = float(_resolve(context, str(row["left"])))
    right_raw = row["right"]
    right = (
        float(_resolve(context, right_raw))
        if isinstance(right_raw, str)
        else float(right_raw)
    )
    operator = str(row.get("operator") or "")
    predicates = {
        "less_equal": left <= right + tolerance,
        "greater_equal": left + tolerance >= right,
        "<": left < right + tolerance,
        "<=": left <= right + tolerance,
        ">": left + tolerance > right,
        ">=": left + tolerance >= right,
        "==": abs(left - right) <= tolerance,
    }
    key = kind if kind != "relation" else operator
    if key not in predicates:
        raise ConstraintViolation(f"unsupported relation operator: {operator}")
    return predicates[key], {"left": left, "right": right}, "relation constraint failed"


def _evaluate_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[bool, Any, str]:
    kind = str(row.get("type") or "")
    tolerance = float(row.get("tolerance", 1e-8))
    if kind in {"bounds", "probability", "nonnegative", "finite", "integer"}:
        return _scalar_check(row, context, tolerance)
    if kind == "sum_equals":
        return _sum_check(row, context, tolerance)
    if kind in {"less_equal", "greater_equal", "relation"}:
        return _relation_check(row, context, tolerance)
    raise ConstraintViolation(f"unsupported hard constraint type: {kind}")


def _record(
    row: Mapping[str, Any],
    passed: bool,
    observed: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "type": str(row.get("type") or ""),
        "passed": passed,
        "observed": observed,
        "message": str(row.get("message") or message),
    }


def evaluate_constraints(
    context: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    rows = profile.get("hard_constraints")
    if not isinstance(rows, list):
        raise ConstraintViolation("hard_constraints must be an array")
    checks: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ConstraintViolation("hard constraint must be an object")
        try:
            item = _record(raw, *_evaluate_row(raw, context))
        except (KeyError, TypeError, ValueError, ConstraintViolation) as exc:
            item = _record(raw, False, None, str(exc))
        checks.append(item)
        if not item["passed"]:
            violations.append(item)
    return {
        "schema_version": "compute-constraint-report-v1",
        "status": "PASS" if not violations else "FAIL",
        "hard_constraint_count": len(checks),
        "violation_count": len(violations),
        "checks": checks,
        "violations": violations,
        "independent_post_check_required": profile.get("independent_post_check") is True,
    }


def enforce_constraints(
    context: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    report = evaluate_constraints(context, profile)
    if report["status"] != "PASS":
        identifiers = ", ".join(
            item["id"] or "<unnamed>" for item in report["violations"]
        )
        raise ConstraintViolation(f"hard constraints failed: {identifiers}")
    return report


def independent_post_check(
    context: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    if profile.get("independent_post_check") is not True:
        raise ConstraintViolation("independent_post_check must be true")
    report = evaluate_constraints(context, profile)
    report["phase"] = "post_solver_independent_recheck"
    return report
