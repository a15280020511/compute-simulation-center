#!/usr/bin/env python3
"""Zero-network canary for candidate certified-computation tools."""
from __future__ import annotations

import json
import platform
from importlib.metadata import version

import cvc5
from cvc5 import Kind
from flint import arb, ctx, fmpz


def check_cvc5() -> dict[str, object]:
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    integer = tm.getIntegerSort()
    x = tm.mkConst(integer, "x")
    zero = tm.mkInteger(0)
    solver.assertFormula(tm.mkTerm(Kind.GT, x, zero))
    solver.assertFormula(tm.mkTerm(Kind.LT, x, zero))
    result = solver.checkSat()
    if not result.isUnsat():
        raise RuntimeError(f"expected UNSAT, got {result}")
    return {"version": version("cvc5"), "unsat_crosscheck": True}


def check_python_flint() -> dict[str, object]:
    exact = fmpz(123456789) ** 2
    if exact != 15241578750190521:
        raise RuntimeError("FLINT exact integer arithmetic failed")
    ctx.dps = 50
    interval = arb(1) / arb(3)
    lower = float(interval.lower())
    upper = float(interval.upper())
    if not lower <= (1.0 / 3.0) <= upper:
        raise RuntimeError("Arb interval does not contain the reference value")
    return {
        "version": version("python-flint"),
        "exact_integer": True,
        "rigorous_interval": True,
        "interval_text": str(interval),
    }


def main() -> int:
    report = {
        "schema_version": "unique-compute-tools-canary-v1",
        "python": platform.python_version(),
        "cvc5": check_cvc5(),
        "python_flint": check_python_flint(),
        "network_calls": 0,
        "production_dependencies_changed": False,
        "status": "PASS",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
