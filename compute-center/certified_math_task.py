#!/usr/bin/env python3
"""Offline, bounded and auditable certified mathematical operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import cvc5
from cvc5 import Kind
from flint import arb, ctx, fmpq, fmpz
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "certified-math-ticket.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_ticket(ticket: Any) -> dict[str, Any]:
    errors = sorted(VALIDATOR.iter_errors(ticket), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "$"
        raise ValueError(f"ticket schema violation at {location}: {first.message}")
    if not isinstance(ticket, dict):
        raise ValueError("ticket root must be an object")
    if ticket["operation"] == "smt-integer-feasibility":
        variables = set(ticket["inputs"]["variables"])
        for index, constraint in enumerate(ticket["inputs"]["constraints"]):
            unknown = sorted(set(constraint["coefficients"]) - variables)
            if unknown:
                raise ValueError(f"constraint {index} references unknown variable: {unknown[0]}")
            if all(int(value) == 0 for value in constraint["coefficients"].values()):
                raise ValueError(f"constraint {index} has only zero coefficients")
    return ticket


def smt_integer_feasibility(inputs: Mapping[str, Any]) -> dict[str, Any]:
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    solver.setLogic("QF_LIA")
    solver.setOption("produce-models", "true")
    integer_sort = tm.getIntegerSort()
    variables = {name: tm.mkConst(integer_sort, name) for name in inputs["variables"]}
    constraint_receipts = []
    for row in inputs["constraints"]:
        terms = []
        normalized_coefficients = {}
        for name in sorted(row["coefficients"]):
            coefficient = int(row["coefficients"][name])
            normalized_coefficients[name] = coefficient
            if coefficient == 0:
                continue
            variable = variables[name]
            term = variable if coefficient == 1 else tm.mkTerm(Kind.MULT, tm.mkInteger(coefficient), variable)
            terms.append(term)
        lhs = terms[0] if len(terms) == 1 else tm.mkTerm(Kind.ADD, *terms)
        rhs = tm.mkInteger(int(row["rhs"]))
        operator = row["operator"]
        kind = {"<=": Kind.LEQ, ">=": Kind.GEQ, "==": Kind.EQUAL}[operator]
        formula = tm.mkTerm(kind, lhs, rhs)
        solver.assertFormula(formula)
        constraint_receipts.append({
            "coefficients": normalized_coefficients,
            "operator": operator,
            "rhs": int(row["rhs"]),
        })
    result = solver.checkSat()
    if result.isSat():
        status = "SAT"
        model = {name: str(solver.getValue(variables[name])) for name in sorted(variables)}
    elif result.isUnsat():
        status = "UNSAT"
        model = None
    else:
        status = "UNKNOWN"
        model = None
    return {
        "operation": "smt-integer-feasibility",
        "solver": "cvc5",
        "solver_version": version("cvc5"),
        "logic": "QF_LIA",
        "status": status,
        "model": model,
        "variables": sorted(variables),
        "constraints": constraint_receipts,
        "deterministic_input_sha256": sha256({"variables": sorted(variables), "constraints": constraint_receipts}),
    }


def exact_integer_polynomial(inputs: Mapping[str, Any]) -> dict[str, Any]:
    coefficients = [fmpz(int(value)) for value in inputs["coefficients"]]
    x = fmpz(int(inputs["x"]))
    result = fmpz(0)
    for coefficient in coefficients:
        result = result * x + coefficient
    return {
        "operation": "exact-integer-polynomial",
        "library": "python-flint",
        "library_version": version("python-flint"),
        "coefficient_order": "highest-degree-first",
        "degree": len(coefficients) - 1,
        "x": str(x),
        "exact_value": str(result),
        "result_sha256": hashlib.sha256(str(result).encode("utf-8")).hexdigest(),
    }


def certified_rational_interval(inputs: Mapping[str, Any]) -> dict[str, Any]:
    numerator = int(inputs["numerator"])
    denominator = int(inputs["denominator"])
    precision = int(inputs.get("precision_digits", 50))
    ctx.dps = precision
    exact = fmpq(numerator, denominator)
    interval = arb(numerator) / arb(denominator)
    lower = interval.lower()
    upper = interval.upper()
    if not (lower <= interval <= upper):
        raise RuntimeError("Arb interval self-containment check failed")
    return {
        "operation": "certified-rational-interval",
        "library": "python-flint",
        "library_version": version("python-flint"),
        "precision_digits": precision,
        "exact_rational": str(exact),
        "interval": str(interval),
        "lower_bound": str(lower),
        "upper_bound": str(upper),
        "contains_exact_rational": True,
    }


OPERATIONS = {
    "smt-integer-feasibility": smt_integer_feasibility,
    "exact-integer-polynomial": exact_integer_polynomial,
    "certified-rational-interval": certified_rational_interval,
}


def execute(ticket_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    status = "COMPUTE_CERTIFIED_MATH_FAILED"
    failure = None
    result = None
    ticket = None
    try:
        if ticket_path.stat().st_size > 200_000:
            raise ValueError("ticket exceeds 200000 bytes")
        ticket = validate_ticket(json.loads(ticket_path.read_text(encoding="utf-8")))
        operation = ticket["operation"]
        result = OPERATIONS[operation](ticket["inputs"])
        status = "COMPUTE_CERTIFIED_MATH_COMPLETED"
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    receipt = {
        "schema_version": "certified-math-receipt-v1",
        "status": status,
        "task_id": ticket.get("task_id") if isinstance(ticket, Mapping) else None,
        "operation": ticket.get("operation") if isinstance(ticket, Mapping) else None,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_ms": elapsed_ms,
        "network_policy": "deny",
        "network_calls": 0,
        "model_calls": 0,
        "arbitrary_code_allowed": False,
        "arbitrary_formulas_allowed": False,
        "result": result,
        "failure": failure,
        "runtime": {
            "python": platform.python_version(),
            "cvc5": version("cvc5"),
            "python_flint": version("python-flint"),
        },
    }
    receipt["receipt_sha256"] = sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    write_json(output_dir / "certified-math-receipt.json", receipt)
    if result is not None:
        write_json(output_dir / "certified-math-result.json", result)
    if failure is not None:
        write_json(output_dir / "certified-math-failure.json", failure)
    manifest = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "certified-math-manifest.json":
            raw = path.read_bytes()
            manifest.append({"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    write_json(output_dir / "certified-math-manifest.json", {
        "schema_version": "certified-math-manifest-v1",
        "status": status,
        "files": manifest,
        "network_calls": 0,
    })
    print(json.dumps({"status": status, "duration_ms": elapsed_ms, "receipt_sha256": receipt["receipt_sha256"]}, ensure_ascii=False))
    return 0 if status == "COMPUTE_CERTIFIED_MATH_COMPLETED" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--output-dir", default="certified-math-artifacts")
    args = parser.parse_args(argv)
    return execute(Path(args.ticket), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
