#!/usr/bin/env python3
"""Independent deterministic computation center.

This module never calls language models, never fetches external data, and never
imports code requested by a ticket. Web GPT supplies facts and assumptions in a
strict JSON ticket; this runner validates and executes a fixed operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator

from operation_validation import validate_operation_inputs

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "compute-ticket.schema.json"
MAX_TICKET_BYTES = 2_000_000
MAX_ITERATIONS = 100_000
MAX_VARIABLES = 50
MAX_SCENARIOS = 50
MAX_DATA_POINTS = 100_000
MAX_CONSTRAINTS = 200


class ComputeError(ValueError):
    """A deterministic, user-correctable compute request error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value: str) -> None:
    raise ComputeError(f"Non-finite JSON number is forbidden: {value}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_schema() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    Draft202012Validator.check_schema(schema)
    return schema


SCHEMA = _load_schema()
VALIDATOR = Draft202012Validator(SCHEMA)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ComputeError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComputeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _model(inputs: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    model = _mapping(inputs.get("model"), "inputs.model")
    allowed = {"intercept", "coefficients"}
    unexpected = sorted(set(model) - allowed)
    if unexpected:
        raise ComputeError(f"inputs.model contains unsupported fields: {unexpected}")
    intercept = _finite_number(model.get("intercept", 0.0), "inputs.model.intercept")
    raw = _mapping(model.get("coefficients"), "inputs.model.coefficients")
    if not raw or len(raw) > MAX_VARIABLES:
        raise ComputeError(f"inputs.model.coefficients must contain 1 to {MAX_VARIABLES} variables")
    coefficients = {
        str(name): _finite_number(value, f"coefficient[{name}]")
        for name, value in raw.items()
    }
    return intercept, coefficients


def _score(values: Mapping[str, Any], intercept: float, coefficients: Mapping[str, float]) -> float:
    unknown = sorted(set(values) - set(coefficients))
    if unknown:
        raise ComputeError(f"values contain variables not present in model: {unknown}")
    missing = sorted(set(coefficients) - set(values))
    if missing:
        raise ComputeError(f"values are missing model variables: {missing}")
    total = intercept
    for name, coefficient in coefficients.items():
        total += coefficient * _finite_number(values[name], f"values[{name}]")
    if not math.isfinite(total):
        raise ComputeError("calculated score is non-finite")
    return float(total)


def _distribution_samples(rng: Any, spec: Mapping[str, Any], iterations: int, name: str) -> Any:
    allowed = {
        "name", "distribution", "value", "minimum", "mode", "maximum",
        "mean", "standard_deviation", "clip_minimum", "clip_maximum",
    }
    unexpected = sorted(set(spec) - allowed)
    if unexpected:
        raise ComputeError(f"variable {name} contains unsupported fields: {unexpected}")
    distribution = str(spec.get("distribution") or "")
    if distribution == "constant":
        value = _finite_number(spec.get("value"), f"{name}.value")
        import numpy as np

        return np.full(iterations, value, dtype=float)
    if distribution == "uniform":
        low = _finite_number(spec.get("minimum"), f"{name}.minimum")
        high = _finite_number(spec.get("maximum"), f"{name}.maximum")
        if not low < high:
            raise ComputeError(f"{name}: uniform minimum must be lower than maximum")
        return rng.uniform(low, high, iterations)
    if distribution == "triangular":
        low = _finite_number(spec.get("minimum"), f"{name}.minimum")
        mode = _finite_number(spec.get("mode"), f"{name}.mode")
        high = _finite_number(spec.get("maximum"), f"{name}.maximum")
        if not low <= mode <= high or low == high:
            raise ComputeError(f"{name}: triangular parameters must satisfy minimum <= mode <= maximum")
        return rng.triangular(low, mode, high, iterations)
    if distribution == "normal":
        import numpy as np

        mean = _finite_number(spec.get("mean"), f"{name}.mean")
        std = _finite_number(spec.get("standard_deviation"), f"{name}.standard_deviation")
        if std <= 0:
            raise ComputeError(f"{name}: standard_deviation must be positive")
        values = rng.normal(mean, std, iterations)
        clip_min = spec.get("clip_minimum")
        clip_max = spec.get("clip_maximum")
        if clip_min is not None:
            values = np.maximum(values, _finite_number(clip_min, f"{name}.clip_minimum"))
        if clip_max is not None:
            values = np.minimum(values, _finite_number(clip_max, f"{name}.clip_maximum"))
        if clip_min is not None and clip_max is not None:
            if _finite_number(clip_min, f"{name}.clip_minimum") > _finite_number(
                clip_max, f"{name}.clip_maximum"
            ):
                raise ComputeError(f"{name}: clip_minimum cannot exceed clip_maximum")
        return values
    raise ComputeError(
        f"{name}.distribution must be one of constant, uniform, triangular, normal"
    )


def monte_carlo(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    iterations = _integer(inputs.get("iterations", 10_000), "inputs.iterations", 100, MAX_ITERATIONS)
    seed = _integer(inputs.get("seed"), "inputs.seed", 0, 2**32 - 1)
    variables = _sequence(inputs.get("variables"), "inputs.variables")
    if not variables or len(variables) > MAX_VARIABLES:
        raise ComputeError(f"inputs.variables must contain 1 to {MAX_VARIABLES} entries")
    intercept, coefficients = _model(inputs)
    rng = np.random.default_rng(seed)
    samples: dict[str, Any] = {}
    for index, raw in enumerate(variables):
        spec = _mapping(raw, f"inputs.variables[{index}]")
        name = str(spec.get("name") or "")
        if not name:
            raise ComputeError(f"inputs.variables[{index}].name is required")
        if name in samples:
            raise ComputeError(f"duplicate variable name: {name}")
        samples[name] = _distribution_samples(rng, spec, iterations, name)
    if set(samples) != set(coefficients):
        raise ComputeError("model coefficients and variable names must match exactly")

    outcome = np.full(iterations, intercept, dtype=float)
    for name, coefficient in coefficients.items():
        outcome += coefficient * samples[name]
    if not np.isfinite(outcome).all():
        raise ComputeError("simulation produced non-finite values")

    percentiles = np.percentile(outcome, [5, 10, 25, 50, 75, 90, 95])
    result: dict[str, Any] = {
        "iterations": iterations,
        "seed": seed,
        "mean": float(np.mean(outcome)),
        "standard_deviation": float(np.std(outcome, ddof=0)),
        "minimum": float(np.min(outcome)),
        "maximum": float(np.max(outcome)),
        "percentiles": {
            "p05": float(percentiles[0]),
            "p10": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p90": float(percentiles[5]),
            "p95": float(percentiles[6]),
        },
    }
    if "threshold" in inputs:
        threshold = _finite_number(inputs["threshold"], "inputs.threshold")
        result["threshold"] = threshold
        result["probability_below_threshold"] = float(np.mean(outcome < threshold))

    sensitivity: list[dict[str, Any]] = []
    outcome_std = float(np.std(outcome))
    for name, values in samples.items():
        value_std = float(np.std(values))
        correlation = 0.0
        if value_std > 0 and outcome_std > 0:
            correlation = float(np.corrcoef(values, outcome)[0, 1])
        sensitivity.append(
            {
                "variable": name,
                "correlation": correlation,
                "absolute_importance": abs(correlation),
            }
        )
    sensitivity.sort(key=lambda row: row["absolute_importance"], reverse=True)
    result["sensitivity"] = sensitivity
    return result


def sensitivity_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    variables = _sequence(inputs.get("variables"), "inputs.variables")
    if not variables or len(variables) > MAX_VARIABLES:
        raise ComputeError(f"inputs.variables must contain 1 to {MAX_VARIABLES} entries")
    intercept, coefficients = _model(inputs)
    baseline: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for index, raw in enumerate(variables):
        spec = _mapping(raw, f"inputs.variables[{index}]")
        allowed = {"name", "low", "base", "high"}
        unexpected = sorted(set(spec) - allowed)
        if unexpected:
            raise ComputeError(f"sensitivity variable contains unsupported fields: {unexpected}")
        name = str(spec.get("name") or "")
        if not name or name in baseline:
            raise ComputeError("sensitivity variable names must be non-empty and unique")
        low = _finite_number(spec.get("low"), f"{name}.low")
        base_value = _finite_number(spec.get("base"), f"{name}.base")
        high = _finite_number(spec.get("high"), f"{name}.high")
        if not low <= base_value <= high or low == high:
            raise ComputeError(f"{name}: low <= base <= high is required")
        baseline[name] = base_value
        ranges[name] = (low, high)
    if set(baseline) != set(coefficients):
        raise ComputeError("model coefficients and sensitivity variable names must match exactly")
    baseline_score = _score(baseline, intercept, coefficients)
    rows: list[dict[str, Any]] = []
    for name, (low, high) in ranges.items():
        low_values = dict(baseline)
        high_values = dict(baseline)
        low_values[name] = low
        high_values[name] = high
        low_score = _score(low_values, intercept, coefficients)
        high_score = _score(high_values, intercept, coefficients)
        rows.append(
            {
                "variable": name,
                "low_score": low_score,
                "baseline_score": baseline_score,
                "high_score": high_score,
                "impact_range": abs(high_score - low_score),
            }
        )
    rows.sort(key=lambda row: row["impact_range"], reverse=True)
    return {"baseline_score": baseline_score, "ranking": rows}


def scenario_compare(inputs: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = _sequence(inputs.get("scenarios"), "inputs.scenarios")
    if not scenarios or len(scenarios) > MAX_SCENARIOS:
        raise ComputeError(f"inputs.scenarios must contain 1 to {MAX_SCENARIOS} entries")
    intercept, coefficients = _model(inputs)
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(scenarios):
        scenario = _mapping(raw, f"inputs.scenarios[{index}]")
        allowed = {"name", "values"}
        unexpected = sorted(set(scenario) - allowed)
        if unexpected:
            raise ComputeError(f"scenario contains unsupported fields: {unexpected}")
        name = str(scenario.get("name") or "")
        if not name or name in names:
            raise ComputeError("scenario names must be non-empty and unique")
        names.add(name)
        values = _mapping(scenario.get("values"), f"scenario[{name}].values")
        rows.append({"name": name, "score": _score(values, intercept, coefficients), "values": dict(values)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {"ranking": rows, "best_scenario": rows[0]["name"]}


def constrained_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import linprog

    objective = [
        _finite_number(value, f"inputs.objective[{index}]")
        for index, value in enumerate(_sequence(inputs.get("objective"), "inputs.objective"))
    ]
    if not objective or len(objective) > MAX_VARIABLES:
        raise ComputeError(f"inputs.objective must contain 1 to {MAX_VARIABLES} coefficients")
    maximize = bool(inputs.get("maximize", False))
    variable_names_raw = inputs.get("variable_names")
    if variable_names_raw is None:
        variable_names = [f"x{index + 1}" for index in range(len(objective))]
    else:
        variable_names = [str(item) for item in _sequence(variable_names_raw, "inputs.variable_names")]
        if len(variable_names) != len(objective) or len(set(variable_names)) != len(variable_names):
            raise ComputeError("variable_names must be unique and match objective length")

    bounds_raw = inputs.get("bounds", [[0, None] for _ in objective])
    bounds_seq = _sequence(bounds_raw, "inputs.bounds")
    if len(bounds_seq) != len(objective):
        raise ComputeError("bounds length must match objective length")
    bounds: list[tuple[float | None, float | None]] = []
    for index, raw in enumerate(bounds_seq):
        pair = _sequence(raw, f"inputs.bounds[{index}]")
        if len(pair) != 2:
            raise ComputeError("each bound must contain [minimum, maximum]")
        low = None if pair[0] is None else _finite_number(pair[0], f"bounds[{index}][0]")
        high = None if pair[1] is None else _finite_number(pair[1], f"bounds[{index}][1]")
        if low is not None and high is not None and low > high:
            raise ComputeError(f"bounds[{index}] minimum cannot exceed maximum")
        bounds.append((low, high))

    a_ub_raw = inputs.get("A_ub", [])
    b_ub_raw = inputs.get("b_ub", [])
    a_ub_seq = _sequence(a_ub_raw, "inputs.A_ub")
    b_ub_seq = _sequence(b_ub_raw, "inputs.b_ub")
    if len(a_ub_seq) != len(b_ub_seq) or len(a_ub_seq) > MAX_CONSTRAINTS:
        raise ComputeError("A_ub and b_ub must have equal lengths within the constraint limit")
    a_ub: list[list[float]] = []
    for row_index, raw in enumerate(a_ub_seq):
        row = [
            _finite_number(value, f"A_ub[{row_index}][{column_index}]")
            for column_index, value in enumerate(_sequence(raw, f"A_ub[{row_index}]"))
        ]
        if len(row) != len(objective):
            raise ComputeError("each A_ub row must match objective length")
        a_ub.append(row)
    b_ub = [_finite_number(value, f"b_ub[{index}]") for index, value in enumerate(b_ub_seq)]

    coefficients = np.asarray(objective, dtype=float)
    solver_objective = -coefficients if maximize else coefficients
    result = linprog(
        solver_objective,
        A_ub=np.asarray(a_ub, dtype=float) if a_ub else None,
        b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ComputeError(f"optimization failed: status={result.status}; {result.message}")
    objective_value = float(np.dot(coefficients, result.x))
    return {
        "success": True,
        "maximize": maximize,
        "objective_value": objective_value,
        "solution": {
            name: float(value)
            for name, value in zip(variable_names, result.x, strict=True)
        },
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }


def break_even_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    fixed_cost = _finite_number(inputs.get("fixed_cost"), "inputs.fixed_cost")
    unit_price = _finite_number(inputs.get("unit_price"), "inputs.unit_price")
    variable_cost = _finite_number(inputs.get("variable_cost"), "inputs.variable_cost")
    target_profit = _finite_number(inputs.get("target_profit", 0.0), "inputs.target_profit")
    if fixed_cost < 0:
        raise ComputeError("fixed_cost cannot be negative")
    if target_profit < 0:
        raise ComputeError("target_profit cannot be negative")
    contribution = unit_price - variable_cost
    if contribution <= 0:
        raise ComputeError("unit_price must exceed variable_cost")
    units = (fixed_cost + target_profit) / contribution
    return {
        "contribution_margin_per_unit": contribution,
        "break_even_units": units,
        "minimum_whole_units": math.ceil(units),
        "target_profit": target_profit,
    }


def descriptive_statistics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    data = [
        _finite_number(value, f"inputs.data[{index}]")
        for index, value in enumerate(_sequence(inputs.get("data"), "inputs.data"))
    ]
    if not data or len(data) > MAX_DATA_POINTS:
        raise ComputeError(f"inputs.data must contain 1 to {MAX_DATA_POINTS} values")
    array = np.asarray(data, dtype=float)
    percentiles = np.percentile(array, [5, 10, 25, 50, 75, 90, 95])
    return {
        "count": len(data),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation_population": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "percentiles": {
            "p05": float(percentiles[0]),
            "p10": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p90": float(percentiles[5]),
            "p95": float(percentiles[6]),
        },
    }


OPERATIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "monte_carlo": monte_carlo,
    "sensitivity_analysis": sensitivity_analysis,
    "scenario_compare": scenario_compare,
    "constrained_optimization": constrained_optimization,
    "break_even_analysis": break_even_analysis,
    "descriptive_statistics": descriptive_statistics,
}


def validate_ticket(ticket: Any) -> dict[str, Any]:
    if not isinstance(ticket, dict):
        raise ComputeError("ticket root must be an object")
    errors = sorted(VALIDATOR.iter_errors(ticket), key=lambda item: list(item.absolute_path))
    if errors:
        parts = []
        for error in errors[:20]:
            path = ".".join(str(item) for item in error.absolute_path) or "$"
            parts.append(f"{path}: {error.message}")
        raise ComputeError("; ".join(parts))
    try:
        validate_operation_inputs(ticket)
    except ValueError as exc:
        raise ComputeError(str(exc)) from exc
    return ticket


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_manifest(output_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = {"version": 1, "created_at": _utc_now(), "files": rows}
    _write_json(output_dir / "artifact-manifest.json", manifest)
    return manifest


def run_ticket(ticket: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validated = validate_ticket(ticket)
    operation = validated["operation"]
    handler = OPERATIONS[operation]
    started = time.perf_counter()
    result_data = handler(_mapping(validated["inputs"], "inputs"))
    elapsed = time.perf_counter() - started

    import numpy as np
    import scipy

    transfer = {
        "schema_version": "compute-result-v1",
        "task_id": validated["task_id"],
        "status": "success",
        "operation": operation,
        "objective": validated.get("objective"),
        "input_sha256": _sha256(validated),
        "assumptions": validated.get("assumptions", []),
        "evidence": validated.get("evidence", []),
        "limitations": validated.get("limitations", []),
        "results": result_data,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "execution": {
            "elapsed_seconds": round(elapsed, 6),
            "network_used": False,
            "model_calls": 0,
            "reproducible": operation != "monte_carlo" or "seed" in validated["inputs"],
        },
    }
    transfer["result_sha256"] = _sha256(
        {
            "schema_version": transfer["schema_version"],
            "task_id": transfer["task_id"],
            "operation": transfer["operation"],
            "input_sha256": transfer["input_sha256"],
            "assumptions": transfer["assumptions"],
            "limitations": transfer["limitations"],
            "results": transfer["results"],
            "software": transfer["software"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "compute-result.json", transfer)
    audit = {
        "version": 1,
        "created_at": _utc_now(),
        "status": "PASS",
        "task_id": validated["task_id"],
        "operation": operation,
        "input_sha256": transfer["input_sha256"],
        "result_sha256": transfer["result_sha256"],
        "elapsed_seconds": transfer["execution"]["elapsed_seconds"],
        "model_calls": 0,
        "network_used": False,
        "secret_values_included": False,
    }
    _write_json(output_dir / "compute-audit.json", audit)
    summary = (
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{validated['task_id']}`\n"
        f"- Operation: `{operation}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        f"- Model calls: `0`\n"
        f"- Network used: `false`\n"
        f"- Elapsed seconds: `{transfer['execution']['elapsed_seconds']}`\n"
    )
    (output_dir / "compute-summary.md").write_text(summary, encoding="utf-8")
    _write_manifest(output_dir)
    return transfer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one independent deterministic compute ticket.")
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--output-dir", default="compute-artifacts")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ticket_path = Path(args.ticket)
        if ticket_path.stat().st_size > MAX_TICKET_BYTES:
            raise ComputeError(f"ticket exceeds {MAX_TICKET_BYTES} bytes")
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        run_ticket(ticket, output_dir)
        print(f"Compute artifacts written to {output_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001 - always converted into structured evidence
        error = {
            "schema_version": "compute-error-v1",
            "status": "error",
            "created_at": _utc_now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "model_calls": 0,
            "network_used": False,
            "retryable": isinstance(exc, (OSError, TimeoutError)),
        }
        _write_json(output_dir / "compute-error.json", error)
        _write_manifest(output_dir)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
