#!/usr/bin/env python3
"""Deterministic maximum-load, boundary, and state-isolation validation.

This suite is executed only in GitHub Actions. It does not use network access,
model calls, arbitrary code, or ticket-supplied plugins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import advanced_operations  # noqa: E402
import compute_dispatch  # noqa: E402
import compute_runner  # noqa: E402


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def timed(name: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - started
    if not finite_tree(value):
        raise AssertionError(f"{name} returned a non-finite value")
    return value, {"case": name, "status": "PASS", "elapsed_seconds": round(elapsed, 6)}


def max_monte_carlo() -> dict[str, Any]:
    variables = [
        {
            "name": f"v{index:02d}",
            "distribution": "normal",
            "mean": index / 10,
            "standard_deviation": 1 + index / 100,
            "clip_minimum": -10,
            "clip_maximum": 20,
        }
        for index in range(50)
    ]
    coefficients = {row["name"]: ((index % 7) - 3) / 5 for index, row in enumerate(variables)}
    inputs = {
        "iterations": 100_000,
        "seed": 20260804,
        "variables": variables,
        "model": {"intercept": 1.5, "coefficients": coefficients},
        "threshold": 0,
    }
    first = compute_runner.monte_carlo(inputs)
    second = compute_runner.monte_carlo(inputs)
    assert canonical_sha(first) == canonical_sha(second)
    assert first["iterations"] == 100_000
    assert len(first["sensitivity"]) == 50
    return first


def max_sensitivity() -> dict[str, Any]:
    variables = [
        {"name": f"v{index:02d}", "low": -1, "base": 0, "high": 1}
        for index in range(50)
    ]
    coefficients = {row["name"]: index + 1 for index, row in enumerate(variables)}
    result = compute_runner.sensitivity_analysis(
        {"variables": variables, "model": {"intercept": 0, "coefficients": coefficients}}
    )
    assert len(result["ranking"]) == 50
    assert result["ranking"][0]["variable"] == "v49"
    return result


def max_scenarios() -> dict[str, Any]:
    coefficients = {f"v{index:02d}": 1 for index in range(50)}
    scenarios = [
        {
            "name": f"scenario-{scenario:02d}",
            "values": {name: scenario + index / 100 for index, name in enumerate(coefficients)},
        }
        for scenario in range(50)
    ]
    result = compute_runner.scenario_compare(
        {"model": {"intercept": 0, "coefficients": coefficients}, "scenarios": scenarios}
    )
    assert len(result["ranking"]) == 50
    assert result["best_scenario"] == "scenario-49"
    return result


def max_linear_optimization() -> dict[str, Any]:
    size = 50
    objective = [1 + index / 100 for index in range(size)]
    identity = [[1 if row == column else 0 for column in range(size)] for row in range(size)]
    aggregate = [[1 for _ in range(size)] for _ in range(150)]
    result = compute_runner.constrained_optimization(
        {
            "objective": objective,
            "maximize": True,
            "variable_names": [f"x{index:02d}" for index in range(size)],
            "A_ub": identity + aggregate,
            "b_ub": [1 for _ in range(size)] + [25 for _ in range(150)],
            "bounds": [[0, 1] for _ in range(size)],
        }
    )
    assert result["success"] is True
    assert len(result["solution"]) == size
    assert math.isclose(sum(result["solution"].values()), 25.0, rel_tol=0, abs_tol=1e-7)
    return result


def max_descriptive_statistics() -> dict[str, Any]:
    data = [((index % 1000) - 500) / 10 for index in range(100_000)]
    result = compute_runner.descriptive_statistics({"data": data})
    assert result["count"] == 100_000
    return result


def max_discrete_event() -> dict[str, Any]:
    result = advanced_operations.discrete_event_simulation(
        {
            "seed": 20260804,
            "entities": 10_000,
            "arrival": {"distribution": "constant", "value": 0.05},
            "stages": [
                {
                    "name": "intake",
                    "capacity": 8,
                    "service": {"distribution": "constant", "value": 0.2},
                },
                {
                    "name": "analysis",
                    "capacity": 10,
                    "service": {"distribution": "triangular", "minimum": 0.1, "mode": 0.2, "maximum": 0.3},
                },
            ],
        }
    )
    assert result["entities_completed"] == 10_000
    assert len(result["stages"]) == 2
    return result


def max_repeated_game() -> dict[str, Any]:
    inputs = {
        "seed": 20260804,
        "rounds": 1000,
        "trials": 500,
        "red_payoffs": [[3, 0], [5, 1]],
        "blue_payoffs": [[3, 5], [0, 1]],
        "red_policy": {"type": "epsilon_best_response", "epsilon": 0.05},
        "blue_policy": {"type": "epsilon_best_response", "epsilon": 0.05},
    }
    first = advanced_operations.repeated_game(inputs)
    second = advanced_operations.repeated_game(inputs)
    assert canonical_sha(first) == canonical_sha(second)
    assert first["rounds"] * first["trials"] == 500_000
    assert math.isclose(sum(first["red_action_frequencies"]), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(first["blue_action_frequencies"]), 1.0, abs_tol=1e-12)
    return first


def max_agent_evolution() -> dict[str, Any]:
    size = 20
    matrix = [
        [1.0 + (1 if row == column else 0) for column in range(size)]
        for row in range(size)
    ]
    result = advanced_operations.agent_evolution(
        {
            "payoff_matrix": matrix,
            "initial_shares": [1 / size for _ in range(size)],
            "generations": 10_000,
            "mutation_rate": 0.01,
            "selection_strength": 1.0,
        }
    )
    assert len(result["final_shares"]) == size
    assert math.isclose(sum(result["final_shares"]), 1.0, abs_tol=1e-12)
    return result


def max_time_series() -> dict[str, Any]:
    data = [100 + 0.01 * index + 2 * math.sin(index / 20) for index in range(100_000)]
    result = advanced_operations.time_series_forecast(
        {"data": data, "horizon": 1000, "holdout": 20}
    )
    assert len(result["forecast"]) == 1000
    assert len(result["prediction_interval_95"]["lower"]) == 1000
    return result


def max_nonlinear_dynamics() -> dict[str, Any]:
    result = advanced_operations.nonlinear_dynamics(
        {
            "model": "sir",
            "initial_state": [999_000, 1000, 0],
            "duration": 365,
            "steps": 2000,
            "parameters": {"beta": 0.25, "gamma": 0.1},
        }
    )
    assert len(result["trajectory"]) == 2000
    assert len(result["final_state"]) == 3
    return result


def max_markov() -> dict[str, Any]:
    size = 50
    transition = [[1 / size for _ in range(size)] for _ in range(size)]
    inputs = {
        "transition_matrix": transition,
        "initial_distribution": [1.0] + [0.0 for _ in range(size - 1)],
        "steps": 1000,
        "state_rewards": [index - 25 for index in range(size)],
    }
    first = advanced_operations.markov_simulation(inputs)
    second = advanced_operations.markov_simulation(inputs)
    assert canonical_sha(first) == canonical_sha(second)
    assert len(first["final_distribution"]) == size
    assert math.isclose(sum(first["final_distribution"]), 1.0, abs_tol=1e-10)
    return first


def sequential_dispatch_isolation() -> dict[str, Any]:
    hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="compute-stress-") as temporary:
        root = Path(temporary)
        for index in range(25):
            ticket = {
                "task_id": f"stress-sequential-{index:03d}",
                "objective": "Verify sequential dispatch isolation and lightweight assumption governance.",
                "operation": "descriptive_statistics",
                "inputs": {"data": [index, index + 1, index + 2, index + 3, index + 4]},
                "quality_profile": {
                    "decision_class": "exploratory",
                    "publication_policy": "artifact_only",
                },
                "assumptions": [
                    {
                        "name": f"fixture-{index:03d}",
                        "value": index,
                        "basis": "Sequential isolation stress fixture.",
                        "confidence": "high",
                        "approved_by": "gpts_policy",
                    }
                ],
            }
            ticket_path = root / f"ticket-{index:03d}.json"
            output = root / f"output-{index:03d}"
            ticket_path.write_text(json.dumps(ticket, ensure_ascii=False), encoding="utf-8")
            return_code = compute_dispatch.main(
                ["--ticket", str(ticket_path), "--output-dir", str(output)]
            )
            assert return_code == 0
            result = json.loads((output / "compute-result.json").read_text(encoding="utf-8"))
            assert result["task_id"] == ticket["task_id"]
            assert result["execution"]["network_used"] is False
            assert result["execution"]["model_calls"] == 0
            assurance = result["assumption_assurance"]
            assert assurance["inline_assumption_count"] == 1
            assert assurance["lightweight_assumption_count"] == 1
            assert assurance["resolved_assumption_count"] == 1
            assert not any(row["code"] == "NO_EXPLICIT_ASSUMPTIONS" for row in assurance["issues"])
            hashes.append(assurance["resolved_snapshot_sha256"])
        assert len(set(hashes)) == 25
    return {"sequential_dispatches": 25, "unique_assumption_snapshots": len(set(hashes))}


def boundary_fail_closed() -> dict[str, Any]:
    failures: list[str] = []

    def expect(name: str, function: Callable[[], Any]) -> None:
        try:
            function()
        except compute_runner.ComputeError:
            failures.append(name)
            return
        raise AssertionError(f"boundary case did not fail closed: {name}")

    expect(
        "monte_carlo_iteration_overflow",
        lambda: compute_runner.monte_carlo(
            {
                "iterations": 100_001,
                "seed": 1,
                "variables": [{"name": "x", "distribution": "constant", "value": 1}],
                "model": {"intercept": 0, "coefficients": {"x": 1}},
            }
        ),
    )
    expect(
        "monte_carlo_variable_overflow",
        lambda: compute_runner.monte_carlo(
            {
                "iterations": 100,
                "seed": 1,
                "variables": [
                    {"name": f"x{index}", "distribution": "constant", "value": 1}
                    for index in range(51)
                ],
                "model": {
                    "intercept": 0,
                    "coefficients": {f"x{index}": 1 for index in range(51)},
                },
            }
        ),
    )
    expect(
        "scenario_duplicate_name",
        lambda: compute_runner.scenario_compare(
            {
                "model": {"intercept": 0, "coefficients": {"x": 1}},
                "scenarios": [
                    {"name": "duplicate", "values": {"x": 1}},
                    {"name": "duplicate", "values": {"x": 2}},
                ],
            }
        ),
    )
    expect(
        "repeated_game_work_overflow",
        lambda: advanced_operations.repeated_game(
            {
                "seed": 1,
                "rounds": 1001,
                "trials": 500,
                "red_payoffs": [[1]],
                "blue_payoffs": [[1]],
                "red_policy": {"type": "fixed", "probabilities": [1]},
                "blue_policy": {"type": "fixed", "probabilities": [1]},
            }
        ),
    )
    expect(
        "non_finite_scalar",
        lambda: compute_runner.descriptive_statistics({"data": [1, 2, float("inf")]}),
    )
    return {"expected_failures": failures, "failure_count": len(failures)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="compute-stress-artifacts/stress-summary.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("max_monte_carlo_100k_x_50", max_monte_carlo),
        ("max_sensitivity_50", max_sensitivity),
        ("max_scenarios_50_x_50", max_scenarios),
        ("max_linear_optimization_50_x_200", max_linear_optimization),
        ("max_descriptive_statistics_100k", max_descriptive_statistics),
        ("max_discrete_event_10k", max_discrete_event),
        ("max_repeated_game_500k", max_repeated_game),
        ("max_agent_evolution_20_x_10k", max_agent_evolution),
        ("max_time_series_100k_h1000", max_time_series),
        ("max_nonlinear_dynamics_2000", max_nonlinear_dynamics),
        ("max_markov_50_x_1000", max_markov),
        ("sequential_dispatch_isolation_25", sequential_dispatch_isolation),
        ("boundary_fail_closed", boundary_fail_closed),
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for name, function in cases:
        try:
            result, row = timed(name, function)
            row["result_sha256"] = canonical_sha(result)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "case": name,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            failures.append(
                {"case": name, "error_type": type(exc).__name__, "message": str(exc)}
            )
    summary = {
        "schema_version": "compute-stress-validation-v1",
        "status": "PASS" if not failures else "FAIL",
        "case_count": len(cases),
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "network_used": False,
        "model_calls": 0,
        "parallel_compute_execution_used": False,
        "rows": rows,
        "failures": failures,
    }
    summary["summary_sha256"] = canonical_sha(summary)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
