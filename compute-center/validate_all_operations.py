#!/usr/bin/env python3
"""Execute every registered compute operation with a bounded deterministic fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_dispatch  # noqa: E402


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fixtures() -> dict[str, dict[str, Any]]:
    confounder = [(index - 20) / 10 for index in range(40)]
    treatment = [index % 2 for index in range(40)]
    outcome = [2 * treatment[index] + 1.5 * confounder[index] for index in range(40)]
    return {
        "monte_carlo": {"iterations": 5000, "seed": 20260728, "variables": [{"name": "demand", "distribution": "triangular", "minimum": 80, "mode": 100, "maximum": 140}, {"name": "margin", "distribution": "normal", "mean": 12, "standard_deviation": 2, "clip_minimum": 0}], "model": {"intercept": 0, "coefficients": {"demand": 1, "margin": 1}}, "threshold": 100},
        "sensitivity_analysis": {"variables": [{"name": "a", "low": 1, "base": 2, "high": 5}, {"name": "b", "low": 4, "base": 5, "high": 6}], "model": {"intercept": 1, "coefficients": {"a": 3, "b": -1}}},
        "scenario_compare": {"model": {"intercept": 0, "coefficients": {"benefit": 1, "cost": -1}}, "scenarios": [{"name": "A", "values": {"benefit": 10, "cost": 8}}, {"name": "B", "values": {"benefit": 9, "cost": 3}}]},
        "constrained_optimization": {"objective": [3, 2], "maximize": True, "variable_names": ["x", "y"], "A_ub": [[1, 1], [1, 0], [0, 1]], "b_ub": [4, 2, 3], "bounds": [[0, None], [0, None]]},
        "break_even_analysis": {"fixed_cost": 1000, "unit_price": 20, "variable_cost": 12, "target_profit": 200},
        "descriptive_statistics": {"data": [1, 2, 3, 4, 5, 8, 13]},
        "discrete_event_simulation": {"seed": 7, "entities": 120, "arrival": {"distribution": "exponential", "mean": 1.2}, "stages": [{"name": "intake", "capacity": 2, "service": {"distribution": "triangular", "minimum": 0.5, "mode": 0.8, "maximum": 1.5}}, {"name": "service", "capacity": 3, "service": {"distribution": "constant", "value": 1.0}}]},
        "repeated_game": {"seed": 11, "rounds": 80, "trials": 12, "red_payoffs": [[3, 0], [5, 1]], "blue_payoffs": [[3, 5], [0, 1]], "red_policy": {"type": "epsilon_best_response", "epsilon": 0.05}, "blue_policy": {"type": "fixed", "probabilities": [0.6, 0.4]}},
        "agent_evolution": {"payoff_matrix": [[1, 3], [2, 1]], "initial_shares": [0.5, 0.5], "generations": 120, "mutation_rate": 0.01, "selection_strength": 1.0},
        "time_series_forecast": {"data": [10 + 2 * index for index in range(30)], "horizon": 5, "holdout": 6},
        "causal_screening": {"treated_pre": [10, 11, 12, 13, 14, 15], "treated_post": [20, 21, 22, 23, 24, 25], "control_pre": [8, 9, 10, 11, 12, 13], "control_post": [10, 11, 12, 13, 14, 15], "seed": 1, "bootstrap": 200},
        "nonlinear_dynamics": {"model": "logistic", "initial_state": [10], "duration": 20, "steps": 100, "parameters": {"growth_rate": 0.3, "carrying_capacity": 100}},
        "pattern_discovery": {"data": [index + (5 if index >= 20 else 0) for index in range(40)]},
        "assumption_validation": {"data": [9.8, 10.1, 10.0, 9.9, 10.2, 10.1, 9.7, 10.0, 10.2, 9.9], "expected_minimum": 9.0, "expected_maximum": 11.0, "expected_mean": 10.0, "mean_tolerance": 0.5, "expected_distribution": "normal"},
        "markov_simulation": {"transition_matrix": [[0.9, 0.1], [0.2, 0.8]], "initial_distribution": [1, 0], "steps": 80, "state_rewards": [1, -1]},
        "gis_spatial_analysis": {"mode": "geodesic_distance_matrix", "points": [{"id": "fuzhou_a", "longitude": 119.2965, "latitude": 26.0745}, {"id": "fuzhou_b", "longitude": 119.3062, "latitude": 26.0637}]},
        "bayesian_inference": {"mode": "beta_binomial", "prior_alpha": 1, "prior_beta": 1, "successes": 8, "trials": 10},
        "econometric_analysis": {"mode": "ols", "x": [[0], [1], [2], [3], [4]], "y": [1, 3, 5, 7, 9], "covariance_type": "HC1"},
        "finance_decision_analysis": {"mode": "performance_metrics", "returns": [0.01, -0.005, 0.012, 0.004, -0.002, 0.009, 0.003, -0.004], "periods_per_year": 252, "risk_free_rate": 0.02, "confidence": 0.95},
        "agent_based_simulation": {"mode": "heterogeneous_worker_choice", "agent_count": 80, "steps": 20, "seed": 20260728, "learning_rate": 0.2, "choice_sensitivity": 2.0, "switching_cost": 0.5, "preference_standard_deviation": 0.4, "reward_standard_deviation": 0.2, "options": [{"name": "zone_a", "base_reward": 20, "cost": 5, "capacity": 35, "congestion_penalty": 8}, {"name": "zone_b", "base_reward": 18, "cost": 3, "capacity": 50, "congestion_penalty": 4}]},
        "missing_data_analysis": {"mode": "missingness_profile", "columns": ["x", "y"], "data": [[1, 2], [2, None], [3, 6], [4, 8]]},
        "system_dynamics_simulation": {"mode": "stock_flow", "steps": 20, "dt": 1, "stocks": [{"name": "inventory", "initial": 10, "inflow": 2, "outflow_rate": 0.1, "capacity": 100}]},
        "crisis_early_warning": {"mode": "composite_risk_index", "indicators": [{"name": "hazard", "value": 8, "minimum": 0, "maximum": 10, "weight": 2}, {"name": "capacity", "value": 3, "minimum": 0, "maximum": 10, "weight": 1, "direction": "lower_risk"}]},
        "information_diffusion_analysis": {"mode": "sir_information_spread", "node_count": 20, "edges": [[index, (index + 1) % 20] for index in range(20)], "initial_nodes": [0], "steps": 10, "seeds": [7, 11], "transmission_probability": 0.3, "recovery_probability": 0.1},
        "causal_policy_evaluation": {"mode": "backdoor_adjustment", "treatment": treatment, "outcome": outcome, "confounders": {"baseline_risk": confounder}},
        "bayesian_network_inference": {"mode": "fixed_network_inference", "nodes": ["A", "B"], "edges": [["A", "B"]], "cpds": [{"variable": "A", "variable_card": 2, "values": [[0.6], [0.4]]}, {"variable": "B", "variable_card": 2, "values": [[0.9, 0.2], [0.1, 0.8]], "evidence": ["A"], "evidence_card": [2]}], "query_variables": ["B"], "evidence": {"A": 1}},
        "sector_model_analysis": {"mode": "nash_bimatrix_equilibria", "row_payoffs": [[3, 0], [5, 1]], "column_payoffs": [[3, 5], [0, 1]]},
        "strategic_policy_analysis": {"mode": "issue_tree_coverage", "root": "profit decline", "branches": [{"name": "revenue", "weight": 0.6, "evidence_count": 2}, {"name": "cost", "weight": 0.4, "evidence_count": 1}]},
        "transport_forecast_analysis": {"mode": "sumo_micro_simulation", "nodes": [{"id": "n0", "x": 0.0, "y": 0.0}, {"id": "n1", "x": 100.0, "y": 0.0}, {"id": "n2", "x": 200.0, "y": 0.0}], "edges": [{"id": "e0", "from": "n0", "to": "n1", "lanes": 1, "speed_mps": 13.89}, {"id": "e1", "from": "n1", "to": "n2", "lanes": 1, "speed_mps": 13.89}], "routes": [{"id": "r0", "edges": ["e0", "e1"]}], "flows": [{"id": "f0", "route": "r0", "begin": 0, "end": 120, "vehicles_per_hour": 360}], "duration_seconds": 180, "seed": 42, "timeout_seconds": 60},
        "symbolic_mathematics": {"mode": "simplify", "variables": ["x"], "expression": "(x^2-1)/(x-1)"},
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="all-compute-operation-artifacts"); args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True); ticket_root = root / "_tickets"; ticket_root.mkdir()
    cases = fixtures(); registered = set(compute_dispatch.OPERATIONS); supplied = set(cases)
    if registered != supplied:
        raise AssertionError(f"operation fixture mismatch: missing={sorted(registered-supplied)} extra={sorted(supplied-registered)}")
    rows = []; failures = []; started_all = time.perf_counter()
    required = {
        "compute-result.json", "compute-audit.json", "compute-diagnostics.json",
        "compute-preflight.json", "compute-summary.md", "compute-model-governance.json",
        "compute-assumption-assurance.json", "compute-experiment-assurance.json",
        "compute-credibility-case.json", "compute-constraint-precheck.json",
        "compute-constraint-postcheck.json"
    }
    for index, operation in enumerate(sorted(cases), 1):
        output = root / operation
        ticket = {"task_id": f"allops-{index:02d}-{operation}", "objective": f"Production execution probe for {operation}", "operation": operation, "inputs": cases[operation]}
        ticket_path = ticket_root / f"{operation}.json"; ticket_path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            return_code = compute_dispatch.main(["--ticket", str(ticket_path), "--output-dir", str(output)])
            assert return_code == 0, f"dispatcher returned {return_code}"
            transfer = load(output / "compute-result.json"); audit = load(output / "compute-audit.json"); diagnostics = load(output / "compute-diagnostics.json"); preflight = load(output / "compute-preflight.json"); manifest = load(output / "artifact-manifest.json")
            manifest_paths = {row["path"] for row in manifest["files"]}
            assert transfer["status"] == "success" and transfer["operation"] == operation
            assert transfer["execution"]["network_used"] is False and transfer["execution"]["model_calls"] == 0
            assert audit["status"] == "PASS" and diagnostics["status"] == "PASS" and preflight["execution_allowed"] is True
            assert required <= manifest_paths
            rows.append({"operation": operation, "status": "PASS", "elapsed_seconds": round(time.perf_counter() - started, 6), "input_sha256": transfer["input_sha256"], "result_sha256": transfer["result_sha256"], "package_sha256": transfer["package_sha256"], "preflight_status": preflight["status"], "assumption_assurance_status": transfer["assumption_assurance"]["status"], "experiment_assurance_status": transfer["experiment_assurance"]["status"], "credibility_case_status": transfer["credibility_case"]["status"], "result_artifact_dir": str(output)})
        except Exception as exc:
            failures.append({"operation": operation, "error_type": type(exc).__name__, "message": str(exc)})
            rows.append({"operation": operation, "status": "FAIL", "elapsed_seconds": round(time.perf_counter() - started, 6), "error_type": type(exc).__name__, "message": str(exc)})
    summary = {"schema_version": "all-compute-operations-validation-v3", "status": "PASS" if not failures else "FAIL", "operation_count_expected": len(registered), "operation_count_executed": len(rows), "passed": sum(row["status"] == "PASS" for row in rows), "failed": len(failures), "elapsed_seconds": round(time.perf_counter() - started_all, 6), "registry": sorted(registered), "rows": rows, "failures": failures}
    summary["summary_sha256"] = canonical_sha(summary)
    (root / "all-operations-validation-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
