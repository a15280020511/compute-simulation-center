#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for the explicit DID causal family.

The family never infers an identification strategy from objective text or correlations.
It always runs conservative DID screening first. The advanced DoWhy-backed DID stage is
eligible only after explicit structured authorization of a difference-in-differences
causal design and compatible equal-length windows.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from ortools.sat.python import cp_model

from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError
from pipeline_engine import PipelineEngineError, _validate_output, load_contracts

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-causal-did-policy.json"
GRAPH_PATH = HERE / "dynamic-causal-did-capability-graph.json"
FAMILY = "causal-did"
DECLARED_OPERATION = "causal_screening"
ENTRY_STAGE = "screening"
ADVANCED_STAGE = "did_policy_evaluation"


class DynamicCausalDidError(ValueError):
    """Raised when the explicit DID family cannot plan or execute safely."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicCausalDidError(f"JSON root must be an object: {path.name}")
    return value


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-causal-did-policy-v1",
        "status": "controlled-preview",
        "family": FAMILY,
        "declared_operation": DECLARED_OPERATION,
        "planner": "ortools-cp-sat",
        "graph_engine": "networkx",
        "network_policy": "deny",
        "model_calls": 0,
        "objective_text_routing_allowed": False,
        "structured_signals_only": True,
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_code_allowed": False,
        "automatic_parallel_execution": False,
        "cycles_allowed": False,
        "maximum_stages": 2,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise DynamicCausalDidError(f"unsafe causal DID policy: {key}")
    if policy.get("allowed_operations") != ["causal_screening", "causal_policy_evaluation"]:
        raise DynamicCausalDidError("causal DID operation allowlist mismatch")
    if policy.get("allowed_adapters") != ["causal_did_screening_inputs", "causal_did_to_policy_evaluation"]:
        raise DynamicCausalDidError("causal DID adapter allowlist mismatch")
    solver = policy.get("solver_policy")
    if not isinstance(solver, Mapping):
        raise DynamicCausalDidError("solver_policy is required")
    if solver.get("require_optimal_status") is not True or int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicCausalDidError("causal DID planner requires deterministic OPTIMAL CP-SAT")
    max_time = solver.get("max_time_seconds")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not 0 < float(max_time) <= 10:
        raise DynamicCausalDidError("invalid causal DID solver time limit")
    rules = policy.get("selection_policy", {}).get("stage_rules")
    if not isinstance(rules, Mapping) or list(rules) != [ADVANCED_STAGE]:
        raise DynamicCausalDidError("causal DID stage rules must contain only the advanced DID stage")
    rule = rules[ADVANCED_STAGE]
    if not isinstance(rule, Mapping) or rule.get("operation") != "causal_policy_evaluation":
        raise DynamicCausalDidError("advanced DID stage rule mismatch")
    if rule.get("eligible_all") != ["advanced_causal_evaluation_authorized", "aligned_did_windows"]:
        raise DynamicCausalDidError("advanced DID eligibility contract mismatch")
    if rule.get("required_if_any") != ["advanced_causal_evaluation_authorized"]:
        raise DynamicCausalDidError("advanced DID required condition mismatch")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph_value = _load_json(GRAPH_PATH)
    expected = {
        "schema_version": "compute-dynamic-causal-did-capability-graph-v1",
        "status": "controlled-preview",
        "family": FAMILY,
        "graph_engine": "networkx",
        "selection_engine": "ortools-cp-sat",
    }
    for key, value in expected.items():
        if graph_value.get(key) != value:
            raise DynamicCausalDidError(f"causal DID graph mismatch: {key}")
    safety = graph_value.get("safety")
    expected_safety = {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "full_graph_must_be_single_serial_chain": True,
        "screening_stage_required": True,
        "advanced_stage_requires_explicit_did_authorization": True,
    }
    if not isinstance(safety, Mapping):
        raise DynamicCausalDidError("causal DID graph safety is required")
    for key, value in expected_safety.items():
        if safety.get(key) != value:
            raise DynamicCausalDidError(f"unsafe causal DID graph policy: {key}")

    contracts = load_contracts()
    allowed_operations = set(policy["allowed_operations"])
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    raw_nodes = graph_value.get("nodes")
    if not isinstance(raw_nodes, list):
        raise DynamicCausalDidError("causal DID graph nodes must be an array")
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise DynamicCausalDidError("causal DID graph node must be an object")
        node = dict(raw)
        node_id = str(node.get("id") or "")
        operation = str(node.get("operation") or "")
        adapter = str(node.get("adapter") or "")
        if not node_id or node_id in nodes:
            raise DynamicCausalDidError(f"invalid or duplicate causal DID node: {node_id}")
        if operation not in allowed_operations or operation not in contracts:
            raise DynamicCausalDidError(f"causal DID operation not contract-allowlisted: {operation}")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicCausalDidError(f"causal DID adapter not allowlisted: {adapter}")
        nodes[node_id] = node
    if set(nodes) != {ENTRY_STAGE, ADVANCED_STAGE}:
        raise DynamicCausalDidError("causal DID graph must contain exactly screening and advanced DID nodes")
    if nodes[ENTRY_STAGE].get("role") != "required-entry" or nodes[ENTRY_STAGE].get("operation") != DECLARED_OPERATION:
        raise DynamicCausalDidError("causal DID screening entry contract mismatch")

    edges_raw = graph_value.get("precedence")
    if edges_raw != [[ENTRY_STAGE, ADVANCED_STAGE]]:
        raise DynamicCausalDidError("causal DID precedence must be screening -> advanced DID")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edge(ENTRY_STAGE, ADVANCED_STAGE)
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicCausalDidError("causal DID graph contains a cycle")
    order = list(nx.topological_sort(graph))
    if order != [ENTRY_STAGE, ADVANCED_STAGE]:
        raise DynamicCausalDidError("causal DID graph order mismatch")
    return {"nodes": nodes, "full_order": order}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _planning_features(ticket: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicCausalDidError("ticket was not routed to causal-did family")
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicCausalDidError("ticket inputs must be an object")
    metadata = family_runtime_metadata(ticket)
    names = ("treated_pre", "treated_post", "control_pre", "control_post")
    lengths = {name: len(inputs[name]) for name in names if isinstance(inputs.get(name), list)}
    if set(lengths) != set(names):
        raise DynamicCausalDidError("causal DID inputs are incomplete")
    aligned = len(set(lengths.values())) == 1
    advanced = metadata.get("advanced_requested") is True
    if advanced and not aligned:
        raise DynamicCausalDidError("advanced DID evaluation requires aligned windows")
    decision_class = _decision_class(ticket)
    features = {
        "causal_design": metadata.get("causal_design"),
        "advanced_causal_evaluation_authorized": advanced,
        "aligned_did_windows": aligned,
        "window_lengths": lengths,
        "decision_class": decision_class,
        "objective_text_used": False,
    }
    signals = {
        "advanced_causal_evaluation_authorized": advanced,
        "aligned_did_windows": aligned,
        "formal_decision": decision_class in {"formal", "high_stakes"},
    }
    return features, signals


def _solve(policy: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rule = policy["selection_policy"]["stage_rules"][ADVANCED_STAGE]
    utility = -int(rule["penalty"])
    for signal, benefit in rule["benefits"].items():
        utility += int(benefit) * int(bool(signals.get(str(signal), False)))
    eligible = all(bool(signals.get(str(name), False)) for name in rule["eligible_all"])
    required = any(bool(signals.get(str(name), False)) for name in rule["required_if_any"])

    model = cp_model.CpModel()
    selected = model.new_bool_var("select_did_policy_evaluation")
    if not eligible:
        model.add(selected == 0)
    if required:
        model.add(selected == 1)
    model.maximize(utility * selected)

    solver_policy = policy["solver_policy"]
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if solver_policy["require_optimal_status"] and status != cp_model.OPTIMAL:
        raise DynamicCausalDidError(f"CP-SAT must prove OPTIMAL; observed status={status_name}")
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise DynamicCausalDidError(f"causal DID CP-SAT found no feasible selection: {status_name}")
    chosen = bool(solver.value(selected))
    objective = int(round(solver.objective_value))

    feasible: list[dict[str, Any]] = []
    for candidate in (False, True):
        if candidate and not eligible:
            continue
        if required and not candidate:
            continue
        feasible.append({"selection": {ADVANCED_STAGE: candidate}, "objective": utility * int(candidate)})
    if not feasible:
        raise DynamicCausalDidError("no feasible causal DID selection during exhaustive cross-check")
    best = max(row["objective"] for row in feasible)
    optimal = [row["selection"] for row in feasible if row["objective"] == best]
    observed = {ADVANCED_STAGE: chosen}
    if objective != best or observed not in optimal:
        raise DynamicCausalDidError(
            f"causal DID CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}"
        )
    return {
        "selected_nodes": observed,
        "solver_status": status_name,
        "objective_value": objective,
        "global_optimal_proven": status == cp_model.OPTIMAL,
        "utility_by_node": {ADVANCED_STAGE: utility},
        "signals": dict(signals),
        "solver_policy": {
            "num_search_workers": int(solver_policy["num_search_workers"]),
            "random_seed": int(solver_policy["random_seed"]),
            "max_time_seconds": float(solver_policy["max_time_seconds"]),
            "require_optimal_status": True,
        },
        "exhaustive_cross_check": {
            "performed": True,
            "optional_node_count": 1,
            "feasible_selection_count": len(feasible),
            "best_objective": best,
            "optimal_selections": optimal,
            "passed": True,
        },
    }


def plan_dynamic_causal_did(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    features, signals = _planning_features(ticket)
    optimization = _solve(policy, signals)
    advanced = bool(optimization["selected_nodes"][ADVANCED_STAGE])
    order = [ENTRY_STAGE, ADVANCED_STAGE] if advanced else [ENTRY_STAGE]
    stage_map: dict[str, dict[str, Any]] = {}
    for index, stage_id in enumerate(order):
        node = graph["nodes"][stage_id]
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "adapter": str(node["adapter"]),
            "depends_on": [] if index == 0 else [order[index - 1]],
        }
    reasons = [
        "causal_screening is the mandatory conservative entry stage for the DID family",
        (
            "OR-Tools CP-SAT proved the policy-optimal feasible advanced-stage selection; "
            f"status={optimization['solver_status']}, objective={optimization['objective_value']}"
        ),
        "independent exhaustive enumeration matched the CP-SAT optimum",
    ]
    if advanced:
        reasons.append(
            "advanced difference_in_differences_refuted evaluation selected only because the ticket explicitly authorized that causal design"
        )
    else:
        reasons.append("advanced causal evaluation was not explicitly authorized; family remains screening-only")
    return {
        "id": "dynamic-auto-v1",
        "family": FAMILY,
        "maturity": "controlled-preview",
        "planning_mode": "explicit-identification-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": DECLARED_OPERATION,
        "result_stage": order[-1],
        "stage_order": order,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": reasons,
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _validate_advanced_result(result: Mapping[str, Any]) -> None:
    if result.get("mode") != "difference_in_differences_refuted":
        raise DynamicCausalDidError("advanced causal stage returned the wrong mode")
    required = {
        "effect",
        "pretrend_slope",
        "normalized_pretrend_slope",
        "parallel_trends_passed",
        "causal_claim_allowed",
        "claim_type",
        "interpretation_boundary",
    }
    missing = sorted(required - set(result))
    if missing:
        raise DynamicCausalDidError(f"advanced causal DID output missing fields: {missing}")
    parallel = result.get("parallel_trends_passed") is True
    causal_allowed = result.get("causal_claim_allowed") is True
    claim_type = str(result.get("claim_type") or "")
    if causal_allowed != parallel:
        raise DynamicCausalDidError("causal-claim gate disagrees with parallel-trends gate")
    if causal_allowed and claim_type != "causal_effect":
        raise DynamicCausalDidError("allowed causal claim must be labeled causal_effect")
    if not causal_allowed and claim_type != "association_only":
        raise DynamicCausalDidError("failed DID gate must downgrade to association_only")


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_causal_did(ticket)
    contracts = load_contracts()
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicCausalDidError("ticket inputs must be an object")
    results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    elapsed_by_stage: dict[str, float] = {}
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": plan["id"],
        "family": FAMILY,
        "status": "RUNNING",
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "network_used": False,
        "model_calls": 0,
        "plan_sha256": _canonical_sha({"stage_order": plan["stage_order"], "planning_features": plan["planning_features"], "optimization": plan["optimization"]}),
        "stages": [{"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "status": "PENDING"} for stage_id in plan["stage_order"]],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = str(stage["operation"])
            if operation not in operations:
                raise DynamicCausalDidError(f"handler unavailable: {operation}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[str(stage["adapter"])](inputs, results, stage)
            except PipelineAdapterError as exc:
                raise DynamicCausalDidError(f"adapter failed at {stage_id}: {exc}") from exc
            derived = dict(ticket)
            derived["operation"] = operation
            derived["inputs"] = stage_inputs
            validate_operation_inputs(derived)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter()
            output = operations[operation](stage_inputs)
            elapsed_by_stage[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(output, Mapping):
                raise DynamicCausalDidError(f"stage returned non-object result: {stage_id}")
            output_dict = dict(output)
            try:
                _validate_output(operation, output_dict, contracts)
            except PipelineEngineError as exc:
                raise DynamicCausalDidError(str(exc)) from exc
            if stage_id == ADVANCED_STAGE:
                _validate_advanced_result(output_dict)
            output_sha = _canonical_sha(output_dict)
            results[stage_id] = output_dict
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", output_dict)
            receipt = {"stage_id": stage_id, "operation": operation, "adapter": str(stage["adapter"]), "status": "PASS", "input_sha256": input_sha, "output_sha256": output_sha}
            receipts.append(receipt)
            state["stages"][index].update(receipt)
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"
        for row in state["stages"]:
            if row["status"] == "RUNNING":
                row["status"] = "FAILED"
        _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise
    state["status"] = "PASS"
    state["pipeline_sha256"] = _canonical_sha(receipts)
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return plan, results, receipts, elapsed_by_stage


def run_dynamic_causal_did_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicCausalDidError("ticket is not an admitted causal-did dynamic request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, stage_elapsed = _execute(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started

    import numpy as np
    import ortools
    import scipy

    result_data = {
        "pipeline_id": plan["id"],
        "dynamic_family": FAMILY,
        "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "stage_order": plan["stage_order"],
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "optimization": plan["optimization"],
        "stage_receipts": receipts,
        "stage_outputs": stage_results,
        "final_stage": plan["result_stage"],
        "final_result": stage_results[plan["result_stage"]],
    }
    transfer: dict[str, Any] = {
        "schema_version": "compute-result-v1",
        "task_id": str(ticket["task_id"]),
        "status": "success",
        "operation": str(ticket["operation"]),
        "objective": ticket.get("objective"),
        "input_sha256": _canonical_sha(ticket),
        "assumptions": ticket.get("assumptions", []),
        "evidence": ticket.get("evidence", []),
        "limitations": ticket.get("limitations", []),
        "results": result_data,
        "maturity_assessment": {"engineering_maturity": "controlled-preview", "evidence_maturity": "controlled-preview"},
        "software": {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "execution": {"elapsed_seconds": round(elapsed, 6), "stage_elapsed_seconds": stage_elapsed, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False},
    }
    transfer["result_sha256"] = _canonical_sha({key: value for key, value in transfer.items() if key != "objective"})
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(output_dir / "compute-audit.json", {
        "version": 1,
        "status": "PASS",
        "task_id": transfer["task_id"],
        "operation": transfer["operation"],
        "pipeline_id": plan["id"],
        "dynamic_family": FAMILY,
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "solver_status": plan["optimization"]["solver_status"],
        "global_optimal_proven": plan["optimization"]["global_optimal_proven"],
        "input_sha256": transfer["input_sha256"],
        "result_sha256": transfer["result_sha256"],
        "elapsed_seconds": transfer["execution"]["elapsed_seconds"],
        "model_calls": 0,
        "network_used": False,
        "automatic_parallel_execution": False,
        "secret_values_included": False,
    })
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic family: `{FAMILY}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Solver status: `{plan['optimization']['solver_status']}`\n"
        f"- Global optimum proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        "- Causal design inference from objective text: `forbidden`\n"
        "- Automatic parallel execution: `false`\n"
        "- Model calls: `0`\n"
        "- Network used: `false`\n",
        encoding="utf-8",
    )
    return transfer
