#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for bounded robust allocation."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from jsonschema import Draft202012Validator
from ortools.sat.python import cp_model

from dynamic_family_router import resolve_dynamic_family
from dynamic_robust_allocation_adapters import install_robust_allocation_adapters
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_robust_allocation_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-robust-allocation-policy.json"
GRAPH_PATH = HERE / "dynamic-robust-allocation-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-robust-allocation-stage-contracts.json"
FAMILY = "robust-allocation"
ENTRY_OPERATION = "finance_decision_analysis"
ENTRY_MODE = "rsome_robust_allocation"
STAGE_ORDER = [
    "rsome_robust_allocation",
    "ortools_maximin_crosscheck",
    "worst_case_objective_consistency_audit",
    "allocation_feasibility_audit",
    "allocation_target_audit",
]
RESULT_STAGE_ID = "rsome_robust_allocation"


class DynamicRobustAllocationError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicRobustAllocationError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicRobustAllocationError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicRobustAllocationError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicRobustAllocationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicRobustAllocationError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise DynamicRobustAllocationError(f"{name} must be non-negative")
    return result


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_contracts() -> dict[str, dict[str, Any]]:
    value = _load_json(CONTRACT_PATH)
    if value.get("schema_version") != "compute-dynamic-robust-allocation-stage-contracts-v1" or value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicRobustAllocationError("invalid robust-allocation stage contracts")
    raw = value.get("contracts")
    if not isinstance(raw, Mapping) or list(raw) != STAGE_ORDER:
        raise DynamicRobustAllocationError("robust-allocation contracts must exactly cover stages in order")
    contracts: dict[str, dict[str, Any]] = {}
    for stage_id, schema_raw in raw.items():
        schema = dict(_mapping(schema_raw, f"contracts.{stage_id}"))
        Draft202012Validator.check_schema(schema)
        contracts[str(stage_id)] = schema
    return contracts


def _validate_output(stage_id: str, result: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]) -> None:
    errors = sorted(Draft202012Validator(dict(contracts[stage_id])).iter_errors(dict(result)), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicRobustAllocationError(f"output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-robust-allocation-policy-v1",
        "status": "controlled-preview",
        "family": FAMILY,
        "declared_operation": ENTRY_OPERATION,
        "declared_mode": ENTRY_MODE,
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
        "branching_allowed": True,
        "maximum_stages": 5,
    }
    for key, wanted in expected.items():
        if policy.get(key) != wanted:
            raise DynamicRobustAllocationError(f"unsafe robust-allocation policy: {key}")
    if policy.get("allowed_operations") != [ENTRY_OPERATION] or policy.get("allowed_entry_modes") != [ENTRY_MODE]:
        raise DynamicRobustAllocationError("robust-allocation allowlist mismatch")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    if list(rules) != STAGE_ORDER[1:]:
        raise DynamicRobustAllocationError("robust-allocation optional rule order is fixed")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_json(GRAPH_PATH)
    if graph.get("schema_version") != "compute-dynamic-robust-allocation-capability-graph-v1" or graph.get("status") != "controlled-preview" or graph.get("family") != FAMILY:
        raise DynamicRobustAllocationError("invalid robust-allocation graph")
    safety = _mapping(graph.get("safety"), "graph.safety")
    expected_safety = {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "ticket_supplied_code_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "branching_allowed": True,
        "execution_remains_strict_serial": True,
    }
    for key, wanted in expected_safety.items():
        if safety.get(key) != wanted:
            raise DynamicRobustAllocationError(f"unsafe robust-allocation graph policy: {key}")
    order = [str(item) for item in _sequence(graph.get("node_order"), "graph.node_order")]
    raw_nodes = _mapping(graph.get("nodes"), "graph.nodes")
    if order != STAGE_ORDER or set(raw_nodes) != set(STAGE_ORDER):
        raise DynamicRobustAllocationError("robust-allocation node order mismatch")
    allowed_operations = set(policy["allowed_operations"])
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(raw_nodes[stage_id], f"graph.nodes.{stage_id}"))
        if str(node.get("operation") or "") not in allowed_operations:
            raise DynamicRobustAllocationError(f"unallowlisted operation at {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicRobustAllocationError(f"unallowlisted adapter at {stage_id}")
        nodes[stage_id] = node
    precedence = [(str(edge[0]), str(edge[1])) for edge in _sequence(graph.get("precedence"), "graph.precedence")]
    expected_edges = {
        (RESULT_STAGE_ID, "ortools_maximin_crosscheck"),
        ("ortools_maximin_crosscheck", "worst_case_objective_consistency_audit"),
        (RESULT_STAGE_ID, "allocation_feasibility_audit"),
        (RESULT_STAGE_ID, "allocation_target_audit"),
    }
    if set(precedence) != expected_edges:
        raise DynamicRobustAllocationError("robust-allocation DAG mismatch")
    full_graph = nx.DiGraph()
    full_graph.add_nodes_from(order)
    full_graph.add_edges_from(precedence)
    if not nx.is_directed_acyclic_graph(full_graph) or max(dict(full_graph.out_degree()).values(), default=0) < 2:
        raise DynamicRobustAllocationError("robust-allocation graph must be a branching DAG")
    index = {stage_id: position for position, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full_graph, key=lambda item: index[item])) != order:
        raise DynamicRobustAllocationError("robust-allocation topology order mismatch")
    return {"nodes": nodes, "precedence": precedence, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicRobustAllocationError("ticket was not routed to robust-allocation family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != ENTRY_MODE:
        raise DynamicRobustAllocationError("robust-allocation entry mode mismatch")
    rows = _sequence(inputs.get("scenario_returns"), "inputs.scenario_returns")
    if not 2 <= len(rows) <= 500:
        raise DynamicRobustAllocationError("scenario_returns must contain 2 to 500 rows")
    width: int | None = None
    for i, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"inputs.scenario_returns[{i}]")
        width = len(row) if width is None else width
        if len(row) != width or width is None or not 2 <= width <= 50:
            raise DynamicRobustAllocationError("scenario_returns must be rectangular with 2 to 50 columns")
        for value in row:
            _finite(value, f"inputs.scenario_returns[{i}][]")
    names = inputs.get("asset_names")
    if names is not None:
        names_seq = [str(item) for item in _sequence(names, "inputs.asset_names")]
        if len(names_seq) != width or len(set(names_seq)) != len(names_seq) or any(not item or len(item) > 80 for item in names_seq):
            raise DynamicRobustAllocationError("asset_names must be unique and match scenario columns")

    raw_context = inputs.get("robust_allocation_context")
    context = {} if raw_context is None else dict(_mapping(raw_context, "inputs.robust_allocation_context"))
    allowed = {
        "independent_crosscheck_requested",
        "feasibility_audit_requested",
        "objective_consistency_tolerance",
        "feasibility_tolerance",
        "minimum_worst_case_return",
        "minimum_mean_return",
        "maximum_single_asset_weight",
        "allocation_target_tolerance",
    }
    unexpected = sorted(set(context) - allowed)
    if unexpected:
        raise DynamicRobustAllocationError(f"robust_allocation_context contains unsupported fields: {unexpected}")
    for name in ("independent_crosscheck_requested", "feasibility_audit_requested"):
        if name in context and not isinstance(context[name], bool):
            raise DynamicRobustAllocationError(f"robust_allocation_context.{name} must be boolean")
    for name in ("objective_consistency_tolerance", "feasibility_tolerance", "allocation_target_tolerance"):
        if name in context:
            _nonnegative(context[name], f"robust_allocation_context.{name}")
    target_names = ("minimum_worst_case_return", "minimum_mean_return", "maximum_single_asset_weight")
    target_count = 0
    for name in target_names:
        if name in context:
            value = _finite(context[name], f"robust_allocation_context.{name}")
            if name == "maximum_single_asset_weight" and not 0.0 <= value <= 1.0:
                raise DynamicRobustAllocationError("maximum_single_asset_weight must be between 0 and 1")
            target_count += 1
    if "allocation_target_tolerance" in context and target_count == 0:
        raise DynamicRobustAllocationError("allocation_target_tolerance requires at least one allocation target")

    decision_class = _decision_class(ticket)
    formal = decision_class in {"formal", "high_stakes"}
    signals = {
        "independent_crosscheck_required": bool(context.get("independent_crosscheck_requested", False)) or "objective_consistency_tolerance" in context or formal,
        "feasibility_audit_required": bool(context.get("feasibility_audit_requested", False)) or "feasibility_tolerance" in context or formal,
        "allocation_targets_available": target_count > 0,
    }
    features = {
        "decision_class": decision_class,
        "scenario_count": len(rows),
        "asset_count": int(width or 0),
        "allocation_target_count": target_count,
        **signals,
    }
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", []))


def _feasible(selection: Mapping[str, bool], rules: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    for stage_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"rules.{stage_id}")
        chosen = bool(selection[stage_id])
        if chosen and not _eligible(rule, signals):
            return False
        if _required(rule, signals) and not chosen:
            return False
        if chosen and any(not bool(selection[str(dep)]) for dep in rule.get("requires_selected", [])):
            return False
    return True


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules")
    stage_ids = list(graph["optional_ids"])
    utility: dict[str, int] = {}
    eligibility: dict[str, bool] = {}
    required: dict[str, bool] = {}
    for stage_id in stage_ids:
        rule = _mapping(rules[stage_id], f"rules.{stage_id}")
        score = -int(rule["penalty"])
        for signal_name, benefit in _mapping(rule["benefits"], "benefits").items():
            score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utility[stage_id] = score
        eligibility[stage_id] = _eligible(rule, signals)
        required[stage_id] = _required(rule, signals)
    model = cp_model.CpModel()
    variables = {stage_id: model.new_bool_var(f"select_{stage_id}") for stage_id in stage_ids}
    for stage_id in stage_ids:
        if not eligibility[stage_id]:
            model.add(variables[stage_id] == 0)
        if required[stage_id]:
            model.add(variables[stage_id] == 1)
        for dependency in rules[stage_id].get("requires_selected", []):
            model.add(variables[stage_id] <= variables[str(dependency)])
    model.maximize(sum(utility[stage_id] * variables[stage_id] for stage_id in stage_ids))
    solver = cp_model.CpSolver()
    solver_policy = _mapping(policy["solver_policy"], "solver_policy")
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise DynamicRobustAllocationError(f"selector must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected = {stage_id: bool(solver.value(variables[stage_id])) for stage_id in stage_ids}
    objective = int(round(solver.objective_value))
    feasible_rows: list[tuple[dict[str, bool], int]] = []
    for bits in itertools.product((False, True), repeat=len(stage_ids)):
        candidate = dict(zip(stage_ids, bits, strict=True))
        if _feasible(candidate, rules, signals):
            feasible_rows.append((candidate, sum(utility[s] * int(candidate[s]) for s in stage_ids)))
    best = max(value for _, value in feasible_rows)
    optima = [candidate for candidate, value in feasible_rows if value == best]
    if objective != best or selected not in optima:
        raise DynamicRobustAllocationError("CP-SAT optimum disagrees with exhaustive cross-check")
    return {
        "selected_nodes": selected,
        "solver_status": solver.StatusName(status),
        "objective_value": objective,
        "global_optimal_proven": True,
        "utility_by_node": utility,
        "eligibility_by_node": eligibility,
        "required_by_node": required,
        "signals": dict(signals),
        "exhaustive_cross_check": {
            "performed": True,
            "optional_node_count": len(stage_ids),
            "feasible_selection_count": len(feasible_rows),
            "best_objective": best,
            "optimal_selections": optima,
            "unique_optimum": len(optima) == 1,
            "passed": True,
        },
    }


def plan_dynamic_robust_allocation(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    _load_contracts()
    signals, features = _signals(ticket)
    optimization = _solve(policy, graph, signals)
    selected = {RESULT_STAGE_ID} | {stage_id for stage_id, enabled in optimization["selected_nodes"].items() if enabled}
    runtime = nx.DiGraph()
    runtime.add_nodes_from(stage_id for stage_id in graph["full_order"] if stage_id in selected)
    runtime.add_edges_from((left, right) for left, right in graph["precedence"] if left in selected and right in selected)
    stage_order = list(nx.lexicographical_topological_sort(runtime, key=lambda item: graph["index"][item]))
    expected_order = [stage_id for stage_id in graph["full_order"] if stage_id in selected]
    if stage_order != expected_order:
        raise DynamicRobustAllocationError("NetworkX order disagrees with robust-allocation policy")
    stage_map = {
        stage_id: {
            "id": stage_id,
            "operation": str(graph["nodes"][stage_id]["operation"]),
            "mode": str(graph["nodes"][stage_id].get("mode") or ""),
            "adapter": str(graph["nodes"][stage_id]["adapter"]),
            "depends_on": sorted(runtime.predecessors(stage_id), key=lambda item: graph["index"][item]),
        }
        for stage_id in stage_order
    }
    return {
        "id": "dynamic-auto-v1",
        "family": FAMILY,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": ENTRY_OPERATION,
        "declared_mode": ENTRY_MODE,
        "result_stage": RESULT_STAGE_ID,
        "required_stages": [RESULT_STAGE_ID],
        "stage_order": stage_order,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": [
            "robust-allocation family is selected only from the explicit RSOME mode and structured scenario matrix",
            "the same max-min allocation problem is deterministically compiled into OR-Tools SCIP for an independent objective cross-check",
            "only the optimal worst-case objective is compared across solvers because optimal weight vectors may be non-unique",
            "RSOME weights are independently checked for sum-to-one, non-negativity and recomputed worst-case return",
            "explicit return or concentration targets remain informative audits rather than execution-health gates",
            "OR-Tools CP-SAT must prove the policy-optimal optional branch subset and exhaustive enumeration independently verifies it",
            "NetworkX preserves the branching DAG while execution remains strict serial",
        ],
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_robust_allocation(ticket)
    contracts = _load_contracts()
    initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    elapsed: dict[str, float] = {}
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": plan["id"],
        "family": FAMILY,
        "status": "RUNNING",
        "automatic_parallel_execution": False,
        "network_used": False,
        "model_calls": 0,
        "stages": [
            {"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "mode": plan["stage_map"][stage_id]["mode"], "depends_on": plan["stage_map"][stage_id]["depends_on"], "status": "PENDING"}
            for stage_id in plan["stage_order"]
        ],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = stage["operation"]
            adapter = stage["adapter"]
            for dependency in stage["depends_on"]:
                if dependency not in results:
                    raise DynamicRobustAllocationError(f"dependency incomplete: {dependency}")
            if operation not in operations:
                raise DynamicRobustAllocationError(f"operation handler is unavailable: {operation}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter](initial_inputs, results, stage)
            except PipelineAdapterError as exc:
                raise DynamicRobustAllocationError(f"adapter failed at {stage_id}: {exc}") from exc
            derived = dict(ticket)
            derived["operation"] = operation
            derived["inputs"] = stage_inputs
            validate_operation_inputs(derived)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter()
            raw_result = operations[operation](stage_inputs)
            elapsed[stage_id] = round(time.perf_counter() - started, 6)
            result = dict(raw_result)
            _validate_output(stage_id, result, contracts)
            if stage_id in {"worst_case_objective_consistency_audit", "allocation_feasibility_audit"} and result.get("status") != "PASS":
                raise DynamicRobustAllocationError(f"internal robust-allocation audit failed: {stage_id}")
            if stage_id == "ortools_maximin_crosscheck" and (result.get("status") != "optimal" or result.get("optimality_not_guaranteed") is not False):
                raise DynamicRobustAllocationError("OR-Tools max-min cross-check must prove an optimal solution")
            output_sha = _canonical_sha(result)
            results[stage_id] = result
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", result)
            receipt = {"stage_id": stage_id, "operation": operation, "mode": stage["mode"], "adapter": adapter, "depends_on": list(stage["depends_on"]), "status": "PASS", "input_sha256": input_sha, "output_sha256": output_sha}
            receipts.append(receipt)
            state["stages"][index].update(receipt)
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"
        _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise
    state["status"] = "PASS"
    state["pipeline_sha256"] = _canonical_sha(receipts)
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return plan, results, receipts, elapsed


def run_dynamic_robust_allocation_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicRobustAllocationError("ticket is not an admitted robust-allocation request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, stage_elapsed = _execute(ticket, operations, output_dir)
    total_elapsed = time.perf_counter() - started
    import numpy as np
    import ortools
    import scipy
    validation_results = {stage_id: stage_results[stage_id] for stage_id in STAGE_ORDER[1:] if stage_id in stage_results}
    result_data: dict[str, Any] = {
        "pipeline_id": plan["id"],
        "dynamic_family": FAMILY,
        "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "stage_order": plan["stage_order"],
        "stage_dependencies": {stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]},
        "planning_features": plan["planning_features"],
        "planning_reasons": plan["planning_reasons"],
        "optimization": plan["optimization"],
        "stage_receipts": receipts,
        "stage_outputs": stage_results,
        "final_stage": RESULT_STAGE_ID,
        "final_result": stage_results[RESULT_STAGE_ID],
    }
    if validation_results:
        result_data["validation_results"] = validation_results
    software = {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "rsome": _package_version("rsome")}
    runtime = nx.DiGraph()
    runtime.add_nodes_from(plan["stage_order"])
    for stage_id in plan["stage_order"]:
        for dependency in plan["stage_map"][stage_id]["depends_on"]:
            runtime.add_edge(dependency, stage_id)
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
        "software": software,
        "execution": {"elapsed_seconds": round(total_elapsed, 6), "stage_elapsed_seconds": stage_elapsed, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": max(dict(runtime.out_degree()).values(), default=0) > 1},
    }
    transfer["result_sha256"] = _canonical_sha({key: transfer[key] for key in ["schema_version", "task_id", "operation", "input_sha256", "assumptions", "limitations", "results", "maturity_assessment", "software"]})
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(output_dir / "compute-audit.json", {
        "version": 1,
        "status": "PASS",
        "task_id": transfer["task_id"],
        "operation": transfer["operation"],
        "pipeline_id": plan["id"],
        "dynamic_family": FAMILY,
        "solver_status": plan["optimization"]["solver_status"],
        "global_optimal_proven": True,
        "result_sha256": transfer["result_sha256"],
        "network_used": False,
        "model_calls": 0,
        "automatic_parallel_execution": False,
        "graph_contains_branching": transfer["execution"]["graph_contains_branching"],
        "primary_engine": "rsome",
        "crosscheck_engine": "ortools-scip",
        "ticket_supplied_code_executed": False,
        "secret_values_included": False,
    })
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Dynamic family: `{FAMILY}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selector: `{plan['optimization']['solver_status']}`\n"
        "- Network used: `false`\n"
        "- Model calls: `0`\n",
        encoding="utf-8",
    )
    return transfer
