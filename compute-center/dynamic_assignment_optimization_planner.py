#!/usr/bin/env python3
"""Policy-optimal orchestration for exact-crosschecked assignment optimization."""
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
import numpy as np
from jsonschema import Draft202012Validator
from ortools.sat.python import cp_model

from dynamic_assignment_optimization_adapters import install_assignment_optimization_adapters
from dynamic_family_router import resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_assignment_optimization_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-assignment-optimization-policy.json"
GRAPH_PATH = HERE / "dynamic-assignment-optimization-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-assignment-optimization-stage-contracts.json"
FAMILY = "assignment-optimization"
ENTRY_OPERATION = "finance_decision_analysis"
ENTRY_MODE = "assignment_optimization"
RESULT_STAGE_ID = "assignment_optimization"
EXACT_AUDIT_STAGE_ID = "scipy_exact_assignment_audit"
STAGE_ORDER = [RESULT_STAGE_ID, EXACT_AUDIT_STAGE_ID, "objective_target_audit"]


class DynamicAssignmentOptimizationError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicAssignmentOptimizationError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicAssignmentOptimizationError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicAssignmentOptimizationError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicAssignmentOptimizationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicAssignmentOptimizationError(f"{name} must be finite")
    return result


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_contracts() -> dict[str, dict[str, Any]]:
    root = _load_json(CONTRACT_PATH)
    if root.get("schema_version") != "compute-dynamic-assignment-optimization-stage-contracts-v1" or root.get("status") != "controlled-preview" or root.get("family") != FAMILY:
        raise DynamicAssignmentOptimizationError("invalid assignment-optimization stage contracts")
    raw = _mapping(root.get("contracts"), "contracts")
    if list(raw) != STAGE_ORDER:
        raise DynamicAssignmentOptimizationError("assignment contracts must exactly cover stages in order")
    out: dict[str, dict[str, Any]] = {}
    for stage_id, schema_raw in raw.items():
        schema = dict(_mapping(schema_raw, f"contracts.{stage_id}")); Draft202012Validator.check_schema(schema); out[str(stage_id)] = schema
    return out


def _validate_output(stage_id: str, result: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]) -> None:
    errors = sorted(Draft202012Validator(dict(contracts[stage_id])).iter_errors(dict(result)), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]; path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicAssignmentOptimizationError(f"output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-assignment-optimization-policy-v1", "status": "controlled-preview", "family": FAMILY,
        "declared_operation": ENTRY_OPERATION, "declared_mode": ENTRY_MODE, "planner": "ortools-cp-sat", "graph_engine": "networkx",
        "network_policy": "deny", "model_calls": 0, "objective_text_routing_allowed": False, "structured_signals_only": True,
        "dynamic_operation_discovery_allowed": False, "ticket_supplied_code_allowed": False, "automatic_parallel_execution": False,
        "cycles_allowed": False, "branching_allowed": True, "maximum_stages": 3,
    }
    for key, wanted in expected.items():
        if policy.get(key) != wanted:
            raise DynamicAssignmentOptimizationError(f"unsafe assignment policy: {key}")
    if policy.get("allowed_operations") != [ENTRY_OPERATION] or policy.get("allowed_entry_modes") != [ENTRY_MODE]:
        raise DynamicAssignmentOptimizationError("assignment allowlist mismatch")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    if list(rules) != STAGE_ORDER[1:]:
        raise DynamicAssignmentOptimizationError("assignment optional rule order mismatch")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    root = _load_json(GRAPH_PATH)
    if root.get("schema_version") != "compute-dynamic-assignment-optimization-capability-graph-v1" or root.get("status") != "controlled-preview" or root.get("family") != FAMILY:
        raise DynamicAssignmentOptimizationError("invalid assignment capability graph")
    safety = _mapping(root.get("safety"), "graph.safety")
    expected_safety = {"dynamic_operation_discovery_allowed": False, "ticket_supplied_nodes_allowed": False, "ticket_supplied_edges_allowed": False, "ticket_supplied_code_allowed": False, "cycles_allowed": False, "automatic_parallel_execution": False, "branching_allowed": True, "execution_remains_strict_serial": True}
    for key, wanted in expected_safety.items():
        if safety.get(key) != wanted:
            raise DynamicAssignmentOptimizationError(f"unsafe assignment graph policy: {key}")
    order = [str(item) for item in _sequence(root.get("node_order"), "graph.node_order")]
    raw_nodes = _mapping(root.get("nodes"), "graph.nodes")
    if order != STAGE_ORDER or set(raw_nodes) != set(STAGE_ORDER):
        raise DynamicAssignmentOptimizationError("assignment node order mismatch")
    allowed_operations = set(policy["allowed_operations"]); allowed_adapters = set(policy["allowed_adapters"]); nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(raw_nodes[stage_id], f"graph.nodes.{stage_id}"))
        if str(node.get("operation") or "") not in allowed_operations: raise DynamicAssignmentOptimizationError(f"unallowlisted operation at {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS: raise DynamicAssignmentOptimizationError(f"unallowlisted adapter at {stage_id}")
        nodes[stage_id] = node
    precedence = [(str(edge[0]), str(edge[1])) for edge in _sequence(root.get("precedence"), "graph.precedence")]
    expected_edges = {(RESULT_STAGE_ID, stage_id) for stage_id in STAGE_ORDER[1:]}
    if set(precedence) != expected_edges: raise DynamicAssignmentOptimizationError("assignment DAG mismatch")
    full = nx.DiGraph(); full.add_nodes_from(order); full.add_edges_from(precedence)
    if not nx.is_directed_acyclic_graph(full) or max(dict(full.out_degree()).values(), default=0) < 2: raise DynamicAssignmentOptimizationError("assignment graph must be a branching DAG")
    index = {stage_id: position for position, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full, key=lambda item: index[item])) != order: raise DynamicAssignmentOptimizationError("assignment topology order mismatch")
    return {"nodes": nodes, "precedence": precedence, "full_order": order, "optional_ids": order[1:], "index": index}


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY: raise DynamicAssignmentOptimizationError("ticket was not routed to assignment-optimization family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != ENTRY_MODE: raise DynamicAssignmentOptimizationError("assignment entry mode mismatch")
    workers = [str(item) for item in _sequence(inputs.get("workers"), "inputs.workers")]; tasks = [str(item) for item in _sequence(inputs.get("tasks"), "inputs.tasks")]
    if not 1 <= len(workers) <= 100 or not 1 <= len(tasks) <= 100: raise DynamicAssignmentOptimizationError("workers/tasks must contain 1 to 100 entries")
    if len(workers) < len(tasks): raise DynamicAssignmentOptimizationError("require_all_tasks=true requires workers >= tasks")
    if any(not item for item in workers + tasks) or len(set(workers)) != len(workers) or len(set(tasks)) != len(tasks): raise DynamicAssignmentOptimizationError("worker/task names must be non-empty and unique")
    if inputs.get("require_all_tasks", True) is not True: raise DynamicAssignmentOptimizationError("assignment dynamic family v1 requires require_all_tasks=true")
    raw_costs = _sequence(inputs.get("costs"), "inputs.costs")
    if len(raw_costs) != len(workers): raise DynamicAssignmentOptimizationError("costs must have one row per worker")
    for i, raw_row in enumerate(raw_costs):
        row = _sequence(raw_row, f"inputs.costs[{i}]")
        if len(row) != len(tasks): raise DynamicAssignmentOptimizationError("costs must be a worker-by-task matrix")
        for j, value in enumerate(row): _finite(value, f"inputs.costs[{i}][{j}]")
    maximize = inputs.get("maximize", False)
    if not isinstance(maximize, bool): raise DynamicAssignmentOptimizationError("maximize must be boolean")
    raw_context = inputs.get("assignment_optimization_context"); context = {} if raw_context is None else dict(_mapping(raw_context, "inputs.assignment_optimization_context"))
    allowed = {"exact_consistency_tolerance", "maximum_objective_value", "minimum_objective_value", "objective_target_tolerance"}
    unexpected = sorted(set(context) - allowed)
    if unexpected: raise DynamicAssignmentOptimizationError(f"assignment_optimization_context contains unsupported fields: {unexpected}")
    if "exact_consistency_tolerance" in context:
        tolerance = _finite(context["exact_consistency_tolerance"], "assignment_optimization_context.exact_consistency_tolerance")
        if not 0 <= tolerance <= 1e-6: raise DynamicAssignmentOptimizationError("exact_consistency_tolerance must be between 0 and 1e-6")
    has_max = "maximum_objective_value" in context; has_min = "minimum_objective_value" in context
    if has_max and has_min: raise DynamicAssignmentOptimizationError("supply only one objective target")
    if maximize and has_max: raise DynamicAssignmentOptimizationError("maximize=true requires minimum_objective_value, not maximum_objective_value")
    if not maximize and has_min: raise DynamicAssignmentOptimizationError("maximize=false requires maximum_objective_value, not minimum_objective_value")
    if "objective_target_tolerance" in context and not (has_max or has_min): raise DynamicAssignmentOptimizationError("objective_target_tolerance requires an objective target")
    if has_max: _finite(context["maximum_objective_value"], "assignment_optimization_context.maximum_objective_value")
    if has_min: _finite(context["minimum_objective_value"], "assignment_optimization_context.minimum_objective_value")
    if "objective_target_tolerance" in context and _finite(context["objective_target_tolerance"], "assignment_optimization_context.objective_target_tolerance") < 0: raise DynamicAssignmentOptimizationError("objective_target_tolerance must be non-negative")
    signals = {"exact_assignment_consistency_required": True, "objective_target_available": has_max or has_min}
    features = {"decision_class": str(_mapping(ticket.get("quality_profile", {}), "quality_profile").get("decision_class") or "exploratory"), "worker_count": len(workers), "task_count": len(tasks), "maximize": maximize, "require_all_tasks": True, **signals}
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", []))


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules"); ids = list(graph["optional_ids"])
    utility: dict[str, int] = {}; eligible: dict[str, bool] = {}; required: dict[str, bool] = {}
    for stage_id in ids:
        rule = _mapping(rules[stage_id], f"rules.{stage_id}"); score = -int(rule["penalty"])
        for signal_name, benefit in _mapping(rule["benefits"], "benefits").items(): score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utility[stage_id] = score; eligible[stage_id] = _eligible(rule, signals); required[stage_id] = _required(rule, signals)
    model = cp_model.CpModel(); variables = {stage_id: model.new_bool_var(f"select_{stage_id}") for stage_id in ids}
    for stage_id in ids:
        if not eligible[stage_id]: model.add(variables[stage_id] == 0)
        if required[stage_id]: model.add(variables[stage_id] == 1)
    model.maximize(sum(utility[stage_id] * variables[stage_id] for stage_id in ids))
    solver = cp_model.CpSolver(); sp = _mapping(policy["solver_policy"], "solver_policy"); solver.parameters.num_search_workers = int(sp["num_search_workers"]); solver.parameters.random_seed = int(sp["random_seed"]); solver.parameters.max_time_in_seconds = float(sp["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL: raise DynamicAssignmentOptimizationError(f"selector must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected = {stage_id: bool(solver.value(variables[stage_id])) for stage_id in ids}; objective = int(round(solver.objective_value))
    feasible: list[tuple[dict[str, bool], int]] = []
    for bits in itertools.product((False, True), repeat=len(ids)):
        candidate = dict(zip(ids, bits, strict=True)); okay = True
        for stage_id in ids:
            if candidate[stage_id] and not eligible[stage_id]: okay = False
            if required[stage_id] and not candidate[stage_id]: okay = False
        if okay: feasible.append((candidate, sum(utility[s] * int(candidate[s]) for s in ids)))
    best = max(value for _, value in feasible); optima = [candidate for candidate, value in feasible if value == best]
    if objective != best or selected not in optima: raise DynamicAssignmentOptimizationError("CP-SAT optimum disagrees with exhaustive cross-check")
    return {"selected_nodes": selected, "solver_status": solver.StatusName(status), "objective_value": objective, "global_optimal_proven": True, "utility_by_node": utility, "eligibility_by_node": eligible, "required_by_node": required, "signals": dict(signals), "exhaustive_cross_check": {"performed": True, "optional_node_count": len(ids), "feasible_selection_count": len(feasible), "best_objective": best, "optimal_selections": optima, "unique_optimum": len(optima) == 1, "passed": True}}


def plan_dynamic_assignment_optimization(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy(); graph = _load_graph(policy); _load_contracts(); signals, features = _signals(ticket); optimization = _solve(policy, graph, signals)
    selected = {RESULT_STAGE_ID} | {stage_id for stage_id, enabled in optimization["selected_nodes"].items() if enabled}; runtime = nx.DiGraph(); runtime.add_nodes_from(stage_id for stage_id in graph["full_order"] if stage_id in selected); runtime.add_edges_from((a, b) for a, b in graph["precedence"] if a in selected and b in selected)
    order = list(nx.lexicographical_topological_sort(runtime, key=lambda item: graph["index"][item])); expected = [stage_id for stage_id in graph["full_order"] if stage_id in selected]
    if order != expected: raise DynamicAssignmentOptimizationError("NetworkX order disagrees with assignment policy")
    stage_map = {stage_id: {"id": stage_id, "operation": str(graph["nodes"][stage_id]["operation"]), "mode": str(graph["nodes"][stage_id].get("mode") or ""), "adapter": str(graph["nodes"][stage_id]["adapter"]), "depends_on": sorted(runtime.predecessors(stage_id), key=lambda item: graph["index"][item])} for stage_id in order}
    return {"id": "dynamic-auto-v1", "family": FAMILY, "maturity": "controlled-preview", "planning_mode": "structured-signal-policy-optimal-family", "selection_engine": "ortools-cp-sat", "graph_engine": "networkx", "objective_text_used": False, "declared_operation": ENTRY_OPERATION, "declared_mode": ENTRY_MODE, "result_stage": RESULT_STAGE_ID, "required_stages": [RESULT_STAGE_ID, EXACT_AUDIT_STAGE_ID], "stage_order": order, "stage_map": stage_map, "planning_features": features, "planning_reasons": ["assignment family is selected only from explicit assignment_optimization inputs", "OR-Tools SCIP solves the primary all-tasks assignment model", "SciPy linear_sum_assignment independently solves the same rectangular LSAP and original costs are used to reconstruct the primary objective", "v1 requires require_all_tasks=true and workers >= tasks so both backends share an identical feasible-set contract", "an explicit objective target is optional decision information and cannot turn a valid computation into an execution failure", "OR-Tools CP-SAT proves the policy-optimal audit subset and exhaustive enumeration independently verifies it", "NetworkX preserves a branching DAG while execution remains strict serial"], "optimization": optimization, "network_policy": "deny", "automatic_parallel_execution": False, "model_calls": 0}


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_assignment_optimization(ticket); contracts = _load_contracts(); initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs"); results: dict[str, dict[str, Any]] = {}; receipts: list[dict[str, Any]] = []; elapsed: dict[str, float] = {}
    state: dict[str, Any] = {"schema_version": "compute-dynamic-pipeline-state-v2", "pipeline_id": plan["id"], "family": FAMILY, "status": "RUNNING", "automatic_parallel_execution": False, "network_used": False, "model_calls": 0, "stages": [{"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "mode": plan["stage_map"][stage_id]["mode"], "depends_on": plan["stage_map"][stage_id]["depends_on"], "status": "PENDING"} for stage_id in plan["stage_order"]]}; _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]; operation = stage["operation"]; adapter = stage["adapter"]
            for dependency in stage["depends_on"]:
                if dependency not in results: raise DynamicAssignmentOptimizationError(f"dependency incomplete: {dependency}")
            if operation not in operations: raise DynamicAssignmentOptimizationError(f"operation handler is unavailable: {operation}")
            state["stages"][index]["status"] = "RUNNING"; _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try: stage_inputs = ADAPTERS[adapter](initial_inputs, results, stage)
            except PipelineAdapterError as exc: raise DynamicAssignmentOptimizationError(f"adapter failed at {stage_id}: {exc}") from exc
            derived = dict(ticket); derived["operation"] = operation; derived["inputs"] = stage_inputs; validate_operation_inputs(derived); input_sha = _canonical_sha(stage_inputs); _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter(); raw_result = operations[operation](stage_inputs); elapsed[stage_id] = round(time.perf_counter() - started, 6); result = dict(raw_result); _validate_output(stage_id, result, contracts)
            if stage_id == EXACT_AUDIT_STAGE_ID and result.get("status") != "PASS": raise DynamicAssignmentOptimizationError("SciPy exact assignment consistency audit failed")
            output_sha = _canonical_sha(result); results[stage_id] = result; _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", result); receipt = {"stage_id": stage_id, "operation": operation, "mode": stage["mode"], "adapter": adapter, "depends_on": list(stage["depends_on"]), "status": "PASS", "input_sha256": input_sha, "output_sha256": output_sha}; receipts.append(receipt); state["stages"][index].update(receipt); _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"; _write_json(output_dir / "compute-dynamic-pipeline-state.json", state); raise
    state["status"] = "PASS"; state["pipeline_sha256"] = _canonical_sha(receipts); _write_json(output_dir / "compute-dynamic-pipeline-state.json", state); return plan, results, receipts, elapsed


def run_dynamic_assignment_optimization_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY: raise DynamicAssignmentOptimizationError("ticket is not an admitted assignment-optimization request")
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); plan, stage_results, receipts, stage_elapsed = _execute(ticket, operations, output_dir); total_elapsed = time.perf_counter() - started
    import ortools, scipy
    validation_results = {stage_id: stage_results[stage_id] for stage_id in STAGE_ORDER[1:] if stage_id in stage_results}; result_data: dict[str, Any] = {"pipeline_id": plan["id"], "dynamic_family": FAMILY, "pipeline_maturity": plan["maturity"], "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"], "automatic_parallel_execution": False, "stage_order": plan["stage_order"], "stage_dependencies": {stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]}, "planning_features": plan["planning_features"], "planning_reasons": plan["planning_reasons"], "optimization": plan["optimization"], "stage_receipts": receipts, "stage_outputs": stage_results, "final_stage": RESULT_STAGE_ID, "final_result": stage_results[RESULT_STAGE_ID], "validation_results": validation_results}
    software = {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__}; runtime = nx.DiGraph(); runtime.add_nodes_from(plan["stage_order"])
    for stage_id in plan["stage_order"]:
        for dependency in plan["stage_map"][stage_id]["depends_on"]: runtime.add_edge(dependency, stage_id)
    transfer: dict[str, Any] = {"schema_version": "compute-result-v1", "task_id": str(ticket["task_id"]), "status": "success", "operation": str(ticket["operation"]), "objective": ticket.get("objective"), "input_sha256": _canonical_sha(ticket), "assumptions": ticket.get("assumptions", []), "evidence": ticket.get("evidence", []), "limitations": ticket.get("limitations", []), "results": result_data, "maturity_assessment": {"engineering_maturity": "controlled-preview", "evidence_maturity": "controlled-preview"}, "software": software, "execution": {"elapsed_seconds": round(total_elapsed, 6), "stage_elapsed_seconds": stage_elapsed, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": max(dict(runtime.out_degree()).values(), default=0) > 1}}
    transfer["result_sha256"] = _canonical_sha({key: transfer[key] for key in ["schema_version", "task_id", "operation", "input_sha256", "assumptions", "limitations", "results", "maturity_assessment", "software"]}); _write_json(output_dir / "compute-result.json", transfer); _write_json(output_dir / "compute-audit.json", {"version": 1, "status": "PASS", "task_id": transfer["task_id"], "operation": transfer["operation"], "pipeline_id": plan["id"], "dynamic_family": FAMILY, "solver_status": plan["optimization"]["solver_status"], "global_optimal_proven": True, "result_sha256": transfer["result_sha256"], "network_used": False, "model_calls": 0, "automatic_parallel_execution": False, "graph_contains_branching": transfer["execution"]["graph_contains_branching"], "primary_engine": "ortools-scip-assignment", "independent_validation_engine": "scipy-linear-sum-assignment", "ticket_supplied_code_executed": False, "secret_values_included": False})
    (output_dir / "compute-summary.md").write_text("# COMPUTE_COMPLETED\n\n" f"- Task ID: `{transfer['task_id']}`\n" f"- Dynamic family: `{FAMILY}`\n" f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n" f"- Selector: `{plan['optimization']['solver_status']}`\n" "- Primary engine: `OR-Tools SCIP assignment`\n" "- Independent exact validation: `SciPy linear_sum_assignment`\n" "- Network used: `false`\n" "- Model calls: `0`\n", encoding="utf-8"); return transfer
