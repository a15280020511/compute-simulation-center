#!/usr/bin/env python3
"""Reusable policy-optimal engine for repository-controlled dynamic capability families.

The engine owns only the common orchestration mechanics: policy/graph validation,
CP-SAT subset selection, exhaustive optimum cross-checking, NetworkX serial ordering,
fixed-adapter execution, operation contracts, per-stage hashes, and deterministic
artifacts. Family-specific signal extraction and quality interpretation remain in the
family planner so free-form objective text never becomes a routing input.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from ortools.sat.python import cp_model

from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError
from pipeline_engine import PipelineEngineError, _validate_output, load_contracts


class StructuredFamilyError(ValueError):
    """Raised when a structured dynamic family is unsafe, invalid, or not executable."""


@dataclass(frozen=True)
class FamilyDefinition:
    family: str
    declared_operation: str
    required_stage_id: str
    policy_path: Path
    graph_path: Path
    policy_schema_version: str
    graph_schema_version: str
    maximum_stages: int
    required_safety: Mapping[str, Any]


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StructuredFamilyError(f"JSON root must be an object: {path.name}")
    return value


def load_family_spec(definition: FamilyDefinition) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _load_json(definition.policy_path)
    if policy.get("schema_version") != definition.policy_schema_version:
        raise StructuredFamilyError(f"invalid {definition.family} dynamic policy schema")
    expected_policy = {
        "status": "controlled-preview",
        "family": definition.family,
        "declared_operation": definition.declared_operation,
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
        "maximum_stages": definition.maximum_stages,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise StructuredFamilyError(f"unsafe {definition.family} dynamic policy: {key}")

    allowed_operations = policy.get("allowed_operations")
    allowed_adapters = policy.get("allowed_adapters")
    if not isinstance(allowed_operations, list) or not allowed_operations or len(allowed_operations) != len(set(allowed_operations)):
        raise StructuredFamilyError("allowed_operations must be a non-empty unique array")
    if not isinstance(allowed_adapters, list) or not allowed_adapters or len(allowed_adapters) != len(set(allowed_adapters)):
        raise StructuredFamilyError("allowed_adapters must be a non-empty unique array")

    solver_policy = policy.get("solver_policy")
    if not isinstance(solver_policy, Mapping):
        raise StructuredFamilyError("solver_policy is required")
    if solver_policy.get("require_optimal_status") is not True:
        raise StructuredFamilyError("structured family planner must require OPTIMAL")
    if int(solver_policy.get("num_search_workers") or 0) != 1:
        raise StructuredFamilyError("structured family planner must use one CP-SAT worker")
    max_time = solver_policy.get("max_time_seconds")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not 0 < float(max_time) <= 10:
        raise StructuredFamilyError("solver max_time_seconds must be in (0,10]")
    max_optional = int(solver_policy.get("exhaustive_cross_check_max_optional_nodes") or 0)
    if not 1 <= max_optional <= 16:
        raise StructuredFamilyError("invalid exhaustive cross-check bound")

    selection = policy.get("selection_policy")
    rules = selection.get("stage_rules") if isinstance(selection, Mapping) else None
    if not isinstance(rules, Mapping) or not rules:
        raise StructuredFamilyError("selection_policy.stage_rules must be a non-empty object")
    for node_id, raw_rule in rules.items():
        if not isinstance(raw_rule, Mapping):
            raise StructuredFamilyError(f"invalid stage rule: {node_id}")
        if not str(raw_rule.get("operation") or ""):
            raise StructuredFamilyError(f"stage operation missing: {node_id}")
        penalty = raw_rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise StructuredFamilyError(f"invalid stage penalty: {node_id}")
        benefits = raw_rule.get("benefits")
        if not isinstance(benefits, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in benefits.values()
        ):
            raise StructuredFamilyError(f"invalid stage benefits: {node_id}")
        for name in ("eligible_all", "required_if_any", "required_if_all"):
            value = raw_rule.get(name, [])
            if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
                raise StructuredFamilyError(f"{name} must be a string array: {node_id}")

    graph_value = _load_json(definition.graph_path)
    if graph_value.get("schema_version") != definition.graph_schema_version:
        raise StructuredFamilyError(f"invalid {definition.family} capability graph schema")
    if graph_value.get("status") != "controlled-preview" or graph_value.get("family") != definition.family:
        raise StructuredFamilyError(f"{definition.family} capability graph identity mismatch")
    if graph_value.get("graph_engine") != "networkx" or graph_value.get("selection_engine") != "ortools-cp-sat":
        raise StructuredFamilyError(f"{definition.family} capability graph engine mismatch")
    safety = graph_value.get("safety")
    if not isinstance(safety, Mapping):
        raise StructuredFamilyError(f"{definition.family} graph safety is required")
    common_safety = {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "full_graph_must_be_single_serial_chain": True,
    }
    for key, expected in {**common_safety, **dict(definition.required_safety)}.items():
        if safety.get(key) != expected:
            raise StructuredFamilyError(f"unsafe {definition.family} graph policy: {key}")

    raw_nodes = graph_value.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise StructuredFamilyError(f"{definition.family} capability graph has no nodes")
    contracts = load_contracts()
    allowed_operation_set = {str(item) for item in allowed_operations}
    allowed_adapter_set = {str(item) for item in allowed_adapters}
    nodes: dict[str, dict[str, Any]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise StructuredFamilyError("capability graph node must be an object")
        node = dict(raw)
        node_id = str(node.get("id") or "")
        operation = str(node.get("operation") or "")
        adapter = str(node.get("adapter") or "")
        if not node_id or node_id in nodes:
            raise StructuredFamilyError(f"invalid or duplicate capability node: {node_id!r}")
        if operation not in allowed_operation_set or operation not in contracts:
            raise StructuredFamilyError(f"operation not contract-allowlisted: {operation}")
        if adapter not in allowed_adapter_set or adapter not in ADAPTERS:
            raise StructuredFamilyError(f"adapter not allowlisted: {adapter}")
        nodes[node_id] = node

    required = nodes.get(definition.required_stage_id)
    if not isinstance(required, Mapping):
        raise StructuredFamilyError(f"required stage is missing: {definition.required_stage_id}")
    if required.get("role") != "required-result" or required.get("operation") != definition.declared_operation:
        raise StructuredFamilyError("required result stage contract mismatch")

    precedence = graph_value.get("precedence")
    if not isinstance(precedence, list):
        raise StructuredFamilyError("capability precedence must be an array")
    edges: list[tuple[str, str]] = []
    for raw_edge in precedence:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise StructuredFamilyError("capability precedence edge must contain two node ids")
        left, right = str(raw_edge[0]), str(raw_edge[1])
        if left not in nodes or right not in nodes or left == right:
            raise StructuredFamilyError(f"invalid capability edge: {left}->{right}")
        edges.append((left, right))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise StructuredFamilyError("capability graph contains a cycle")
    full_order = list(nx.topological_sort(graph))
    if not full_order or full_order[-1] != definition.required_stage_id:
        raise StructuredFamilyError("required result stage must be final")
    if set(edges) != set(zip(full_order, full_order[1:], strict=False)):
        raise StructuredFamilyError("full capability graph must be one explicit serial chain")
    for index, node_id in enumerate(full_order):
        if graph.in_degree(node_id) != (0 if index == 0 else 1):
            raise StructuredFamilyError("capability graph branching/disconnection is forbidden")
        if graph.out_degree(node_id) != (0 if index == len(full_order) - 1 else 1):
            raise StructuredFamilyError("capability graph branching/disconnection is forbidden")

    optional_ids = full_order[:-1]
    if optional_ids != [str(item) for item in rules]:
        raise StructuredFamilyError("stage rule order must match graph precedence")
    for node_id in optional_ids:
        if str(rules[node_id]["operation"]) != str(nodes[node_id]["operation"]):
            raise StructuredFamilyError(f"stage rule operation mismatch: {node_id}")
    return policy, {"nodes": nodes, "precedence": edges, "full_order": full_order}


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    required_any = any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", []))
    required_all_names = rule.get("required_if_all", [])
    required_all = bool(required_all_names) and all(
        bool(signals.get(str(name), False)) for name in required_all_names
    )
    return required_any or required_all


def _utilities(policy: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, int]:
    result: dict[str, int] = {}
    for node_id, rule in policy["selection_policy"]["stage_rules"].items():
        score = -int(rule["penalty"])
        for signal, benefit in rule["benefits"].items():
            score += int(benefit) * int(bool(signals.get(str(signal), False)))
        result[str(node_id)] = score
    return result


def _feasible(candidate: Mapping[str, bool], policy: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    rules = policy["selection_policy"]["stage_rules"]
    if set(candidate) != {str(item) for item in rules}:
        return False
    for node_id, rule in rules.items():
        chosen = bool(candidate[str(node_id)])
        if chosen and not _eligible(rule, signals):
            return False
        if _required(rule, signals) and not chosen:
            return False
    return True


def solve_selection(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    optional_ids = list(graph["full_order"][:-1])
    rules = policy["selection_policy"]["stage_rules"]
    utilities = _utilities(policy, signals)
    model = cp_model.CpModel()
    variables = {node_id: model.new_bool_var(f"select_{node_id}") for node_id in optional_ids}
    for node_id in optional_ids:
        rule = rules[node_id]
        if not _eligible(rule, signals):
            model.add(variables[node_id] == 0)
        if _required(rule, signals):
            model.add(variables[node_id] == 1)
    model.maximize(sum(int(utilities[node_id]) * variables[node_id] for node_id in optional_ids))

    solver_policy = policy["solver_policy"]
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if solver_policy["require_optimal_status"] and status != cp_model.OPTIMAL:
        raise StructuredFamilyError(f"CP-SAT must prove OPTIMAL; observed status={status_name}")
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise StructuredFamilyError(f"CP-SAT found no feasible selection: {status_name}")
    selected = {node_id: bool(solver.value(variables[node_id])) for node_id in optional_ids}
    objective = int(round(solver.objective_value))

    max_optional = int(solver_policy["exhaustive_cross_check_max_optional_nodes"])
    if len(optional_ids) > max_optional:
        cross: dict[str, Any] = {"performed": False, "reason": "optional-node-count-exceeds-policy"}
    else:
        feasible: list[dict[str, Any]] = []
        for bits in itertools.product((False, True), repeat=len(optional_ids)):
            candidate = dict(zip(optional_ids, bits, strict=True))
            if not _feasible(candidate, policy, signals):
                continue
            score = sum(int(utilities[node_id]) * int(candidate[node_id]) for node_id in optional_ids)
            feasible.append({"selection": candidate, "objective": score})
        if not feasible:
            raise StructuredFamilyError("no feasible selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise StructuredFamilyError(
                f"CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}"
            )
        cross = {
            "performed": True,
            "optional_node_count": len(optional_ids),
            "feasible_selection_count": len(feasible),
            "best_objective": best,
            "optimal_selections": optimal,
            "passed": True,
        }
    return {
        "selected_nodes": selected,
        "solver_status": status_name,
        "objective_value": objective,
        "global_optimal_proven": status == cp_model.OPTIMAL and bool(cross.get("passed", True)),
        "utility_by_node": utilities,
        "signals": dict(signals),
        "solver_policy": {
            "num_search_workers": int(solver_policy["num_search_workers"]),
            "random_seed": int(solver_policy["random_seed"]),
            "max_time_seconds": float(solver_policy["max_time_seconds"]),
            "require_optimal_status": True,
        },
        "exhaustive_cross_check": cross,
    }


def build_plan(
    definition: FamilyDefinition,
    policy: Mapping[str, Any],
    graph: Mapping[str, Any],
    signals: Mapping[str, bool],
    features: Mapping[str, Any],
    *,
    family_reason: str,
) -> dict[str, Any]:
    optimization = solve_selection(policy, graph, signals)
    selected = optimization["selected_nodes"]
    ordered = [
        node_id
        for node_id in graph["full_order"]
        if node_id == definition.required_stage_id or bool(selected.get(node_id, False))
    ]
    if not ordered or ordered[-1] != definition.required_stage_id:
        raise StructuredFamilyError("required result stage must remain selected and final")
    if len(ordered) > definition.maximum_stages:
        raise StructuredFamilyError("selected plan exceeds maximum stages")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(ordered)
    runtime_graph.add_edges_from(zip(ordered, ordered[1:], strict=False))
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise StructuredFamilyError("selected plan contains a cycle")

    stage_map: dict[str, dict[str, Any]] = {}
    for index, stage_id in enumerate(ordered):
        node = graph["nodes"][stage_id]
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "adapter": str(node["adapter"]),
            "depends_on": [] if index == 0 else [ordered[index - 1]],
        }
    reasons = [
        family_reason,
        (
            "OR-Tools CP-SAT proved the policy-optimal feasible optional-stage subset; "
            f"status={optimization['solver_status']}, objective={optimization['objective_value']}"
        ),
    ]
    if optimization["exhaustive_cross_check"].get("performed"):
        reasons.append("independent exhaustive enumeration matched the CP-SAT optimum")
    for node_id in ordered:
        if node_id == definition.required_stage_id:
            continue
        reasons.append(
            f"{graph['nodes'][node_id]['operation']} selected with utility={optimization['utility_by_node'][node_id]}"
        )
    reasons.append(f"{definition.declared_operation} is the required result stage for this capability family")
    return {
        "id": "dynamic-auto-v1",
        "family": definition.family,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": definition.declared_operation,
        "result_stage": definition.required_stage_id,
        "stage_order": ordered,
        "stage_map": stage_map,
        "planning_features": dict(features),
        "planning_reasons": reasons,
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def execute_plan(
    ticket: Mapping[str, Any],
    plan: Mapping[str, Any],
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    contracts = load_contracts()
    initial_inputs = ticket.get("inputs")
    if not isinstance(initial_inputs, Mapping):
        raise StructuredFamilyError("ticket inputs must be an object")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    stage_elapsed: dict[str, float] = {}
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": plan["id"],
        "family": plan["family"],
        "status": "RUNNING",
        "planning_mode": plan["planning_mode"],
        "selection_engine": plan["selection_engine"],
        "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False,
        "network_used": False,
        "model_calls": 0,
        "plan_sha256": canonical_sha({
            "family": plan["family"],
            "stage_order": plan["stage_order"],
            "planning_features": plan["planning_features"],
            "optimization": plan["optimization"],
        }),
        "stages": [
            {
                "stage_id": stage_id,
                "operation": plan["stage_map"][stage_id]["operation"],
                "status": "PENDING",
            }
            for stage_id in plan["stage_order"]
        ],
    }
    write_json(output_dir / "compute-dynamic-pipeline-state.json", state)

    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = str(stage["operation"])
            if operation not in operations:
                raise StructuredFamilyError(f"handler unavailable: {operation}")
            adapter_name = str(stage["adapter"])
            if adapter_name not in ADAPTERS:
                raise StructuredFamilyError(f"adapter unavailable: {adapter_name}")
            state["stages"][index]["status"] = "RUNNING"
            write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise StructuredFamilyError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = canonical_sha(stage_inputs)
            write_json(
                output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json",
                stage_inputs,
            )
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            stage_elapsed[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise StructuredFamilyError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            try:
                _validate_output(operation, result_dict, contracts)
            except PipelineEngineError as exc:
                raise StructuredFamilyError(str(exc)) from exc
            output_sha = canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            write_json(
                output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json",
                result_dict,
            )
            receipt = {
                "stage_id": stage_id,
                "operation": operation,
                "adapter": adapter_name,
                "status": "PASS",
                "input_sha256": input_sha,
                "output_sha256": output_sha,
            }
            receipts.append(receipt)
            state["stages"][index].update(receipt)
            write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"
        for row in state["stages"]:
            if row["status"] == "RUNNING":
                row["status"] = "FAILED"
        write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
        raise

    state["status"] = "PASS"
    state["pipeline_sha256"] = canonical_sha(receipts)
    write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return stage_results, receipts, stage_elapsed


def run_structured_family(
    ticket: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
    *,
    quality_gate: Callable[[Mapping[str, dict[str, Any]], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage_results, receipts, stage_elapsed = execute_plan(ticket, plan, operations, output_dir)
    elapsed = time.perf_counter() - started
    quality = dict(quality_gate(stage_results, plan)) if quality_gate is not None else {"status": "PASS"}

    import ortools

    result_data = {
        "pipeline_id": plan["id"],
        "dynamic_family": plan["family"],
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
        "quality_gate": quality,
    }
    transfer: dict[str, Any] = {
        "schema_version": "compute-result-v1",
        "task_id": str(ticket["task_id"]),
        "status": "success",
        "operation": str(ticket["operation"]),
        "objective": ticket.get("objective"),
        "input_sha256": canonical_sha(ticket),
        "assumptions": ticket.get("assumptions", []),
        "evidence": ticket.get("evidence", []),
        "limitations": ticket.get("limitations", []),
        "results": result_data,
        "maturity_assessment": {
            "engineering_maturity": "controlled-preview",
            "evidence_maturity": "controlled-preview",
        },
        "software": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "ortools": ortools.__version__,
        },
        "execution": {
            "elapsed_seconds": round(elapsed, 6),
            "stage_elapsed_seconds": stage_elapsed,
            "network_used": False,
            "model_calls": 0,
            "reproducible": True,
            "automatic_parallel_execution": False,
        },
    }
    transfer["result_sha256"] = canonical_sha({
        "schema_version": transfer["schema_version"],
        "task_id": transfer["task_id"],
        "operation": transfer["operation"],
        "input_sha256": transfer["input_sha256"],
        "assumptions": transfer["assumptions"],
        "limitations": transfer["limitations"],
        "results": transfer["results"],
        "maturity_assessment": transfer["maturity_assessment"],
        "software": transfer["software"],
    })
    write_json(output_dir / "compute-result.json", transfer)
    write_json(
        output_dir / "compute-audit.json",
        {
            "version": 1,
            "status": "PASS",
            "task_id": transfer["task_id"],
            "operation": transfer["operation"],
            "pipeline_id": plan["id"],
            "dynamic_family": plan["family"],
            "planning_mode": plan["planning_mode"],
            "selection_engine": plan["selection_engine"],
            "graph_engine": plan["graph_engine"],
            "solver_status": plan["optimization"]["solver_status"],
            "global_optimal_proven": plan["optimization"]["global_optimal_proven"],
            "quality_gate_status": quality.get("status"),
            "input_sha256": transfer["input_sha256"],
            "result_sha256": transfer["result_sha256"],
            "elapsed_seconds": transfer["execution"]["elapsed_seconds"],
            "model_calls": 0,
            "network_used": False,
            "automatic_parallel_execution": False,
            "secret_values_included": False,
        },
    )
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic family: `{plan['family']}`\n"
        f"- Dynamic pipeline: `{plan['id']}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selection engine: `{plan['selection_engine']}`\n"
        f"- Graph engine: `{plan['graph_engine']}`\n"
        f"- Solver status: `{plan['optimization']['solver_status']}`\n"
        f"- Global optimum proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        f"- Quality gate: `{quality.get('status', 'UNKNOWN')}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        "- Execution policy: `strict-serial`\n"
        "- Automatic parallel execution: `false`\n"
        "- Model calls: `0`\n"
        "- Network used: `false`\n",
        encoding="utf-8",
    )
    return transfer
