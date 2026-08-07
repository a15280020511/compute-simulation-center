#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for the time-series capability family."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from ortools.sat.python import cp_model

from dynamic_family_router import resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError
from pipeline_engine import PipelineEngineError, _validate_output, load_contracts

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-time-series-policy.json"
GRAPH_PATH = HERE / "dynamic-time-series-capability-graph.json"
FAMILY = "time-series"
DECLARED_OPERATION = "time_series_forecast"
REQUIRED_STAGE_ID = "forecast"


class DynamicTimeSeriesError(ValueError):
    """Raised when a time-series family plan cannot be generated or executed safely."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicTimeSeriesError(f"JSON root must be an object: {path.name}")
    return value


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    if policy.get("schema_version") != "compute-dynamic-time-series-policy-v1":
        raise DynamicTimeSeriesError("invalid time-series dynamic policy schema")
    expected = {
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
        "maximum_stages": 4,
    }
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise DynamicTimeSeriesError(f"unsafe time-series dynamic policy: {key}")
    allowed_operations = policy.get("allowed_operations")
    allowed_adapters = policy.get("allowed_adapters")
    if not isinstance(allowed_operations, list) or len(allowed_operations) != len(set(allowed_operations)):
        raise DynamicTimeSeriesError("allowed_operations must be a unique array")
    if not isinstance(allowed_adapters, list) or len(allowed_adapters) != len(set(allowed_adapters)):
        raise DynamicTimeSeriesError("allowed_adapters must be a unique array")

    solver_policy = policy.get("solver_policy")
    if not isinstance(solver_policy, Mapping):
        raise DynamicTimeSeriesError("solver_policy is required")
    if solver_policy.get("require_optimal_status") is not True:
        raise DynamicTimeSeriesError("time-series dynamic planner must require OPTIMAL")
    if int(solver_policy.get("num_search_workers") or 0) != 1:
        raise DynamicTimeSeriesError("time-series dynamic planner must use one CP-SAT worker")
    max_time = solver_policy.get("max_time_seconds")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not 0 < float(max_time) <= 10:
        raise DynamicTimeSeriesError("solver max_time_seconds must be in (0,10]")
    max_optional = int(solver_policy.get("exhaustive_cross_check_max_optional_nodes") or 0)
    if not 1 <= max_optional <= 16:
        raise DynamicTimeSeriesError("invalid exhaustive cross-check bound")

    selection = policy.get("selection_policy")
    rules = selection.get("stage_rules") if isinstance(selection, Mapping) else None
    if not isinstance(rules, Mapping) or not rules:
        raise DynamicTimeSeriesError("selection_policy.stage_rules must be a non-empty object")
    for node_id, raw_rule in rules.items():
        if not isinstance(raw_rule, Mapping):
            raise DynamicTimeSeriesError(f"invalid stage rule: {node_id}")
        if not str(raw_rule.get("operation") or ""):
            raise DynamicTimeSeriesError(f"stage operation missing: {node_id}")
        penalty = raw_rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicTimeSeriesError(f"invalid stage penalty: {node_id}")
        benefits = raw_rule.get("benefits")
        if not isinstance(benefits, Mapping) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in benefits.values()
        ):
            raise DynamicTimeSeriesError(f"invalid stage benefits: {node_id}")
        for name in ("eligible_all", "required_if_any", "required_if_all"):
            value = raw_rule.get(name, [])
            if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
                raise DynamicTimeSeriesError(f"{name} must be a string array: {node_id}")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(GRAPH_PATH)
    if value.get("schema_version") != "compute-dynamic-time-series-capability-graph-v1":
        raise DynamicTimeSeriesError("invalid time-series capability graph schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicTimeSeriesError("time-series capability graph identity mismatch")
    if value.get("graph_engine") != "networkx" or value.get("selection_engine") != "ortools-cp-sat":
        raise DynamicTimeSeriesError("time-series capability graph engine mismatch")
    safety = value.get("safety")
    expected_safety = {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "full_graph_must_be_single_serial_chain": True,
        "forecast_stage_required": True,
    }
    if not isinstance(safety, Mapping):
        raise DynamicTimeSeriesError("time-series graph safety is required")
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise DynamicTimeSeriesError(f"unsafe time-series graph policy: {key}")

    raw_nodes = value.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise DynamicTimeSeriesError("time-series capability graph has no nodes")
    contracts = load_contracts()
    allowed_operations = {str(item) for item in policy["allowed_operations"]}
    allowed_adapters = {str(item) for item in policy["allowed_adapters"]}
    nodes: dict[str, dict[str, Any]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise DynamicTimeSeriesError("time-series graph node must be an object")
        node = dict(raw)
        node_id = str(node.get("id") or "")
        operation = str(node.get("operation") or "")
        adapter = str(node.get("adapter") or "")
        if not node_id or node_id in nodes:
            raise DynamicTimeSeriesError(f"invalid or duplicate time-series node: {node_id!r}")
        if operation not in allowed_operations or operation not in contracts:
            raise DynamicTimeSeriesError(f"operation not contract-allowlisted: {operation}")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicTimeSeriesError(f"adapter not allowlisted: {adapter}")
        nodes[node_id] = node
    if REQUIRED_STAGE_ID not in nodes:
        raise DynamicTimeSeriesError("forecast stage is missing")
    forecast = nodes[REQUIRED_STAGE_ID]
    if forecast.get("role") != "required-result" or forecast.get("operation") != DECLARED_OPERATION:
        raise DynamicTimeSeriesError("forecast stage contract mismatch")

    precedence = value.get("precedence")
    if not isinstance(precedence, list):
        raise DynamicTimeSeriesError("time-series precedence must be an array")
    edges: list[tuple[str, str]] = []
    for edge in precedence:
        if not isinstance(edge, list) or len(edge) != 2:
            raise DynamicTimeSeriesError("time-series precedence edge must contain two node ids")
        left, right = str(edge[0]), str(edge[1])
        if left not in nodes or right not in nodes or left == right:
            raise DynamicTimeSeriesError(f"invalid time-series edge: {left}->{right}")
        edges.append((left, right))
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise DynamicTimeSeriesError("time-series capability graph contains a cycle")
    full_order = list(nx.topological_sort(graph))
    if not full_order or full_order[-1] != REQUIRED_STAGE_ID:
        raise DynamicTimeSeriesError("forecast must be the final time-series capability node")
    if set(edges) != set(zip(full_order, full_order[1:], strict=False)):
        raise DynamicTimeSeriesError("time-series full graph must be one explicit serial chain")
    for index, node_id in enumerate(full_order):
        if graph.in_degree(node_id) != (0 if index == 0 else 1):
            raise DynamicTimeSeriesError("time-series graph branching/disconnection is forbidden")
        if graph.out_degree(node_id) != (0 if index == len(full_order) - 1 else 1):
            raise DynamicTimeSeriesError("time-series graph branching/disconnection is forbidden")

    rules = policy["selection_policy"]["stage_rules"]
    optional_ids = full_order[:-1]
    if optional_ids != [str(item) for item in rules]:
        raise DynamicTimeSeriesError("time-series stage rule order must match graph precedence")
    for node_id in optional_ids:
        if str(rules[node_id]["operation"]) != str(nodes[node_id]["operation"]):
            raise DynamicTimeSeriesError(f"time-series stage rule mismatch: {node_id}")
    return {"nodes": nodes, "precedence": edges, "full_order": full_order}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicTimeSeriesError("ticket was not routed to time-series family")
    inputs = ticket.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DynamicTimeSeriesError("ticket inputs must be an object")
    raw_data = inputs.get("data")
    if isinstance(raw_data, (str, bytes)) or not isinstance(raw_data, list):
        raise DynamicTimeSeriesError("inputs.data must be an array")
    data: list[float] = []
    for index, raw in enumerate(raw_data):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DynamicTimeSeriesError(f"inputs.data[{index}] must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise DynamicTimeSeriesError(f"inputs.data[{index}] must be finite")
        data.append(value)
    if len(data) < 5:
        raise DynamicTimeSeriesError("time-series family requires at least five observations")
    horizon_raw = inputs.get("horizon", 1)
    if isinstance(horizon_raw, bool) or not isinstance(horizon_raw, int) or horizon_raw < 1:
        raise DynamicTimeSeriesError("inputs.horizon must be a positive integer")
    context = inputs.get("dynamic_context")
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise DynamicTimeSeriesError("inputs.dynamic_context must be an object")
    diagnostics_requested = context.get("time_series_diagnostics") is True
    assumption_fields = (
        "expected_minimum",
        "expected_maximum",
        "expected_mean",
        "expected_distribution",
    )
    assumptions_supplied = any(name in inputs for name in assumption_fields)
    decision_class = _decision_class(ticket)
    selection = policy["selection_policy"]
    signals = {
        "forecast_input_valid": True,
        "pattern_eligible": len(data) >= int(selection["pattern_min_observations"]),
        "diagnostics_requested": diagnostics_requested,
        "long_horizon": horizon_raw >= int(selection["long_horizon_threshold"]),
        "assumptions_supplied": assumptions_supplied,
        "formal_decision": decision_class in {"formal", "high_stakes"},
    }
    features = {
        "observation_count": len(data),
        "forecast_horizon": horizon_raw,
        "diagnostics_requested": diagnostics_requested,
        "assumptions_supplied": assumptions_supplied,
        "decision_class": decision_class,
    }
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", [])) or all(
        bool(signals.get(str(name), False)) for name in rule.get("required_if_all", [])
    ) if rule.get("required_if_all") else any(
        bool(signals.get(str(name), False)) for name in rule.get("required_if_any", [])
    )


def _utilities(policy: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, int]:
    result: dict[str, int] = {}
    for node_id, rule in policy["selection_policy"]["stage_rules"].items():
        score = -int(rule["penalty"])
        for signal, benefit in rule["benefits"].items():
            score += int(benefit) * int(bool(signals.get(str(signal), False)))
        result[str(node_id)] = score
    return result


def _feasible(
    candidate: Mapping[str, bool],
    policy: Mapping[str, Any],
    signals: Mapping[str, bool],
) -> bool:
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


def _solve(
    policy: Mapping[str, Any],
    graph: Mapping[str, Any],
    signals: Mapping[str, bool],
) -> dict[str, Any]:
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
        raise DynamicTimeSeriesError(f"CP-SAT must prove OPTIMAL; observed status={status_name}")
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise DynamicTimeSeriesError(f"time-series CP-SAT found no feasible selection: {status_name}")
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
            raise DynamicTimeSeriesError("no feasible time-series selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise DynamicTimeSeriesError(
                f"time-series CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}"
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


def plan_dynamic_time_series(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    signals, features = _signals(ticket, policy)
    optimization = _solve(policy, graph, signals)
    selected = optimization["selected_nodes"]
    ordered = [
        node_id
        for node_id in graph["full_order"]
        if node_id == REQUIRED_STAGE_ID or bool(selected.get(node_id, False))
    ]
    if not ordered or ordered[-1] != REQUIRED_STAGE_ID:
        raise DynamicTimeSeriesError("forecast stage must remain selected and final")
    if len(ordered) > int(policy["maximum_stages"]):
        raise DynamicTimeSeriesError("time-series plan exceeds maximum stages")
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(ordered)
    runtime_graph.add_edges_from(zip(ordered, ordered[1:], strict=False))
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicTimeSeriesError("time-series selected plan contains a cycle")

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
        "time-series family was selected from explicit operation=time_series_forecast and numeric data input",
        (
            "OR-Tools CP-SAT proved the policy-optimal feasible optional-stage subset; "
            f"status={optimization['solver_status']}, objective={optimization['objective_value']}"
        ),
    ]
    if optimization["exhaustive_cross_check"].get("performed"):
        reasons.append("independent exhaustive enumeration matched the time-series CP-SAT optimum")
    for node_id in ordered:
        if node_id == REQUIRED_STAGE_ID:
            continue
        reasons.append(
            f"{graph['nodes'][node_id]['operation']} selected with utility={optimization['utility_by_node'][node_id]}"
        )
    reasons.append("time_series_forecast is the required result stage for this capability family")
    return {
        "id": "dynamic-auto-v1",
        "family": FAMILY,
        "maturity": "controlled-preview",
        "planning_mode": "structured-signal-policy-optimal-family",
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "objective_text_used": False,
        "declared_operation": DECLARED_OPERATION,
        "result_stage": REQUIRED_STAGE_ID,
        "stage_order": ordered,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": reasons,
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute(
    ticket: Mapping[str, Any],
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_time_series(ticket)
    contracts = load_contracts()
    initial_inputs = ticket.get("inputs")
    if not isinstance(initial_inputs, Mapping):
        raise DynamicTimeSeriesError("ticket inputs must be an object")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    stage_elapsed: dict[str, float] = {}
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
        "plan_sha256": _canonical_sha({
            "family": FAMILY,
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
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)

    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = str(stage["operation"])
            if operation not in operations:
                raise DynamicTimeSeriesError(f"handler unavailable: {operation}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[str(stage["adapter"])](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicTimeSeriesError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(
                output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json",
                stage_inputs,
            )
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            stage_elapsed[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise DynamicTimeSeriesError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            try:
                _validate_output(operation, result_dict, contracts)
            except PipelineEngineError as exc:
                raise DynamicTimeSeriesError(str(exc)) from exc
            output_sha = _canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            _write_json(
                output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json",
                result_dict,
            )
            receipt = {
                "stage_id": stage_id,
                "operation": operation,
                "adapter": str(stage["adapter"]),
                "status": "PASS",
                "input_sha256": input_sha,
                "output_sha256": output_sha,
            }
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
    return plan, stage_results, receipts, stage_elapsed


def run_dynamic_time_series_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicTimeSeriesError("ticket is not an admitted time-series dynamic request")
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
        "maturity_assessment": {
            "engineering_maturity": "controlled-preview",
            "evidence_maturity": "controlled-preview",
        },
        "software": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "ortools": ortools.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
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
    transfer["result_sha256"] = _canonical_sha({
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
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(
        output_dir / "compute-audit.json",
        {
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
        },
    )
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Dynamic family: `{FAMILY}`\n"
        f"- Dynamic pipeline: `{plan['id']}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selection engine: `{plan['selection_engine']}`\n"
        f"- Graph engine: `{plan['graph_engine']}`\n"
        f"- Solver status: `{plan['optimization']['solver_status']}`\n"
        f"- Global optimum proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        "- Execution policy: `strict-serial`\n"
        "- Automatic parallel execution: `false`\n"
        "- Model calls: `0`\n"
        "- Network used: `false`\n",
        encoding="utf-8",
    )
    return transfer
