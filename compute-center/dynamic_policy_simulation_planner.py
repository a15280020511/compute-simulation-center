#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for bounded policy microsimulation."""
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
from dynamic_policy_simulation_adapters import install_policy_simulation_adapters
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_policy_simulation_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-policy-simulation-policy.json"
GRAPH_PATH = HERE / "dynamic-policy-simulation-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-policy-simulation-stage-contracts.json"
FAMILY = "policy-simulation"
ENTRY_OPERATION = "finance_decision_analysis"
ENTRY_MODE = "policy_microsimulation"
STAGE_ORDER = [
    "policy_microsimulation",
    "disposable_distribution_statistics",
    "mean_consistency_audit",
    "policy_target_audit",
]
RESULT_STAGE_ID = "policy_microsimulation"


class DynamicPolicySimulationError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicPolicySimulationError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicPolicySimulationError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicPolicySimulationError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicPolicySimulationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicPolicySimulationError(f"{name} must be finite")
    return result


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_contracts() -> dict[str, Any]:
    value = _load_json(CONTRACT_PATH)
    if value.get("schema_version") != "compute-dynamic-policy-simulation-stage-contracts-v1":
        raise DynamicPolicySimulationError("invalid policy-simulation stage contract schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicPolicySimulationError("policy-simulation stage contract identity mismatch")
    contracts = value.get("contracts")
    if not isinstance(contracts, Mapping) or list(contracts) != STAGE_ORDER:
        raise DynamicPolicySimulationError("policy-simulation contracts must exactly cover stages in fixed order")
    normalized: dict[str, Any] = {}
    for stage_id, schema in contracts.items():
        if not isinstance(schema, Mapping):
            raise DynamicPolicySimulationError(f"invalid policy-simulation contract: {stage_id}")
        Draft202012Validator.check_schema(dict(schema))
        normalized[str(stage_id)] = dict(schema)
    return normalized


def _validate_stage_output(stage_id: str, result: Mapping[str, Any], contracts: Mapping[str, Any]) -> None:
    schema = contracts.get(stage_id)
    if not isinstance(schema, Mapping):
        raise DynamicPolicySimulationError(f"no output contract for stage: {stage_id}")
    errors = sorted(Draft202012Validator(dict(schema)).iter_errors(dict(result)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicPolicySimulationError(f"output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-policy-simulation-policy-v1",
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
        "maximum_stages": 4,
    }
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise DynamicPolicySimulationError(f"unsafe policy-simulation policy: {key}")
    if policy.get("allowed_operations") != ["finance_decision_analysis", "descriptive_statistics"]:
        raise DynamicPolicySimulationError("policy-simulation operation allowlist mismatch")
    if policy.get("allowed_entry_modes") != [ENTRY_MODE]:
        raise DynamicPolicySimulationError("policy-simulation entry mode allowlist mismatch")
    adapters = policy.get("allowed_adapters")
    if not isinstance(adapters, list) or len(adapters) != len(set(adapters)):
        raise DynamicPolicySimulationError("allowed_adapters must be unique")
    solver = _mapping(policy.get("solver_policy"), "solver_policy")
    if solver.get("require_optimal_status") is not True or int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicPolicySimulationError("selector must require OPTIMAL with one worker")
    if not 0 < float(solver.get("max_time_seconds") or 0) <= 10:
        raise DynamicPolicySimulationError("selector time bound is invalid")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    if list(rules) != STAGE_ORDER[1:]:
        raise DynamicPolicySimulationError("optional rule order is fixed")
    for stage_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"stage_rules.{stage_id}")
        penalty = rule.get("penalty")
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise DynamicPolicySimulationError(f"invalid penalty: {stage_id}")
        benefits = _mapping(rule.get("benefits"), f"stage_rules.{stage_id}.benefits")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in benefits.values()):
            raise DynamicPolicySimulationError(f"invalid benefits: {stage_id}")
        for name in ("eligible_all", "required_if_any", "requires_selected"):
            rows = rule.get(name, [])
            if not isinstance(rows, list) or any(not isinstance(item, str) or not item for item in rows):
                raise DynamicPolicySimulationError(f"{name} must be a string array: {stage_id}")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_json(GRAPH_PATH)
    if graph.get("schema_version") != "compute-dynamic-policy-simulation-capability-graph-v1":
        raise DynamicPolicySimulationError("invalid policy-simulation graph schema")
    if graph.get("status") != "controlled-preview" or graph.get("family") != FAMILY:
        raise DynamicPolicySimulationError("policy-simulation graph identity mismatch")
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
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise DynamicPolicySimulationError(f"unsafe graph policy: {key}")
    order = [str(item) for item in _sequence(graph.get("node_order"), "graph.node_order")]
    nodes_raw = _mapping(graph.get("nodes"), "graph.nodes")
    if order != STAGE_ORDER or set(nodes_raw) != set(STAGE_ORDER):
        raise DynamicPolicySimulationError("policy-simulation node order is fixed")
    allowed_operations = set(policy["allowed_operations"])
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(nodes_raw[stage_id], f"graph.nodes.{stage_id}"))
        if str(node.get("operation") or "") not in allowed_operations:
            raise DynamicPolicySimulationError(f"operation not allowlisted: {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicPolicySimulationError(f"adapter not allowlisted: {adapter}")
        nodes[stage_id] = node
    if nodes[RESULT_STAGE_ID].get("required") is not True or nodes[RESULT_STAGE_ID].get("result_required") is not True:
        raise DynamicPolicySimulationError("policy_microsimulation must be the required result stage")
    edges: list[tuple[str, str]] = []
    for raw_edge in _sequence(graph.get("precedence"), "graph.precedence"):
        edge = _sequence(raw_edge, "graph.precedence[]")
        if len(edge) != 2:
            raise DynamicPolicySimulationError("precedence edge requires two nodes")
        edges.append((str(edge[0]), str(edge[1])))
    expected_edges = {
        ("policy_microsimulation", "disposable_distribution_statistics"),
        ("disposable_distribution_statistics", "mean_consistency_audit"),
        ("policy_microsimulation", "policy_target_audit"),
    }
    if set(edges) != expected_edges:
        raise DynamicPolicySimulationError("policy-simulation DAG does not match controlled structure")
    full = nx.DiGraph()
    full.add_nodes_from(order)
    full.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full) or max(dict(full.out_degree()).values()) < 2:
        raise DynamicPolicySimulationError("policy-simulation graph must be an acyclic branching DAG")
    index = {stage_id: position for position, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full, key=lambda node: index[node])) != order:
        raise DynamicPolicySimulationError("policy order disagrees with NetworkX topology")
    return {"nodes": nodes, "precedence": edges, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicPolicySimulationError("ticket was not routed to policy-simulation family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != ENTRY_MODE:
        raise DynamicPolicySimulationError("policy-simulation entry mode mismatch")
    incomes = _sequence(inputs.get("incomes"), "inputs.incomes")
    if not 10 <= len(incomes) <= 100_000:
        raise DynamicPolicySimulationError("policy-simulation requires 10 to 100000 incomes")
    for index, value in enumerate(incomes):
        _finite(value, f"inputs.incomes[{index}]")
    brackets = _sequence(inputs.get("tax_brackets"), "inputs.tax_brackets")
    if len(brackets) > 100:
        raise DynamicPolicySimulationError("policy-simulation admits at most 100 tax brackets")
    context_raw = inputs.get("policy_context")
    context = {} if context_raw is None else dict(_mapping(context_raw, "inputs.policy_context"))
    allowed_context = {
        "distribution_profile_requested",
        "mean_consistency_requested",
        "mean_consistency_tolerance",
        "minimum_net_fiscal_balance",
        "net_fiscal_balance_tolerance",
        "maximum_gini_after",
        "gini_after_tolerance",
        "maximum_poverty_rate_after",
        "poverty_rate_after_tolerance",
    }
    unexpected = sorted(set(context) - allowed_context)
    if unexpected:
        raise DynamicPolicySimulationError(f"policy_context contains unsupported fields: {unexpected}")
    for name in ("distribution_profile_requested", "mean_consistency_requested"):
        if name in context and not isinstance(context[name], bool):
            raise DynamicPolicySimulationError(f"{name} must be boolean")
    if "mean_consistency_tolerance" in context and _finite(context["mean_consistency_tolerance"], "policy_context.mean_consistency_tolerance") < 0:
        raise DynamicPolicySimulationError("mean_consistency_tolerance must be non-negative")
    target_specs = (
        ("minimum_net_fiscal_balance", "net_fiscal_balance_tolerance", False),
        ("maximum_gini_after", "gini_after_tolerance", True),
        ("maximum_poverty_rate_after", "poverty_rate_after_tolerance", True),
    )
    target_count = 0
    for target_name, tolerance_name, probability_like in target_specs:
        if tolerance_name in context and target_name not in context:
            raise DynamicPolicySimulationError(f"{tolerance_name} requires {target_name}")
        if target_name not in context:
            continue
        target = _finite(context[target_name], f"policy_context.{target_name}")
        if probability_like and not 0.0 <= target <= 1.0:
            raise DynamicPolicySimulationError(f"{target_name} must be between 0 and 1")
        if tolerance_name in context and _finite(context[tolerance_name], f"policy_context.{tolerance_name}") < 0:
            raise DynamicPolicySimulationError(f"{tolerance_name} must be non-negative")
        target_count += 1
    decision_class = _decision_class(ticket)
    signals = {
        "distribution_profile_requested": bool(context.get("distribution_profile_requested", False)),
        "mean_consistency_requested": bool(context.get("mean_consistency_requested", False)),
        "policy_targets_available": target_count > 0,
        "formal_or_high_stakes": decision_class in {"formal", "high_stakes"},
    }
    return signals, {
        "decision_class": decision_class,
        "population": len(incomes),
        "tax_bracket_count": len(brackets),
        "policy_target_count": target_count,
        **signals,
    }


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", []))


def _selection_feasible(selected: Mapping[str, bool], rules: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    for stage_id, raw_rule in rules.items():
        rule = _mapping(raw_rule, f"rules.{stage_id}")
        chosen = bool(selected[stage_id])
        if chosen and not _eligible(rule, signals):
            return False
        if _required(rule, signals) and not chosen:
            return False
        if chosen and any(not bool(selected[name]) for name in rule.get("requires_selected", [])):
            return False
    return True


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules")
    optional_ids = list(graph["optional_ids"])
    utilities: dict[str, int] = {}
    eligibility: dict[str, bool] = {}
    required: dict[str, bool] = {}
    for stage_id in optional_ids:
        rule = _mapping(rules[stage_id], f"rules.{stage_id}")
        score = -int(rule["penalty"])
        for signal_name, benefit in _mapping(rule["benefits"], f"rules.{stage_id}.benefits").items():
            score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utilities[stage_id] = score
        eligibility[stage_id] = _eligible(rule, signals)
        required[stage_id] = _required(rule, signals)
    model = cp_model.CpModel()
    variables = {stage_id: model.new_bool_var(f"select_{stage_id}") for stage_id in optional_ids}
    for stage_id in optional_ids:
        if not eligibility[stage_id]:
            model.add(variables[stage_id] == 0)
        if required[stage_id]:
            model.add(variables[stage_id] == 1)
        for dependency in rules[stage_id].get("requires_selected", []):
            model.add(variables[stage_id] <= variables[str(dependency)])
    model.maximize(sum(utilities[stage_id] * variables[stage_id] for stage_id in optional_ids))
    solver = cp_model.CpSolver()
    solver_policy = _mapping(policy["solver_policy"], "solver_policy")
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"])
    solver.parameters.random_seed = int(solver_policy["random_seed"])
    solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise DynamicPolicySimulationError(f"stage selection must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected = {stage_id: bool(solver.value(variables[stage_id])) for stage_id in optional_ids}
    objective = int(round(solver.objective_value))
    feasible: list[dict[str, Any]] = []
    if len(optional_ids) <= int(solver_policy["exhaustive_cross_check_max_optional_nodes"]):
        for bits in itertools.product((False, True), repeat=len(optional_ids)):
            candidate = dict(zip(optional_ids, bits, strict=True))
            if not _selection_feasible(candidate, rules, signals):
                continue
            score = sum(utilities[stage_id] * int(candidate[stage_id]) for stage_id in optional_ids)
            feasible.append({"selection": candidate, "objective": score})
        if not feasible:
            raise DynamicPolicySimulationError("no feasible selections during exhaustive cross-check")
        best = max(row["objective"] for row in feasible)
        optimal = [row["selection"] for row in feasible if row["objective"] == best]
        if objective != best or selected not in optimal:
            raise DynamicPolicySimulationError(f"CP-SAT optimum disagrees with exhaustive cross-check: solver={objective}, exhaustive={best}")
        cross = {"performed": True, "optional_node_count": len(optional_ids), "feasible_selection_count": len(feasible), "best_objective": best, "optimal_selections": optimal, "unique_optimum": len(optimal) == 1, "passed": True}
    else:
        cross = {"performed": False, "passed": True}
    return {
        "selected_nodes": selected,
        "solver_status": solver.StatusName(status),
        "objective_value": objective,
        "global_optimal_proven": True,
        "utility_by_node": utilities,
        "eligibility_by_node": eligibility,
        "required_by_node": required,
        "signals": dict(signals),
        "solver_policy": {"num_search_workers": int(solver_policy["num_search_workers"]), "random_seed": int(solver_policy["random_seed"]), "max_time_seconds": float(solver_policy["max_time_seconds"]), "require_optimal_status": True},
        "exhaustive_cross_check": cross,
    }


def plan_dynamic_policy_simulation(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    graph = _load_graph(policy)
    _load_contracts()
    signals, features = _signals(ticket)
    optimization = _solve(policy, graph, signals)
    selected_nodes = {RESULT_STAGE_ID} | {stage_id for stage_id, chosen in optimization["selected_nodes"].items() if chosen}
    runtime_graph = nx.DiGraph()
    runtime_graph.add_nodes_from(stage_id for stage_id in graph["full_order"] if stage_id in selected_nodes)
    runtime_graph.add_edges_from((left, right) for left, right in graph["precedence"] if left in selected_nodes and right in selected_nodes)
    if not nx.is_directed_acyclic_graph(runtime_graph):
        raise DynamicPolicySimulationError("selected policy-simulation plan contains a cycle")
    execution_order = list(nx.lexicographical_topological_sort(runtime_graph, key=lambda node: graph["index"][node]))
    expected_order = [stage_id for stage_id in graph["full_order"] if stage_id in selected_nodes]
    if execution_order != expected_order:
        raise DynamicPolicySimulationError("NetworkX deterministic order disagrees with policy order")
    stage_map: dict[str, dict[str, Any]] = {}
    for stage_id in execution_order:
        node = graph["nodes"][stage_id]
        stage_map[stage_id] = {
            "id": stage_id,
            "operation": str(node["operation"]),
            "mode": str(node.get("mode") or ""),
            "adapter": str(node["adapter"]),
            "depends_on": sorted(runtime_graph.predecessors(stage_id), key=lambda item: graph["index"][item]),
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
        "stage_order": execution_order,
        "stage_map": stage_map,
        "planning_features": features,
        "planning_reasons": [
            "policy-simulation family is selected only from the explicit policy_microsimulation mode and structured bounded inputs",
            f"OR-Tools CP-SAT proved the policy-optimal optional validation subset; status={optimization['solver_status']}, objective={optimization['objective_value']}",
            "the microsimulation emits structured individual disposable-income results plus fiscal, inequality and poverty metrics",
            "descriptive_statistics independently consumes individual_results instead of reusing the microsimulation aggregate mean",
            "benchmark_comparison can fail closed on internal cross-tool mean inconsistency while explicit policy-target audits remain informative PASS/FAIL results",
            "policy targets use directional minimum/maximum semantics so outcomes better than a threshold are not misclassified as failures",
            "NetworkX preserves the branching dependency DAG while actual execution remains strict serial deterministic topological order",
            "independent exhaustive enumeration cross-checks the bounded optional-stage optimum",
        ],
        "optimization": optimization,
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "model_calls": 0,
    }


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_policy_simulation(ticket)
    contracts = _load_contracts()
    initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    stage_results: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    elapsed_by_stage: dict[str, float] = {}
    state: dict[str, Any] = {
        "schema_version": "compute-dynamic-pipeline-state-v2",
        "pipeline_id": plan["id"], "family": FAMILY, "status": "RUNNING",
        "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False, "network_used": False, "model_calls": 0,
        "plan_sha256": _canonical_sha({"family": FAMILY, "stage_order": plan["stage_order"], "stage_map": plan["stage_map"], "planning_features": plan["planning_features"], "optimization": plan["optimization"]}),
        "stages": [{"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "mode": plan["stage_map"][stage_id]["mode"], "depends_on": plan["stage_map"][stage_id]["depends_on"], "status": "PENDING"} for stage_id in plan["stage_order"]],
    }
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]
            operation = stage["operation"]
            adapter_name = stage["adapter"]
            if operation not in operations or adapter_name not in ADAPTERS:
                raise DynamicPolicySimulationError(f"handler or adapter unavailable at {stage_id}")
            for dependency in stage["depends_on"]:
                if dependency not in stage_results:
                    raise DynamicPolicySimulationError(f"dependency has not completed: {dependency}")
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise DynamicPolicySimulationError(f"adapter failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter()
            result = operations[operation](stage_inputs)
            elapsed_by_stage[stage_id] = round(time.perf_counter() - started, 6)
            if not isinstance(result, Mapping):
                raise DynamicPolicySimulationError(f"stage returned non-object result: {stage_id}")
            result_dict = dict(result)
            _validate_stage_output(stage_id, result_dict, contracts)
            if stage_id == "mean_consistency_audit" and result_dict.get("status") != "PASS":
                raise DynamicPolicySimulationError("cross-tool disposable-income mean consistency audit failed")
            output_sha = _canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", result_dict)
            receipt = {"stage_id": stage_id, "operation": operation, "mode": stage["mode"], "adapter": adapter_name, "depends_on": list(stage["depends_on"]), "status": "PASS", "input_sha256": input_sha, "output_sha256": output_sha}
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
    return plan, stage_results, receipts, elapsed_by_stage


def run_dynamic_policy_simulation_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicPolicySimulationError("ticket is not an admitted policy-simulation dynamic request")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, stage_results, receipts, elapsed_by_stage = _execute(ticket, operations, output_dir)
    elapsed = time.perf_counter() - started
    import numpy as np
    import ortools
    import scipy
    validation_results = {stage_id: stage_results[stage_id] for stage_id in STAGE_ORDER[1:] if stage_id in stage_results}
    result_data: dict[str, Any] = {
        "pipeline_id": plan["id"], "dynamic_family": FAMILY, "pipeline_maturity": plan["maturity"],
        "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"],
        "automatic_parallel_execution": False, "stage_order": plan["stage_order"],
        "stage_dependencies": {stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]},
        "planning_features": plan["planning_features"], "planning_reasons": plan["planning_reasons"], "optimization": plan["optimization"],
        "stage_receipts": receipts, "stage_outputs": stage_results, "final_stage": RESULT_STAGE_ID, "final_result": stage_results[RESULT_STAGE_ID],
    }
    if validation_results:
        result_data["validation_results"] = validation_results
    software: dict[str, Any] = {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    software["policy_microsimulation_backend"] = "numpy-core"
    transfer: dict[str, Any] = {
        "schema_version": "compute-result-v1", "task_id": str(ticket["task_id"]), "status": "success", "operation": str(ticket["operation"]),
        "objective": ticket.get("objective"), "input_sha256": _canonical_sha(ticket), "assumptions": ticket.get("assumptions", []), "evidence": ticket.get("evidence", []), "limitations": ticket.get("limitations", []),
        "results": result_data, "maturity_assessment": {"engineering_maturity": "controlled-preview", "evidence_maturity": "controlled-preview"}, "software": software,
        "execution": {"elapsed_seconds": round(elapsed, 6), "stage_elapsed_seconds": elapsed_by_stage, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": max((len([edge for edge in plan["stage_map"] if stage_id in plan["stage_map"][edge]["depends_on"]]) for stage_id in plan["stage_order"]), default=0) > 1},
    }
    transfer["result_sha256"] = _canonical_sha({"schema_version": transfer["schema_version"], "task_id": transfer["task_id"], "operation": transfer["operation"], "input_sha256": transfer["input_sha256"], "assumptions": transfer["assumptions"], "limitations": transfer["limitations"], "results": transfer["results"], "maturity_assessment": transfer["maturity_assessment"], "software": transfer["software"]})
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(output_dir / "compute-audit.json", {
        "version": 1, "status": "PASS", "task_id": transfer["task_id"], "operation": transfer["operation"], "pipeline_id": plan["id"], "dynamic_family": FAMILY,
        "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"], "solver_status": plan["optimization"]["solver_status"], "global_optimal_proven": plan["optimization"]["global_optimal_proven"],
        "input_sha256": transfer["input_sha256"], "result_sha256": transfer["result_sha256"], "elapsed_seconds": transfer["execution"]["elapsed_seconds"], "model_calls": 0, "network_used": False, "automatic_parallel_execution": False,
        "graph_contains_branching": transfer["execution"]["graph_contains_branching"], "primary_engine": "policy-microsimulation-numpy", "cross_check_engines": [stage_id for stage_id in STAGE_ORDER[1:] if stage_id in stage_results],
        "ticket_supplied_code_executed": False, "secret_values_included": False,
    })
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n- Operation: `{transfer['operation']}`\n- Dynamic family: `{FAMILY}`\n"
        f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n"
        f"- Selector: `{plan['optimization']['solver_status']}`\n- Selector global optimal proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n"
        "- Network used: `false`\n- Model calls: `0`\n", encoding="utf-8")
    return transfer
