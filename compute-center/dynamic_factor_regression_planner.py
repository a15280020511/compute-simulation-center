#!/usr/bin/env python3
"""Policy-optimal orchestration for exact-crosschecked factor regression."""
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

from dynamic_factor_regression_adapters import install_factor_regression_adapters
from dynamic_family_router import resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_factor_regression_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-factor-regression-policy.json"
GRAPH_PATH = HERE / "dynamic-factor-regression-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-factor-regression-stage-contracts.json"
FAMILY = "factor-regression"
ENTRY_OPERATION = "finance_decision_analysis"
ENTRY_MODE = "factor_regression"
RESULT_STAGE_ID = "factor_regression"
EXACT_AUDIT_STAGE_ID = "numpy_exact_regression_audit"
STAGE_ORDER = [RESULT_STAGE_ID, EXACT_AUDIT_STAGE_ID, "r_squared_target_audit", "residual_volatility_target_audit"]


class DynamicFactorRegressionError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicFactorRegressionError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicFactorRegressionError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicFactorRegressionError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicFactorRegressionError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicFactorRegressionError(f"{name} must be finite")
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
    if value.get("schema_version") != "compute-dynamic-factor-regression-stage-contracts-v1" or value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicFactorRegressionError("invalid factor-regression stage contracts")
    raw = value.get("contracts")
    if not isinstance(raw, Mapping) or list(raw) != STAGE_ORDER:
        raise DynamicFactorRegressionError("factor-regression contracts must exactly cover stages in order")
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
        raise DynamicFactorRegressionError(f"output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-factor-regression-policy-v1",
        "status": "controlled-preview", "family": FAMILY,
        "declared_operation": ENTRY_OPERATION, "declared_mode": ENTRY_MODE,
        "planner": "ortools-cp-sat", "graph_engine": "networkx",
        "network_policy": "deny", "model_calls": 0,
        "objective_text_routing_allowed": False, "structured_signals_only": True,
        "dynamic_operation_discovery_allowed": False, "ticket_supplied_code_allowed": False,
        "automatic_parallel_execution": False, "cycles_allowed": False,
        "branching_allowed": True, "maximum_stages": 4,
    }
    for key, wanted in expected.items():
        if policy.get(key) != wanted:
            raise DynamicFactorRegressionError(f"unsafe factor-regression policy: {key}")
    if policy.get("allowed_operations") != [ENTRY_OPERATION] or policy.get("allowed_entry_modes") != [ENTRY_MODE]:
        raise DynamicFactorRegressionError("factor-regression allowlist mismatch")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    if list(rules) != STAGE_ORDER[1:]:
        raise DynamicFactorRegressionError("factor-regression optional rule order is fixed")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_json(GRAPH_PATH)
    if graph.get("schema_version") != "compute-dynamic-factor-regression-capability-graph-v1" or graph.get("status") != "controlled-preview" or graph.get("family") != FAMILY:
        raise DynamicFactorRegressionError("invalid factor-regression graph")
    safety = _mapping(graph.get("safety"), "graph.safety")
    expected_safety = {
        "dynamic_operation_discovery_allowed": False, "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False, "ticket_supplied_code_allowed": False,
        "cycles_allowed": False, "automatic_parallel_execution": False,
        "branching_allowed": True, "execution_remains_strict_serial": True,
    }
    for key, wanted in expected_safety.items():
        if safety.get(key) != wanted:
            raise DynamicFactorRegressionError(f"unsafe factor-regression graph policy: {key}")
    order = [str(item) for item in _sequence(graph.get("node_order"), "graph.node_order")]
    raw_nodes = _mapping(graph.get("nodes"), "graph.nodes")
    if order != STAGE_ORDER or set(raw_nodes) != set(STAGE_ORDER):
        raise DynamicFactorRegressionError("factor-regression node order mismatch")
    allowed_operations = set(policy["allowed_operations"])
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(raw_nodes[stage_id], f"graph.nodes.{stage_id}"))
        if str(node.get("operation") or "") not in allowed_operations:
            raise DynamicFactorRegressionError(f"unallowlisted operation at {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicFactorRegressionError(f"unallowlisted adapter at {stage_id}")
        nodes[stage_id] = node
    precedence = [(str(edge[0]), str(edge[1])) for edge in _sequence(graph.get("precedence"), "graph.precedence")]
    expected_edges = {(RESULT_STAGE_ID, stage_id) for stage_id in STAGE_ORDER[1:]}
    if set(precedence) != expected_edges:
        raise DynamicFactorRegressionError("factor-regression DAG mismatch")
    full_graph = nx.DiGraph()
    full_graph.add_nodes_from(order)
    full_graph.add_edges_from(precedence)
    if not nx.is_directed_acyclic_graph(full_graph) or max(dict(full_graph.out_degree()).values(), default=0) < 2:
        raise DynamicFactorRegressionError("factor-regression graph must be a branching DAG")
    index = {stage_id: position for position, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full_graph, key=lambda item: index[item])) != order:
        raise DynamicFactorRegressionError("factor-regression topology order mismatch")
    return {"nodes": nodes, "precedence": precedence, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicFactorRegressionError("ticket was not routed to factor-regression family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != ENTRY_MODE:
        raise DynamicFactorRegressionError("factor-regression entry mode mismatch")
    asset = np.asarray([_finite(v, "inputs.asset_returns[]") for v in _sequence(inputs.get("asset_returns"), "inputs.asset_returns")], dtype=float)
    if not 10 <= asset.size <= 20_000:
        raise DynamicFactorRegressionError("factor regression requires 10 to 20000 observations")
    factors = _mapping(inputs.get("factors"), "inputs.factors")
    if not 1 <= len(factors) <= 20:
        raise DynamicFactorRegressionError("factor regression requires 1 to 20 factors")
    names = [str(name) for name in factors]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise DynamicFactorRegressionError("factor names must be non-empty and unique")
    columns = []
    for name in names:
        values = np.asarray([_finite(v, f"inputs.factors[{name}][]") for v in _sequence(factors[name], f"inputs.factors[{name}]")], dtype=float)
        if values.size != asset.size:
            raise DynamicFactorRegressionError("all factor series must match asset return length")
        columns.append(values)
    include_intercept = inputs.get("include_intercept", True)
    if not isinstance(include_intercept, bool):
        raise DynamicFactorRegressionError("include_intercept must be boolean")
    x = np.column_stack(columns)
    if include_intercept:
        x = np.column_stack([np.ones(asset.size, dtype=float), x])
    if asset.size <= x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise DynamicFactorRegressionError("dynamic factor regression requires an overdetermined full-rank design matrix")
    if float(np.sum((asset - np.mean(asset)) ** 2)) <= np.finfo(float).eps:
        raise DynamicFactorRegressionError("dynamic factor regression requires non-constant asset returns")
    covariance = str(inputs.get("covariance_type") or "HAC")
    if covariance not in {"HAC", "HC0", "HC1", "HC2", "HC3", "nonrobust"}:
        raise DynamicFactorRegressionError("unsupported covariance_type")
    if covariance == "HAC":
        lags = inputs.get("hac_lags", 5)
        if isinstance(lags, bool) or not isinstance(lags, int) or not 0 <= lags <= min(250, asset.size - 2):
            raise DynamicFactorRegressionError("invalid hac_lags")

    raw_context = inputs.get("factor_regression_context")
    context = {} if raw_context is None else dict(_mapping(raw_context, "inputs.factor_regression_context"))
    allowed = {"exact_consistency_tolerance", "minimum_r_squared", "r_squared_target_tolerance", "maximum_residual_volatility", "residual_volatility_target_tolerance"}
    unexpected = sorted(set(context) - allowed)
    if unexpected:
        raise DynamicFactorRegressionError(f"factor_regression_context contains unsupported fields: {unexpected}")
    if "exact_consistency_tolerance" in context:
        tolerance = _finite(context["exact_consistency_tolerance"], "factor_regression_context.exact_consistency_tolerance")
        if not 0 <= tolerance <= 1e-6:
            raise DynamicFactorRegressionError("exact_consistency_tolerance must be between 0 and 1e-6")
    if "r_squared_target_tolerance" in context and "minimum_r_squared" not in context:
        raise DynamicFactorRegressionError("r_squared_target_tolerance requires minimum_r_squared")
    if "minimum_r_squared" in context:
        target = _finite(context["minimum_r_squared"], "factor_regression_context.minimum_r_squared")
        if target > 1.0:
            raise DynamicFactorRegressionError("minimum_r_squared cannot exceed 1")
        if "r_squared_target_tolerance" in context and _finite(context["r_squared_target_tolerance"], "factor_regression_context.r_squared_target_tolerance") < 0:
            raise DynamicFactorRegressionError("r_squared_target_tolerance must be non-negative")
    if "residual_volatility_target_tolerance" in context and "maximum_residual_volatility" not in context:
        raise DynamicFactorRegressionError("residual_volatility_target_tolerance requires maximum_residual_volatility")
    if "maximum_residual_volatility" in context:
        target = _finite(context["maximum_residual_volatility"], "factor_regression_context.maximum_residual_volatility")
        if target < 0:
            raise DynamicFactorRegressionError("maximum_residual_volatility must be non-negative")
        if "residual_volatility_target_tolerance" in context and _finite(context["residual_volatility_target_tolerance"], "factor_regression_context.residual_volatility_target_tolerance") < 0:
            raise DynamicFactorRegressionError("residual_volatility_target_tolerance must be non-negative")
    signals = {
        "exact_regression_consistency_required": True,
        "r_squared_target_available": "minimum_r_squared" in context,
        "residual_target_available": "maximum_residual_volatility" in context,
    }
    features = {
        "decision_class": _decision_class(ticket), "observations": int(asset.size),
        "factor_count": len(names), "parameter_count": int(x.shape[1]),
        "include_intercept": include_intercept, "matrix_rank": int(np.linalg.matrix_rank(x)),
        "condition_number": float(np.linalg.cond(x)), "covariance_type": covariance, **signals,
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
        if chosen and not _eligible(rule, signals): return False
        if _required(rule, signals) and not chosen: return False
        if chosen and any(not bool(selection[str(dep)]) for dep in rule.get("requires_selected", [])): return False
    return True


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules")
    stage_ids = list(graph["optional_ids"])
    utility: dict[str, int] = {}; eligibility: dict[str, bool] = {}; required: dict[str, bool] = {}
    for stage_id in stage_ids:
        rule = _mapping(rules[stage_id], f"rules.{stage_id}")
        score = -int(rule["penalty"])
        for signal_name, benefit in _mapping(rule["benefits"], "benefits").items():
            score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utility[stage_id] = score; eligibility[stage_id] = _eligible(rule, signals); required[stage_id] = _required(rule, signals)
    model = cp_model.CpModel(); variables = {stage_id: model.new_bool_var(f"select_{stage_id}") for stage_id in stage_ids}
    for stage_id in stage_ids:
        if not eligibility[stage_id]: model.add(variables[stage_id] == 0)
        if required[stage_id]: model.add(variables[stage_id] == 1)
        for dependency in rules[stage_id].get("requires_selected", []): model.add(variables[stage_id] <= variables[str(dependency)])
    model.maximize(sum(utility[stage_id] * variables[stage_id] for stage_id in stage_ids))
    solver = cp_model.CpSolver(); solver_policy = _mapping(policy["solver_policy"], "solver_policy")
    solver.parameters.num_search_workers = int(solver_policy["num_search_workers"]); solver.parameters.random_seed = int(solver_policy["random_seed"]); solver.parameters.max_time_in_seconds = float(solver_policy["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise DynamicFactorRegressionError(f"selector must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected = {stage_id: bool(solver.value(variables[stage_id])) for stage_id in stage_ids}; objective = int(round(solver.objective_value))
    feasible_rows: list[tuple[dict[str, bool], int]] = []
    for bits in itertools.product((False, True), repeat=len(stage_ids)):
        candidate = dict(zip(stage_ids, bits, strict=True))
        if _feasible(candidate, rules, signals): feasible_rows.append((candidate, sum(utility[s] * int(candidate[s]) for s in stage_ids)))
    best = max(value for _, value in feasible_rows); optima = [candidate for candidate, value in feasible_rows if value == best]
    if objective != best or selected not in optima:
        raise DynamicFactorRegressionError("CP-SAT optimum disagrees with exhaustive cross-check")
    return {"selected_nodes": selected, "solver_status": solver.StatusName(status), "objective_value": objective, "global_optimal_proven": True, "utility_by_node": utility, "eligibility_by_node": eligibility, "required_by_node": required, "signals": dict(signals), "exhaustive_cross_check": {"performed": True, "optional_node_count": len(stage_ids), "feasible_selection_count": len(feasible_rows), "best_objective": best, "optimal_selections": optima, "unique_optimum": len(optima) == 1, "passed": True}}


def plan_dynamic_factor_regression(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy = _load_policy(); graph = _load_graph(policy); _load_contracts(); signals, features = _signals(ticket); optimization = _solve(policy, graph, signals)
    selected = {RESULT_STAGE_ID} | {stage_id for stage_id, enabled in optimization["selected_nodes"].items() if enabled}
    runtime = nx.DiGraph(); runtime.add_nodes_from(stage_id for stage_id in graph["full_order"] if stage_id in selected); runtime.add_edges_from((left, right) for left, right in graph["precedence"] if left in selected and right in selected)
    stage_order = list(nx.lexicographical_topological_sort(runtime, key=lambda item: graph["index"][item])); expected_order = [stage_id for stage_id in graph["full_order"] if stage_id in selected]
    if stage_order != expected_order: raise DynamicFactorRegressionError("NetworkX order disagrees with factor-regression policy")
    stage_map = {stage_id: {"id": stage_id, "operation": str(graph["nodes"][stage_id]["operation"]), "mode": str(graph["nodes"][stage_id].get("mode") or ""), "adapter": str(graph["nodes"][stage_id]["adapter"]), "depends_on": sorted(runtime.predecessors(stage_id), key=lambda item: graph["index"][item])} for stage_id in stage_order}
    return {"id": "dynamic-auto-v1", "family": FAMILY, "maturity": "controlled-preview", "planning_mode": "structured-signal-policy-optimal-family", "selection_engine": "ortools-cp-sat", "graph_engine": "networkx", "objective_text_used": False, "declared_operation": ENTRY_OPERATION, "declared_mode": ENTRY_MODE, "result_stage": RESULT_STAGE_ID, "required_stages": [RESULT_STAGE_ID, EXACT_AUDIT_STAGE_ID], "stage_order": stage_order, "stage_map": stage_map, "planning_features": features, "planning_reasons": ["factor-regression family is selected only from explicit factor_regression inputs", "statsmodels OLS performs the primary bounded factor regression", "NumPy lstsq independently rebuilds the design matrix and recomputes every coefficient, R-squared and residual volatility", "rank-deficient or underdetermined designs fail closed before exact comparison", "explicit R-squared and residual-volatility targets remain informative audits rather than execution-health gates", "OR-Tools CP-SAT proves the policy-optimal branch subset and exhaustive enumeration independently verifies it", "NetworkX preserves the branching DAG while execution remains strict serial"], "optimization": optimization, "network_policy": "deny", "automatic_parallel_execution": False, "model_calls": 0}


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    plan = plan_dynamic_factor_regression(ticket); contracts = _load_contracts(); initial_inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    results: dict[str, dict[str, Any]] = {}; receipts: list[dict[str, Any]] = []; elapsed: dict[str, float] = {}
    state: dict[str, Any] = {"schema_version": "compute-dynamic-pipeline-state-v2", "pipeline_id": plan["id"], "family": FAMILY, "status": "RUNNING", "automatic_parallel_execution": False, "network_used": False, "model_calls": 0, "stages": [{"stage_id": stage_id, "operation": plan["stage_map"][stage_id]["operation"], "mode": plan["stage_map"][stage_id]["mode"], "depends_on": plan["stage_map"][stage_id]["depends_on"], "status": "PENDING"} for stage_id in plan["stage_order"]]}
    _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    try:
        for index, stage_id in enumerate(plan["stage_order"]):
            stage = plan["stage_map"][stage_id]; operation = stage["operation"]; adapter = stage["adapter"]
            for dependency in stage["depends_on"]:
                if dependency not in results: raise DynamicFactorRegressionError(f"dependency incomplete: {dependency}")
            if operation not in operations: raise DynamicFactorRegressionError(f"operation handler is unavailable: {operation}")
            state["stages"][index]["status"] = "RUNNING"; _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
            try: stage_inputs = ADAPTERS[adapter](initial_inputs, results, stage)
            except PipelineAdapterError as exc: raise DynamicFactorRegressionError(f"adapter failed at {stage_id}: {exc}") from exc
            derived = dict(ticket); derived["operation"] = operation; derived["inputs"] = stage_inputs; validate_operation_inputs(derived)
            input_sha = _canonical_sha(stage_inputs); _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            started = time.perf_counter(); raw_result = operations[operation](stage_inputs); elapsed[stage_id] = round(time.perf_counter() - started, 6); result = dict(raw_result); _validate_output(stage_id, result, contracts)
            if stage_id == EXACT_AUDIT_STAGE_ID and result.get("status") != "PASS": raise DynamicFactorRegressionError("NumPy exact factor-regression consistency audit failed")
            output_sha = _canonical_sha(result); results[stage_id] = result; _write_json(output_dir / "dynamic-pipeline-stages" / f"{index + 1:02d}-{stage_id}-output.json", result)
            receipt = {"stage_id": stage_id, "operation": operation, "mode": stage["mode"], "adapter": adapter, "depends_on": list(stage["depends_on"]), "status": "PASS", "input_sha256": input_sha, "output_sha256": output_sha}; receipts.append(receipt); state["stages"][index].update(receipt); _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"; _write_json(output_dir / "compute-dynamic-pipeline-state.json", state); raise
    state["status"] = "PASS"; state["pipeline_sha256"] = _canonical_sha(receipts); _write_json(output_dir / "compute-dynamic-pipeline-state.json", state)
    return plan, results, receipts, elapsed


def run_dynamic_factor_regression_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket) != FAMILY: raise DynamicFactorRegressionError("ticket is not an admitted factor-regression request")
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); plan, stage_results, receipts, stage_elapsed = _execute(ticket, operations, output_dir); total_elapsed = time.perf_counter() - started
    import ortools
    validation_results = {stage_id: stage_results[stage_id] for stage_id in STAGE_ORDER[1:] if stage_id in stage_results}
    result_data: dict[str, Any] = {"pipeline_id": plan["id"], "dynamic_family": FAMILY, "pipeline_maturity": plan["maturity"], "planning_mode": plan["planning_mode"], "selection_engine": plan["selection_engine"], "graph_engine": plan["graph_engine"], "automatic_parallel_execution": False, "stage_order": plan["stage_order"], "stage_dependencies": {stage_id: plan["stage_map"][stage_id]["depends_on"] for stage_id in plan["stage_order"]}, "planning_features": plan["planning_features"], "planning_reasons": plan["planning_reasons"], "optimization": plan["optimization"], "stage_receipts": receipts, "stage_outputs": stage_results, "final_stage": RESULT_STAGE_ID, "final_result": stage_results[RESULT_STAGE_ID], "validation_results": validation_results}
    software = {"python": platform.python_version(), "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__, "statsmodels": _package_version("statsmodels")}
    runtime = nx.DiGraph(); runtime.add_nodes_from(plan["stage_order"])
    for stage_id in plan["stage_order"]:
        for dependency in plan["stage_map"][stage_id]["depends_on"]: runtime.add_edge(dependency, stage_id)
    transfer: dict[str, Any] = {"schema_version": "compute-result-v1", "task_id": str(ticket["task_id"]), "status": "success", "operation": str(ticket["operation"]), "objective": ticket.get("objective"), "input_sha256": _canonical_sha(ticket), "assumptions": ticket.get("assumptions", []), "evidence": ticket.get("evidence", []), "limitations": ticket.get("limitations", []), "results": result_data, "maturity_assessment": {"engineering_maturity": "controlled-preview", "evidence_maturity": "controlled-preview"}, "software": software, "execution": {"elapsed_seconds": round(total_elapsed, 6), "stage_elapsed_seconds": stage_elapsed, "network_used": False, "model_calls": 0, "reproducible": True, "automatic_parallel_execution": False, "graph_contains_branching": max(dict(runtime.out_degree()).values(), default=0) > 1}}
    transfer["result_sha256"] = _canonical_sha({key: transfer[key] for key in ["schema_version", "task_id", "operation", "input_sha256", "assumptions", "limitations", "results", "maturity_assessment", "software"]})
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(output_dir / "compute-audit.json", {"version": 1, "status": "PASS", "task_id": transfer["task_id"], "operation": transfer["operation"], "pipeline_id": plan["id"], "dynamic_family": FAMILY, "solver_status": plan["optimization"]["solver_status"], "global_optimal_proven": True, "result_sha256": transfer["result_sha256"], "network_used": False, "model_calls": 0, "automatic_parallel_execution": False, "graph_contains_branching": transfer["execution"]["graph_contains_branching"], "primary_engine": "statsmodels-ols", "independent_validation_engine": "numpy-lstsq", "ticket_supplied_code_executed": False, "secret_values_included": False})
    (output_dir / "compute-summary.md").write_text("# COMPUTE_COMPLETED\n\n" f"- Task ID: `{transfer['task_id']}`\n" f"- Dynamic family: `{FAMILY}`\n" f"- Stage order: `{' -> '.join(plan['stage_order'])}`\n" f"- Selector: `{plan['optimization']['solver_status']}`\n" "- Primary engine: `statsmodels OLS`\n" "- Independent regression validation: `NumPy lstsq`\n" "- Network used: `false`\n" "- Model calls: `0`\n", encoding="utf-8")
    return transfer
