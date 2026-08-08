#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for bounded linear control responses."""
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

from dynamic_control_response_adapters import install_control_response_adapters
from dynamic_family_router import resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_control_response_adapters()

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "dynamic-control-response-policy.json"
GRAPH_PATH = HERE / "dynamic-control-response-capability-graph.json"
CONTRACT_PATH = HERE / "dynamic-control-response-stage-contracts.json"
FAMILY = "control-response"
ENTRY_OPERATION = "finance_decision_analysis"
ENTRY_MODE = "control_step_response"
STAGE_ORDER = [
    "control_step_response",
    "tail_response_statistics",
    "tail_stability_audit",
    "dc_gain_consistency_audit",
    "control_target_audit",
]
RESULT_STAGE_ID = "control_step_response"


class DynamicControlResponseError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicControlResponseError(f"JSON root must be an object: {path.name}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicControlResponseError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DynamicControlResponseError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicControlResponseError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicControlResponseError(f"{name} must be finite")
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
    if value.get("schema_version") != "compute-dynamic-control-response-stage-contracts-v1":
        raise DynamicControlResponseError("invalid control-response stage contract schema")
    if value.get("status") != "controlled-preview" or value.get("family") != FAMILY:
        raise DynamicControlResponseError("control-response stage contract identity mismatch")
    contracts = value.get("contracts")
    if not isinstance(contracts, Mapping) or list(contracts) != STAGE_ORDER:
        raise DynamicControlResponseError("control-response contracts must exactly cover stages in fixed order")
    result: dict[str, Any] = {}
    for stage_id, schema in contracts.items():
        if not isinstance(schema, Mapping):
            raise DynamicControlResponseError(f"invalid contract: {stage_id}")
        Draft202012Validator.check_schema(dict(schema))
        result[str(stage_id)] = dict(schema)
    return result


def _validate_stage_output(stage_id: str, result: Mapping[str, Any], contracts: Mapping[str, Any]) -> None:
    schema = contracts.get(stage_id)
    if not isinstance(schema, Mapping):
        raise DynamicControlResponseError(f"no output contract for stage: {stage_id}")
    errors = sorted(Draft202012Validator(dict(schema)).iter_errors(dict(result)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise DynamicControlResponseError(f"output contract failed for {stage_id} at {path}: {error.message}")


def _load_policy() -> dict[str, Any]:
    policy = _load_json(POLICY_PATH)
    expected = {
        "schema_version": "compute-dynamic-control-response-policy-v1",
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
    for key, expected_value in expected.items():
        if policy.get(key) != expected_value:
            raise DynamicControlResponseError(f"unsafe control-response policy: {key}")
    if policy.get("allowed_operations") != ["finance_decision_analysis", "descriptive_statistics"]:
        raise DynamicControlResponseError("control-response operation allowlist mismatch")
    if policy.get("allowed_entry_modes") != [ENTRY_MODE]:
        raise DynamicControlResponseError("control-response entry mode allowlist mismatch")
    solver = _mapping(policy.get("solver_policy"), "solver_policy")
    if solver.get("require_optimal_status") is not True or int(solver.get("num_search_workers") or 0) != 1:
        raise DynamicControlResponseError("selector must require OPTIMAL with one worker")
    rules = _mapping(_mapping(policy.get("selection_policy"), "selection_policy").get("stage_rules"), "stage_rules")
    if list(rules) != STAGE_ORDER[1:]:
        raise DynamicControlResponseError("optional rule order is fixed")
    return policy


def _load_graph(policy: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_json(GRAPH_PATH)
    if graph.get("schema_version") != "compute-dynamic-control-response-capability-graph-v1":
        raise DynamicControlResponseError("invalid control-response graph schema")
    if graph.get("status") != "controlled-preview" or graph.get("family") != FAMILY:
        raise DynamicControlResponseError("control-response graph identity mismatch")
    safety = _mapping(graph.get("safety"), "graph.safety")
    for key, expected in {
        "dynamic_operation_discovery_allowed": False,
        "ticket_supplied_nodes_allowed": False,
        "ticket_supplied_edges_allowed": False,
        "ticket_supplied_code_allowed": False,
        "cycles_allowed": False,
        "automatic_parallel_execution": False,
        "branching_allowed": True,
        "execution_remains_strict_serial": True,
    }.items():
        if safety.get(key) != expected:
            raise DynamicControlResponseError(f"unsafe graph policy: {key}")
    order = [str(item) for item in _sequence(graph.get("node_order"), "graph.node_order")]
    raw_nodes = _mapping(graph.get("nodes"), "graph.nodes")
    if order != STAGE_ORDER or set(raw_nodes) != set(STAGE_ORDER):
        raise DynamicControlResponseError("control-response node order is fixed")
    allowed_operations = set(policy["allowed_operations"])
    allowed_adapters = set(policy["allowed_adapters"])
    nodes: dict[str, dict[str, Any]] = {}
    for stage_id in order:
        node = dict(_mapping(raw_nodes[stage_id], f"graph.nodes.{stage_id}"))
        if str(node.get("operation") or "") not in allowed_operations:
            raise DynamicControlResponseError(f"operation not allowlisted: {stage_id}")
        adapter = str(node.get("adapter") or "")
        if adapter not in allowed_adapters or adapter not in ADAPTERS:
            raise DynamicControlResponseError(f"adapter not allowlisted: {adapter}")
        nodes[stage_id] = node
    edges = [(str(edge[0]), str(edge[1])) for edge in _sequence(graph.get("precedence"), "graph.precedence")]
    expected_edges = {
        ("control_step_response", "tail_response_statistics"),
        ("tail_response_statistics", "tail_stability_audit"),
        ("control_step_response", "dc_gain_consistency_audit"),
        ("control_step_response", "control_target_audit"),
    }
    if set(edges) != expected_edges:
        raise DynamicControlResponseError("control-response DAG does not match controlled structure")
    full = nx.DiGraph()
    full.add_nodes_from(order)
    full.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full) or max(dict(full.out_degree()).values()) < 2:
        raise DynamicControlResponseError("control-response graph must be an acyclic branching DAG")
    index = {stage_id: i for i, stage_id in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full, key=lambda node: index[node])) != order:
        raise DynamicControlResponseError("policy order disagrees with NetworkX topology")
    return {"nodes": nodes, "precedence": edges, "full_order": order, "optional_ids": order[1:], "index": index}


def _decision_class(ticket: Mapping[str, Any]) -> str:
    profile = ticket.get("quality_profile")
    value = str(profile.get("decision_class") or "exploratory") if isinstance(profile, Mapping) else "exploratory"
    return value if value in {"exploratory", "formal", "high_stakes"} else "exploratory"


def _stable_dc_eligible(inputs: Mapping[str, Any]) -> tuple[bool, float | None, list[complex]]:
    denominator = np.asarray([_finite(item, "inputs.denominator[]") for item in _sequence(inputs.get("denominator"), "inputs.denominator")], dtype=float)
    numerator = np.asarray([_finite(item, "inputs.numerator[]") for item in _sequence(inputs.get("numerator"), "inputs.numerator")], dtype=float)
    if denominator.size < 2 or abs(float(denominator[0])) < 1e-15:
        raise DynamicControlResponseError("denominator is invalid")
    poles = [complex(item) for item in np.roots(denominator)]
    stable = bool(poles) and all(item.real < -1e-12 for item in poles) and abs(float(denominator[-1])) >= 1e-15
    dc_gain = float(numerator[-1] / denominator[-1]) if stable else None
    return stable, dc_gain, poles


def _signals(ticket: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    if resolve_dynamic_family(ticket) != FAMILY:
        raise DynamicControlResponseError("ticket was not routed to control-response family")
    inputs = _mapping(ticket.get("inputs"), "ticket.inputs")
    if str(inputs.get("mode") or "") != ENTRY_MODE:
        raise DynamicControlResponseError("control-response entry mode mismatch")
    numerator = _sequence(inputs.get("numerator"), "inputs.numerator")
    denominator = _sequence(inputs.get("denominator"), "inputs.denominator")
    if not 1 <= len(numerator) <= 10 or not 2 <= len(denominator) <= 10:
        raise DynamicControlResponseError("control-response coefficient counts are out of bounds")
    for index, value in enumerate(numerator): _finite(value, f"inputs.numerator[{index}]")
    for index, value in enumerate(denominator): _finite(value, f"inputs.denominator[{index}]")
    time_end = _finite(inputs.get("time_end", 10.0), "inputs.time_end")
    points = inputs.get("points", 101)
    if time_end <= 0 or isinstance(points, bool) or not isinstance(points, int) or not 10 <= points <= 1000:
        raise DynamicControlResponseError("time_end/points are out of bounds")
    context_raw = inputs.get("control_context")
    context = {} if context_raw is None else dict(_mapping(context_raw, "inputs.control_context"))
    allowed = {
        "tail_profile_requested", "tail_fraction", "maximum_tail_standard_deviation", "tail_standard_deviation_tolerance",
        "dc_gain_consistency_requested", "dc_gain_tolerance",
        "maximum_overshoot_percent", "overshoot_tolerance", "minimum_final_value", "maximum_final_value", "final_value_tolerance",
    }
    unexpected = sorted(set(context) - allowed)
    if unexpected:
        raise DynamicControlResponseError(f"control_context contains unsupported fields: {unexpected}")
    for name in ("tail_profile_requested", "dc_gain_consistency_requested"):
        if name in context and not isinstance(context[name], bool):
            raise DynamicControlResponseError(f"{name} must be boolean")
    if "tail_fraction" in context:
        fraction = _finite(context["tail_fraction"], "control_context.tail_fraction")
        if not 0.05 <= fraction <= 0.5:
            raise DynamicControlResponseError("tail_fraction must be between 0.05 and 0.5")
    tail_target = "maximum_tail_standard_deviation" in context
    if "tail_standard_deviation_tolerance" in context and not tail_target:
        raise DynamicControlResponseError("tail_standard_deviation_tolerance requires maximum_tail_standard_deviation")
    if tail_target:
        if _finite(context["maximum_tail_standard_deviation"], "control_context.maximum_tail_standard_deviation") < 0:
            raise DynamicControlResponseError("maximum_tail_standard_deviation must be non-negative")
        if _finite(context.get("tail_standard_deviation_tolerance", 0.0), "control_context.tail_standard_deviation_tolerance") < 0:
            raise DynamicControlResponseError("tail_standard_deviation_tolerance must be non-negative")
    dc_requested = bool(context.get("dc_gain_consistency_requested", False))
    stable_dc, dc_gain, poles = _stable_dc_eligible(inputs)
    if dc_requested and not stable_dc:
        raise DynamicControlResponseError("DC-gain consistency requires a stable continuous-time transfer function with finite DC gain")
    if "dc_gain_tolerance" in context and not dc_requested:
        raise DynamicControlResponseError("dc_gain_tolerance requires dc_gain_consistency_requested=true")
    if dc_requested and _finite(context.get("dc_gain_tolerance", 1e-3), "control_context.dc_gain_tolerance") < 0:
        raise DynamicControlResponseError("dc_gain_tolerance must be non-negative")
    target_names = {"maximum_overshoot_percent", "minimum_final_value", "maximum_final_value"}
    target_count = len(target_names & set(context))
    if "overshoot_tolerance" in context and "maximum_overshoot_percent" not in context:
        raise DynamicControlResponseError("overshoot_tolerance requires maximum_overshoot_percent")
    if "final_value_tolerance" in context and not ({"minimum_final_value", "maximum_final_value"} & set(context)):
        raise DynamicControlResponseError("final_value_tolerance requires a final-value target")
    if "maximum_overshoot_percent" in context and _finite(context["maximum_overshoot_percent"], "control_context.maximum_overshoot_percent") < 0:
        raise DynamicControlResponseError("maximum_overshoot_percent must be non-negative")
    for name in ("overshoot_tolerance", "final_value_tolerance"):
        if name in context and _finite(context[name], f"control_context.{name}") < 0:
            raise DynamicControlResponseError(f"{name} must be non-negative")
    decision_class = _decision_class(ticket)
    signals = {
        "tail_profile_requested": bool(context.get("tail_profile_requested", False)),
        "tail_stability_target_available": tail_target,
        "dc_gain_consistency_requested": dc_requested,
        "control_targets_available": target_count > 0,
        "formal_or_high_stakes": decision_class in {"formal", "high_stakes"},
    }
    features = {
        "decision_class": decision_class,
        "numerator_coefficients": len(numerator), "denominator_coefficients": len(denominator),
        "time_end": time_end, "points": points, "stable_dc_gain_eligible": stable_dc,
        "independent_dc_gain": dc_gain,
        "independent_poles": [[float(p.real), float(p.imag)] for p in poles],
        "control_target_count": target_count,
        **signals,
    }
    return signals, features


def _eligible(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return all(bool(signals.get(str(name), False)) for name in rule.get("eligible_all", []))


def _required(rule: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    return any(bool(signals.get(str(name), False)) for name in rule.get("required_if_any", []))


def _feasible(selected: Mapping[str, bool], rules: Mapping[str, Any], signals: Mapping[str, bool]) -> bool:
    for stage_id, raw in rules.items():
        rule = _mapping(raw, f"rules.{stage_id}")
        chosen = bool(selected[stage_id])
        if chosen and not _eligible(rule, signals): return False
        if _required(rule, signals) and not chosen: return False
        if chosen and any(not bool(selected[name]) for name in rule.get("requires_selected", [])): return False
    return True


def _solve(policy: Mapping[str, Any], graph: Mapping[str, Any], signals: Mapping[str, bool]) -> dict[str, Any]:
    rules = _mapping(_mapping(policy["selection_policy"], "selection_policy")["stage_rules"], "stage_rules")
    ids = list(graph["optional_ids"])
    utilities: dict[str, int] = {}; eligibility: dict[str, bool] = {}; required: dict[str, bool] = {}
    for stage_id in ids:
        rule = _mapping(rules[stage_id], f"rules.{stage_id}")
        score = -int(rule["penalty"])
        for signal_name, benefit in _mapping(rule["benefits"], "benefits").items():
            score += int(benefit) * int(bool(signals.get(str(signal_name), False)))
        utilities[stage_id] = score; eligibility[stage_id] = _eligible(rule, signals); required[stage_id] = _required(rule, signals)
    model = cp_model.CpModel(); variables = {sid: model.new_bool_var(f"select_{sid}") for sid in ids}
    for sid in ids:
        if not eligibility[sid]: model.add(variables[sid] == 0)
        if required[sid]: model.add(variables[sid] == 1)
        for dep in rules[sid].get("requires_selected", []): model.add(variables[sid] <= variables[str(dep)])
    model.maximize(sum(utilities[sid] * variables[sid] for sid in ids))
    solver = cp_model.CpSolver(); sp = _mapping(policy["solver_policy"], "solver_policy")
    solver.parameters.num_search_workers = int(sp["num_search_workers"]); solver.parameters.random_seed = int(sp["random_seed"]); solver.parameters.max_time_in_seconds = float(sp["max_time_seconds"])
    status = solver.solve(model)
    if status != cp_model.OPTIMAL: raise DynamicControlResponseError(f"stage selection must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected = {sid: bool(solver.value(variables[sid])) for sid in ids}; objective = int(round(solver.objective_value))
    rows=[]
    for bits in itertools.product((False, True), repeat=len(ids)):
        candidate=dict(zip(ids,bits,strict=True))
        if _feasible(candidate,rules,signals): rows.append((candidate,sum(utilities[sid]*int(candidate[sid]) for sid in ids)))
    best=max(score for _,score in rows); optimal=[candidate for candidate,score in rows if score==best]
    if objective != best or selected not in optimal: raise DynamicControlResponseError("CP-SAT optimum disagrees with exhaustive cross-check")
    return {"selected_nodes":selected,"solver_status":solver.StatusName(status),"objective_value":objective,"global_optimal_proven":True,"utility_by_node":utilities,"eligibility_by_node":eligibility,"required_by_node":required,"signals":dict(signals),"solver_policy":{"num_search_workers":int(sp["num_search_workers"]),"random_seed":int(sp["random_seed"]),"max_time_seconds":float(sp["max_time_seconds"]),"require_optimal_status":True},"exhaustive_cross_check":{"performed":True,"optional_node_count":len(ids),"feasible_selection_count":len(rows),"best_objective":best,"optimal_selections":optimal,"unique_optimum":len(optimal)==1,"passed":True}}


def plan_dynamic_control_response(ticket: Mapping[str, Any]) -> dict[str, Any]:
    policy=_load_policy(); graph=_load_graph(policy); _load_contracts(); signals,features=_signals(ticket); optimization=_solve(policy,graph,signals)
    selected={RESULT_STAGE_ID}|{sid for sid,val in optimization["selected_nodes"].items() if val}
    runtime=nx.DiGraph(); runtime.add_nodes_from(sid for sid in graph["full_order"] if sid in selected); runtime.add_edges_from((a,b) for a,b in graph["precedence"] if a in selected and b in selected)
    if not nx.is_directed_acyclic_graph(runtime): raise DynamicControlResponseError("selected plan contains a cycle")
    order=list(nx.lexicographical_topological_sort(runtime,key=lambda node:graph["index"][node])); expected=[sid for sid in graph["full_order"] if sid in selected]
    if order!=expected: raise DynamicControlResponseError("NetworkX order disagrees with policy order")
    stage_map={sid:{"id":sid,"operation":str(graph["nodes"][sid]["operation"]),"mode":str(graph["nodes"][sid].get("mode") or ""),"adapter":str(graph["nodes"][sid]["adapter"]),"depends_on":sorted(runtime.predecessors(sid),key=lambda item:graph["index"][item])} for sid in order}
    return {"id":"dynamic-auto-v1","family":FAMILY,"maturity":"controlled-preview","planning_mode":"structured-signal-policy-optimal-family","selection_engine":"ortools-cp-sat","graph_engine":"networkx","objective_text_used":False,"declared_operation":ENTRY_OPERATION,"declared_mode":ENTRY_MODE,"result_stage":RESULT_STAGE_ID,"required_stages":[RESULT_STAGE_ID],"stage_order":order,"stage_map":stage_map,"planning_features":features,"planning_reasons":["control-response family is selected only from the explicit control_step_response mode and bounded transfer-function coefficients","OR-Tools CP-SAT selects only structured-signal validation branches and must prove OPTIMAL","NumPy independently computes continuous-time poles and DC gain; the DC consistency branch never reuses Python-Control for the benchmark","descriptive_statistics consumes only the selected tail of the Python-Control response for independent tail dispersion measurement","explicit overshoot/final-value targets and tail-stability thresholds are directional informative audits; DC-gain inconsistency fails closed","NetworkX preserves a true branching DAG while execution remains strict serial deterministic topological order","exhaustive enumeration independently verifies the bounded optional-stage optimum"],"optimization":optimization,"network_policy":"deny","automatic_parallel_execution":False,"model_calls":0}


def _execute(ticket: Mapping[str, Any], operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]], output_dir: Path):
    plan=plan_dynamic_control_response(ticket); contracts=_load_contracts(); initial=_mapping(ticket.get("inputs"),"ticket.inputs"); results={}; receipts=[]; elapsed_by={}
    state={"schema_version":"compute-dynamic-pipeline-state-v2","pipeline_id":plan["id"],"family":FAMILY,"status":"RUNNING","planning_mode":plan["planning_mode"],"selection_engine":plan["selection_engine"],"graph_engine":plan["graph_engine"],"automatic_parallel_execution":False,"network_used":False,"model_calls":0,"plan_sha256":_canonical_sha({"family":FAMILY,"stage_order":plan["stage_order"],"stage_map":plan["stage_map"],"planning_features":plan["planning_features"],"optimization":plan["optimization"]}),"stages":[{"stage_id":sid,"operation":plan["stage_map"][sid]["operation"],"mode":plan["stage_map"][sid]["mode"],"depends_on":plan["stage_map"][sid]["depends_on"],"status":"PENDING"} for sid in plan["stage_order"]]}
    _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
    try:
        for index,sid in enumerate(plan["stage_order"]):
            stage=plan["stage_map"][sid]; operation=stage["operation"]; adapter=stage["adapter"]
            if operation not in operations or adapter not in ADAPTERS: raise DynamicControlResponseError(f"handler or adapter unavailable at {sid}")
            for dep in stage["depends_on"]:
                if dep not in results: raise DynamicControlResponseError(f"dependency has not completed: {dep}")
            state["stages"][index]["status"]="RUNNING"; _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
            try: stage_inputs=ADAPTERS[adapter](initial,results,stage)
            except PipelineAdapterError as exc: raise DynamicControlResponseError(f"adapter failed at {sid}: {exc}") from exc
            derived=dict(ticket); derived["operation"]=operation; derived["inputs"]=stage_inputs; validate_operation_inputs(derived)
            input_sha=_canonical_sha(stage_inputs); _write_json(output_dir/"dynamic-pipeline-stages"/f"{index+1:02d}-{sid}-input.json",stage_inputs)
            started=time.perf_counter(); raw=operations[operation](stage_inputs); elapsed_by[sid]=round(time.perf_counter()-started,6)
            if not isinstance(raw,Mapping): raise DynamicControlResponseError(f"stage returned non-object result: {sid}")
            result=dict(raw); _validate_stage_output(sid,result,contracts)
            if sid=="dc_gain_consistency_audit" and result.get("status")!="PASS": raise DynamicControlResponseError("independent DC-gain consistency audit failed")
            output_sha=_canonical_sha(result); results[sid]=result; _write_json(output_dir/"dynamic-pipeline-stages"/f"{index+1:02d}-{sid}-output.json",result)
            receipt={"stage_id":sid,"operation":operation,"mode":stage["mode"],"adapter":adapter,"depends_on":list(stage["depends_on"]),"status":"PASS","input_sha256":input_sha,"output_sha256":output_sha}; receipts.append(receipt); state["stages"][index].update(receipt); _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
    except Exception:
        state["status"]="FAILED"
        for row in state["stages"]:
            if row["status"]=="RUNNING": row["status"]="FAILED"
        _write_json(output_dir/"compute-dynamic-pipeline-state.json",state); raise
    state["status"]="PASS"; state["pipeline_sha256"]=_canonical_sha(receipts); _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
    return plan,results,receipts,elapsed_by


def run_dynamic_control_response_ticket(ticket: Mapping[str, Any], output_dir: Path, operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    if resolve_dynamic_family(ticket)!=FAMILY: raise DynamicControlResponseError("ticket is not an admitted control-response dynamic request")
    output_dir.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); plan,stage_results,receipts,elapsed_by=_execute(ticket,operations,output_dir); elapsed=time.perf_counter()-started
    import ortools, scipy
    validations={sid:stage_results[sid] for sid in STAGE_ORDER[1:] if sid in stage_results}
    result_data={"pipeline_id":plan["id"],"dynamic_family":FAMILY,"pipeline_maturity":plan["maturity"],"planning_mode":plan["planning_mode"],"selection_engine":plan["selection_engine"],"graph_engine":plan["graph_engine"],"automatic_parallel_execution":False,"stage_order":plan["stage_order"],"stage_dependencies":{sid:plan["stage_map"][sid]["depends_on"] for sid in plan["stage_order"]},"planning_features":plan["planning_features"],"planning_reasons":plan["planning_reasons"],"optimization":plan["optimization"],"stage_receipts":receipts,"stage_outputs":stage_results,"final_stage":RESULT_STAGE_ID,"final_result":stage_results[RESULT_STAGE_ID]}
    if validations: result_data["validation_results"]=validations
    software={"python":platform.python_version(),"networkx":nx.__version__,"ortools":ortools.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"control":_package_version("control")}
    runtime=nx.DiGraph(); runtime.add_nodes_from(plan["stage_order"])
    for sid in plan["stage_order"]:
        for dep in plan["stage_map"][sid]["depends_on"]: runtime.add_edge(dep,sid)
    branching=max(dict(runtime.out_degree()).values(),default=0)>1
    transfer={"schema_version":"compute-result-v1","task_id":str(ticket["task_id"]),"status":"success","operation":str(ticket["operation"]),"objective":ticket.get("objective"),"input_sha256":_canonical_sha(ticket),"assumptions":ticket.get("assumptions",[]),"evidence":ticket.get("evidence",[]),"limitations":ticket.get("limitations",[]),"results":result_data,"maturity_assessment":{"engineering_maturity":"controlled-preview","evidence_maturity":"controlled-preview"},"software":software,"execution":{"elapsed_seconds":round(elapsed,6),"stage_elapsed_seconds":elapsed_by,"network_used":False,"model_calls":0,"reproducible":True,"automatic_parallel_execution":False,"graph_contains_branching":branching}}
    transfer["result_sha256"]=_canonical_sha({k:transfer[k] for k in ["schema_version","task_id","operation","input_sha256","assumptions","limitations","results","maturity_assessment","software"]})
    _write_json(output_dir/"compute-result.json",transfer)
    _write_json(output_dir/"compute-audit.json",{"version":1,"status":"PASS","task_id":transfer["task_id"],"operation":transfer["operation"],"pipeline_id":plan["id"],"dynamic_family":FAMILY,"planning_mode":plan["planning_mode"],"selection_engine":plan["selection_engine"],"graph_engine":plan["graph_engine"],"solver_status":plan["optimization"]["solver_status"],"global_optimal_proven":plan["optimization"]["global_optimal_proven"],"input_sha256":transfer["input_sha256"],"result_sha256":transfer["result_sha256"],"elapsed_seconds":transfer["execution"]["elapsed_seconds"],"model_calls":0,"network_used":False,"automatic_parallel_execution":False,"graph_contains_branching":branching,"primary_engine":"python-control","cross_check_engines":[sid for sid in STAGE_ORDER[1:] if sid in stage_results],"ticket_supplied_code_executed":False,"secret_values_included":False})
    (output_dir/"compute-summary.md").write_text("# COMPUTE_COMPLETED\n\n"+f"- Task ID: `{transfer['task_id']}`\n- Operation: `{transfer['operation']}`\n- Dynamic family: `{FAMILY}`\n- Stage order: `{' -> '.join(plan['stage_order'])}`\n- Selector: `{plan['optimization']['solver_status']}`\n- Selector global optimal proven: `{str(plan['optimization']['global_optimal_proven']).lower()}`\n- Network used: `false`\n- Model calls: `0`\n",encoding="utf-8")
    return transfer
