#!/usr/bin/env python3
"""Policy-optimal dynamic orchestration for bounded exponential calibration."""
from __future__ import annotations

import hashlib, itertools, json, math, platform, time
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from jsonschema import Draft202012Validator
from ortools.sat.python import cp_model

from dynamic_calibration_adapters import install_calibration_adapters
from dynamic_family_router import resolve_dynamic_family
from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

install_calibration_adapters()
HERE=Path(__file__).resolve().parent
POLICY_PATH=HERE/"dynamic-calibration-policy.json"; GRAPH_PATH=HERE/"dynamic-calibration-capability-graph.json"; CONTRACT_PATH=HERE/"dynamic-calibration-stage-contracts.json"
FAMILY="calibration"; ENTRY_OPERATION="finance_decision_analysis"; ENTRY_MODE="lmfit_exponential_calibration"
STAGE_ORDER=["exponential_calibration","residual_statistics","rmse_consistency_audit","residual_bias_audit","parameter_target_audit"]
RESULT_STAGE_ID="exponential_calibration"

class DynamicCalibrationError(ValueError): pass

def _load_json(path):
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise DynamicCalibrationError(f"JSON root must be an object: {path.name}")
    return value

def _mapping(v,n):
    if not isinstance(v,Mapping): raise DynamicCalibrationError(f"{n} must be an object")
    return v

def _sequence(v,n):
    if isinstance(v,(str,bytes)) or not isinstance(v,Sequence): raise DynamicCalibrationError(f"{n} must be an array")
    return v

def _finite(v,n):
    if isinstance(v,bool) or not isinstance(v,(int,float)): raise DynamicCalibrationError(f"{n} must be numeric")
    r=float(v)
    if not math.isfinite(r): raise DynamicCalibrationError(f"{n} must be finite")
    return r

def _canonical_sha(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def _write_json(path,v): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def _package_version(name):
    try: return version(name)
    except PackageNotFoundError: return None

def _load_contracts():
    v=_load_json(CONTRACT_PATH)
    if v.get("schema_version")!="compute-dynamic-calibration-stage-contracts-v1" or v.get("status")!="controlled-preview" or v.get("family")!=FAMILY: raise DynamicCalibrationError("invalid calibration stage contracts")
    c=v.get("contracts")
    if not isinstance(c,Mapping) or list(c)!=STAGE_ORDER: raise DynamicCalibrationError("calibration contracts must exactly cover stages")
    out={}
    for sid,schema in c.items(): Draft202012Validator.check_schema(dict(schema)); out[str(sid)]=dict(schema)
    return out

def _validate_output(sid,result,contracts):
    errors=sorted(Draft202012Validator(contracts[sid]).iter_errors(dict(result)),key=lambda e:list(e.absolute_path))
    if errors:
        e=errors[0]; path=".".join(str(i) for i in e.absolute_path) or "<root>"; raise DynamicCalibrationError(f"output contract failed for {sid} at {path}: {e.message}")

def _load_policy():
    p=_load_json(POLICY_PATH)
    expected={"schema_version":"compute-dynamic-calibration-policy-v1","status":"controlled-preview","family":FAMILY,"declared_operation":ENTRY_OPERATION,"declared_mode":ENTRY_MODE,"planner":"ortools-cp-sat","graph_engine":"networkx","network_policy":"deny","model_calls":0,"objective_text_routing_allowed":False,"structured_signals_only":True,"dynamic_operation_discovery_allowed":False,"ticket_supplied_code_allowed":False,"automatic_parallel_execution":False,"cycles_allowed":False,"branching_allowed":True,"maximum_stages":5}
    for k,v in expected.items():
        if p.get(k)!=v: raise DynamicCalibrationError(f"unsafe calibration policy: {k}")
    if p.get("allowed_operations") != ["finance_decision_analysis","descriptive_statistics"] or p.get("allowed_entry_modes") != [ENTRY_MODE]: raise DynamicCalibrationError("calibration allowlist mismatch")
    rules=_mapping(_mapping(p.get("selection_policy"),"selection_policy").get("stage_rules"),"stage_rules")
    if list(rules)!=STAGE_ORDER[1:]: raise DynamicCalibrationError("calibration optional rule order is fixed")
    return p

def _load_graph(policy):
    g=_load_json(GRAPH_PATH)
    if g.get("schema_version")!="compute-dynamic-calibration-capability-graph-v1" or g.get("status")!="controlled-preview" or g.get("family")!=FAMILY: raise DynamicCalibrationError("invalid calibration graph")
    safety=_mapping(g.get("safety"),"graph.safety")
    for k,v in {"dynamic_operation_discovery_allowed":False,"ticket_supplied_nodes_allowed":False,"ticket_supplied_edges_allowed":False,"ticket_supplied_code_allowed":False,"cycles_allowed":False,"automatic_parallel_execution":False,"branching_allowed":True,"execution_remains_strict_serial":True}.items():
        if safety.get(k)!=v: raise DynamicCalibrationError(f"unsafe calibration graph policy: {k}")
    order=[str(i) for i in _sequence(g.get("node_order"),"graph.node_order")]; raw=_mapping(g.get("nodes"),"graph.nodes")
    if order!=STAGE_ORDER or set(raw)!=set(STAGE_ORDER): raise DynamicCalibrationError("calibration node order mismatch")
    nodes={}; allowed_ops=set(policy["allowed_operations"]); allowed_adapters=set(policy["allowed_adapters"])
    for sid in order:
        n=dict(_mapping(raw[sid],f"graph.nodes.{sid}")); adapter=str(n.get("adapter") or "")
        if str(n.get("operation") or "") not in allowed_ops or adapter not in allowed_adapters or adapter not in ADAPTERS: raise DynamicCalibrationError(f"unallowlisted calibration node: {sid}")
        nodes[sid]=n
    edges=[(str(e[0]),str(e[1])) for e in _sequence(g.get("precedence"),"graph.precedence")]
    expected={(RESULT_STAGE_ID,"residual_statistics"),("residual_statistics","rmse_consistency_audit"),("residual_statistics","residual_bias_audit"),(RESULT_STAGE_ID,"parameter_target_audit")}
    if set(edges)!=expected: raise DynamicCalibrationError("calibration DAG mismatch")
    full=nx.DiGraph(); full.add_nodes_from(order); full.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(full) or max(dict(full.out_degree()).values())<2: raise DynamicCalibrationError("calibration graph must be branching DAG")
    index={sid:i for i,sid in enumerate(order)}
    if list(nx.lexicographical_topological_sort(full,key=lambda n:index[n]))!=order: raise DynamicCalibrationError("calibration topology order mismatch")
    return {"nodes":nodes,"precedence":edges,"full_order":order,"optional_ids":order[1:],"index":index}

def _decision_class(ticket):
    q=ticket.get("quality_profile"); v=str(q.get("decision_class") or "exploratory") if isinstance(q,Mapping) else "exploratory"; return v if v in {"exploratory","formal","high_stakes"} else "exploratory"

def _signals(ticket):
    if resolve_dynamic_family(ticket)!=FAMILY: raise DynamicCalibrationError("ticket was not routed to calibration family")
    inputs=_mapping(ticket.get("inputs"),"ticket.inputs")
    if str(inputs.get("mode") or "")!=ENTRY_MODE: raise DynamicCalibrationError("calibration entry mode mismatch")
    x=_sequence(inputs.get("x"),"inputs.x"); y=_sequence(inputs.get("y"),"inputs.y")
    if len(x)!=len(y) or not 5<=len(x)<=5000: raise DynamicCalibrationError("calibration x/y must align with 5 to 5000 observations")
    for i,v in enumerate(x): _finite(v,f"inputs.x[{i}]")
    for i,v in enumerate(y): _finite(v,f"inputs.y[{i}]")
    ctx_raw=inputs.get("calibration_context"); ctx={} if ctx_raw is None else dict(_mapping(ctx_raw,"inputs.calibration_context"))
    allowed={"residual_profile_requested","rmse_consistency_requested","rmse_consistency_tolerance","maximum_abs_residual_mean","expected_amplitude","amplitude_tolerance","expected_decay","decay_tolerance","expected_offset","offset_tolerance"}
    unexpected=sorted(set(ctx)-allowed)
    if unexpected: raise DynamicCalibrationError(f"calibration_context contains unsupported fields: {unexpected}")
    for name in ("residual_profile_requested","rmse_consistency_requested"):
        if name in ctx and not isinstance(ctx[name],bool): raise DynamicCalibrationError(f"{name} must be boolean")
    rmse_req=bool(ctx.get("rmse_consistency_requested",False))
    if "rmse_consistency_tolerance" in ctx and not rmse_req: raise DynamicCalibrationError("rmse_consistency_tolerance requires rmse_consistency_requested=true")
    if rmse_req and _finite(ctx.get("rmse_consistency_tolerance",1e-10),"calibration_context.rmse_consistency_tolerance")<0: raise DynamicCalibrationError("rmse consistency tolerance must be non-negative")
    bias="maximum_abs_residual_mean" in ctx
    if bias and _finite(ctx["maximum_abs_residual_mean"],"calibration_context.maximum_abs_residual_mean")<0: raise DynamicCalibrationError("maximum_abs_residual_mean must be non-negative")
    targets=("amplitude","decay","offset"); count=0
    for name in targets:
        t=f"expected_{name}"; tol=f"{name}_tolerance"
        if tol in ctx and t not in ctx: raise DynamicCalibrationError(f"{tol} requires {t}")
        if t in ctx:
            _finite(ctx[t],f"calibration_context.{t}"); count+=1
            if _finite(ctx.get(tol,0.0),f"calibration_context.{tol}")<0: raise DynamicCalibrationError(f"{tol} must be non-negative")
    decision=_decision_class(ticket)
    signals={"residual_profile_requested":bool(ctx.get("residual_profile_requested",False)),"rmse_consistency_requested":rmse_req,"residual_bias_target_available":bias,"parameter_targets_available":count>0,"formal_or_high_stakes":decision in {"formal","high_stakes"}}
    return signals,{"decision_class":decision,"observations":len(x),"parameter_target_count":count,**signals}

def _eligible(rule,signals): return all(bool(signals.get(str(n),False)) for n in rule.get("eligible_all",[]))
def _required(rule,signals): return any(bool(signals.get(str(n),False)) for n in rule.get("required_if_any",[]))
def _feasible(sel,rules,signals):
    for sid,raw in rules.items():
        rule=_mapping(raw,f"rules.{sid}"); chosen=bool(sel[sid])
        if chosen and not _eligible(rule,signals): return False
        if _required(rule,signals) and not chosen: return False
        if chosen and any(not bool(sel[d]) for d in rule.get("requires_selected",[])): return False
    return True

def _solve(policy,graph,signals):
    rules=_mapping(_mapping(policy["selection_policy"],"selection_policy")["stage_rules"],"stage_rules"); ids=list(graph["optional_ids"]); util={}; elig={}; req={}
    for sid in ids:
        r=_mapping(rules[sid],f"rules.{sid}"); score=-int(r["penalty"])
        for sn,b in _mapping(r["benefits"],"benefits").items(): score+=int(b)*int(bool(signals.get(str(sn),False)))
        util[sid]=score; elig[sid]=_eligible(r,signals); req[sid]=_required(r,signals)
    model=cp_model.CpModel(); vars={sid:model.new_bool_var(f"select_{sid}") for sid in ids}
    for sid in ids:
        if not elig[sid]: model.add(vars[sid]==0)
        if req[sid]: model.add(vars[sid]==1)
        for dep in rules[sid].get("requires_selected",[]): model.add(vars[sid]<=vars[str(dep)])
    model.maximize(sum(util[sid]*vars[sid] for sid in ids)); solver=cp_model.CpSolver(); sp=_mapping(policy["solver_policy"],"solver_policy"); solver.parameters.num_search_workers=int(sp["num_search_workers"]); solver.parameters.random_seed=int(sp["random_seed"]); solver.parameters.max_time_in_seconds=float(sp["max_time_seconds"])
    status=solver.solve(model)
    if status!=cp_model.OPTIMAL: raise DynamicCalibrationError(f"selector must prove OPTIMAL; observed {solver.StatusName(status)}")
    selected={sid:bool(solver.value(vars[sid])) for sid in ids}; objective=int(round(solver.objective_value)); rows=[]
    for bits in itertools.product((False,True),repeat=len(ids)):
        c=dict(zip(ids,bits,strict=True))
        if _feasible(c,rules,signals): rows.append((c,sum(util[s]*int(c[s]) for s in ids)))
    best=max(v for _,v in rows); opts=[c for c,v in rows if v==best]
    if objective!=best or selected not in opts: raise DynamicCalibrationError("CP-SAT optimum disagrees with exhaustive cross-check")
    return {"selected_nodes":selected,"solver_status":solver.StatusName(status),"objective_value":objective,"global_optimal_proven":True,"utility_by_node":util,"eligibility_by_node":elig,"required_by_node":req,"signals":dict(signals),"exhaustive_cross_check":{"performed":True,"optional_node_count":len(ids),"feasible_selection_count":len(rows),"best_objective":best,"optimal_selections":opts,"unique_optimum":len(opts)==1,"passed":True}}

def plan_dynamic_calibration(ticket):
    p=_load_policy(); g=_load_graph(p); _load_contracts(); signals,features=_signals(ticket); opt=_solve(p,g,signals); selected={RESULT_STAGE_ID}|{s for s,v in opt["selected_nodes"].items() if v}; runtime=nx.DiGraph(); runtime.add_nodes_from(s for s in g["full_order"] if s in selected); runtime.add_edges_from((a,b) for a,b in g["precedence"] if a in selected and b in selected)
    order=list(nx.lexicographical_topological_sort(runtime,key=lambda n:g["index"][n])); expected=[s for s in g["full_order"] if s in selected]
    if order!=expected: raise DynamicCalibrationError("NetworkX order disagrees with calibration policy")
    sm={s:{"id":s,"operation":str(g["nodes"][s]["operation"]),"mode":str(g["nodes"][s].get("mode") or ""),"adapter":str(g["nodes"][s]["adapter"]),"depends_on":sorted(runtime.predecessors(s),key=lambda n:g["index"][n])} for s in order}
    return {"id":"dynamic-auto-v1","family":FAMILY,"maturity":"controlled-preview","planning_mode":"structured-signal-policy-optimal-family","selection_engine":"ortools-cp-sat","graph_engine":"networkx","objective_text_used":False,"declared_operation":ENTRY_OPERATION,"declared_mode":ENTRY_MODE,"result_stage":RESULT_STAGE_ID,"required_stages":[RESULT_STAGE_ID],"stage_order":order,"stage_map":sm,"planning_features":features,"planning_reasons":["calibration family is selected only from explicit lmfit_exponential_calibration inputs","fixed adapters reconstruct point residuals from original x/y plus fitted parameters, then core descriptive_statistics independently summarizes those residuals","RMSE is reconstructed from residual mean and population standard deviation and cross-checked against lmfit's reported RMSE; inconsistency fails closed","residual bias and parameter targets are informative user/model-quality audits rather than execution-health gates","OR-Tools CP-SAT must prove the policy-optimal optional branch subset and exhaustive enumeration independently verifies it","NetworkX preserves the branching DAG while execution remains strict serial"],"optimization":opt,"network_policy":"deny","automatic_parallel_execution":False,"model_calls":0}

def _execute(ticket,operations,output_dir):
    plan=plan_dynamic_calibration(ticket); contracts=_load_contracts(); initial=_mapping(ticket.get("inputs"),"ticket.inputs"); results={}; receipts=[]; elapsed={}; state={"schema_version":"compute-dynamic-pipeline-state-v2","pipeline_id":plan["id"],"family":FAMILY,"status":"RUNNING","automatic_parallel_execution":False,"network_used":False,"model_calls":0,"stages":[{"stage_id":s,"operation":plan["stage_map"][s]["operation"],"mode":plan["stage_map"][s]["mode"],"depends_on":plan["stage_map"][s]["depends_on"],"status":"PENDING"} for s in plan["stage_order"]]}; _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
    try:
        for i,sid in enumerate(plan["stage_order"]):
            stage=plan["stage_map"][sid]; op=stage["operation"]; adapter=stage["adapter"]
            for dep in stage["depends_on"]:
                if dep not in results: raise DynamicCalibrationError(f"dependency incomplete: {dep}")
            state["stages"][i]["status"]="RUNNING"; _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
            try: inputs=ADAPTERS[adapter](initial,results,stage)
            except PipelineAdapterError as exc: raise DynamicCalibrationError(f"adapter failed at {sid}: {exc}") from exc
            derived=dict(ticket); derived["operation"]=op; derived["inputs"]=inputs; validate_operation_inputs(derived); insha=_canonical_sha(inputs); _write_json(output_dir/"dynamic-pipeline-stages"/f"{i+1:02d}-{sid}-input.json",inputs)
            started=time.perf_counter(); raw=operations[op](inputs); elapsed[sid]=round(time.perf_counter()-started,6); result=dict(raw); _validate_output(sid,result,contracts)
            if sid=="rmse_consistency_audit" and result.get("status")!="PASS": raise DynamicCalibrationError("RMSE consistency audit failed")
            outsha=_canonical_sha(result); results[sid]=result; _write_json(output_dir/"dynamic-pipeline-stages"/f"{i+1:02d}-{sid}-output.json",result); receipt={"stage_id":sid,"operation":op,"mode":stage["mode"],"adapter":adapter,"depends_on":list(stage["depends_on"]),"status":"PASS","input_sha256":insha,"output_sha256":outsha}; receipts.append(receipt); state["stages"][i].update(receipt); _write_json(output_dir/"compute-dynamic-pipeline-state.json",state)
    except Exception:
        state["status"]="FAILED"; _write_json(output_dir/"compute-dynamic-pipeline-state.json",state); raise
    state["status"]="PASS"; state["pipeline_sha256"]=_canonical_sha(receipts); _write_json(output_dir/"compute-dynamic-pipeline-state.json",state); return plan,results,receipts,elapsed

def run_dynamic_calibration_ticket(ticket,output_dir,operations):
    if resolve_dynamic_family(ticket)!=FAMILY: raise DynamicCalibrationError("ticket is not an admitted calibration request")
    output_dir.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); plan,stage_results,receipts,elapsed=_execute(ticket,operations,output_dir); total=time.perf_counter()-started
    import numpy as np, ortools, scipy
    validations={s:stage_results[s] for s in STAGE_ORDER[1:] if s in stage_results}; result_data={"pipeline_id":plan["id"],"dynamic_family":FAMILY,"pipeline_maturity":plan["maturity"],"planning_mode":plan["planning_mode"],"selection_engine":plan["selection_engine"],"graph_engine":plan["graph_engine"],"automatic_parallel_execution":False,"stage_order":plan["stage_order"],"stage_dependencies":{s:plan["stage_map"][s]["depends_on"] for s in plan["stage_order"]},"planning_features":plan["planning_features"],"planning_reasons":plan["planning_reasons"],"optimization":plan["optimization"],"stage_receipts":receipts,"stage_outputs":stage_results,"final_stage":RESULT_STAGE_ID,"final_result":stage_results[RESULT_STAGE_ID]}
    if validations: result_data["validation_results"]=validations
    software={"python":platform.python_version(),"networkx":nx.__version__,"ortools":ortools.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"lmfit":_package_version("lmfit")}; runtime=nx.DiGraph(); runtime.add_nodes_from(plan["stage_order"])
    for s in plan["stage_order"]:
        for d in plan["stage_map"][s]["depends_on"]: runtime.add_edge(d,s)
    transfer={"schema_version":"compute-result-v1","task_id":str(ticket["task_id"]),"status":"success","operation":str(ticket["operation"]),"objective":ticket.get("objective"),"input_sha256":_canonical_sha(ticket),"assumptions":ticket.get("assumptions",[]),"evidence":ticket.get("evidence",[]),"limitations":ticket.get("limitations",[]),"results":result_data,"maturity_assessment":{"engineering_maturity":"controlled-preview","evidence_maturity":"controlled-preview"},"software":software,"execution":{"elapsed_seconds":round(total,6),"stage_elapsed_seconds":elapsed,"network_used":False,"model_calls":0,"reproducible":True,"automatic_parallel_execution":False,"graph_contains_branching":max(dict(runtime.out_degree()).values(),default=0)>1}}
    transfer["result_sha256"]=_canonical_sha({k:transfer[k] for k in ["schema_version","task_id","operation","input_sha256","assumptions","limitations","results","maturity_assessment","software"]}); _write_json(output_dir/"compute-result.json",transfer); _write_json(output_dir/"compute-audit.json",{"version":1,"status":"PASS","task_id":transfer["task_id"],"operation":transfer["operation"],"pipeline_id":plan["id"],"dynamic_family":FAMILY,"solver_status":plan["optimization"]["solver_status"],"global_optimal_proven":True,"result_sha256":transfer["result_sha256"],"network_used":False,"model_calls":0,"automatic_parallel_execution":False,"graph_contains_branching":transfer["execution"]["graph_contains_branching"],"primary_engine":"lmfit","ticket_supplied_code_executed":False,"secret_values_included":False}); (output_dir/"compute-summary.md").write_text(f"# COMPUTE_COMPLETED\n\n- Task ID: `{transfer['task_id']}`\n- Dynamic family: `{FAMILY}`\n- Stage order: `{' -> '.join(plan['stage_order'])}`\n- Selector: `{plan['optimization']['solver_status']}`\n- Network used: `false`\n- Model calls: `0`\n",encoding="utf-8"); return transfer
