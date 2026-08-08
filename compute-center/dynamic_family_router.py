#!/usr/bin/env python3
"""Fail-closed router for repository-controlled dynamic capability families."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable
from drift_registry import drift_requirements

DYNAMIC_PIPELINE_ID="dynamic-auto-v1"
DYNAMIC_STAGE_ID="dynamic"
FAMILY_BY_OPERATION={
"scenario_compare":"scenario-decision","time_series_forecast":"time-series",
"causal_policy_evaluation":"causal-policy","bayesian_network_inference":"bayesian-network",
"descriptive_statistics":"reliability","system_dynamics_simulation":"system-dynamics"}
FAMILY_BY_OPERATION_MODE={
("finance_decision_analysis","indirect_intelligence_analysis"):"indirect-intelligence",
("finance_decision_analysis","bounded_linear_kalman_filter"):"state-estimation",
("finance_decision_analysis","mixed_integer_optimization"):"optimization",
("finance_decision_analysis","open_spiel_policy_evaluation"):"game-theory",
("finance_decision_analysis","evidently_data_drift"):"drift",
("finance_decision_analysis","policy_microsimulation"):"policy-simulation",
("finance_decision_analysis","control_step_response"):"control-response",
("finance_decision_analysis","lmfit_exponential_calibration"):"calibration",
("finance_decision_analysis","pm4py_directly_follows"):"process-mining",
("finance_decision_analysis","rsome_robust_allocation"):"robust-allocation",
("finance_decision_analysis","mapie_conformal_interval"):"conformal-prediction"}
INDIRECT_INTELLIGENCE_REQUIREMENTS=[
"requirements-ortools.txt","requirements-intelligence-rapidfuzz.txt",
"requirements-intelligence-datasketch.txt","requirements-intelligence-splink.txt",
"requirements-graph-rdflib.txt","requirements-graph-owlready2.txt",
"requirements-graph-pyshacl.txt","requirements-graph-igraph.txt",
"requirements-global-pm4py.txt","requirements-bayesian-network.txt",
"requirements-intelligence-problog.txt"]
SYSTEM_DYNAMICS_MODES={"stock_flow","feedback_delay","policy_switch","coupled_capacity",
"resource_depletion","adoption_saturation"}

class DynamicFamilyRoutingError(ValueError): pass

def is_dynamic_request(t:Mapping[str,Any])->bool:
 p=t.get("pipeline")
 return bool(isinstance(p,Mapping) and p.get("pipeline_id")==DYNAMIC_PIPELINE_ID and p.get("stage_id")==DYNAMIC_STAGE_ID)

def _s(v:Any,n:str)->Sequence[Any]:
 if isinstance(v,(str,bytes)) or not isinstance(v,Sequence): raise DynamicFamilyRoutingError(f"{n} must be an array")
 return v

def _m(v:Any,n:str,r0:int,r1:int,c0:int,c1:int)->tuple[int,int]:
 rows=_s(v,n)
 if not r0<=len(rows)<=r1: raise DynamicFamilyRoutingError(f"{n} row count is outside the governed range")
 w=None
 for i,x in enumerate(rows):
  row=_s(x,f"{n}[{i}]"); w=len(row) if w is None else w
  if len(row)!=w: raise DynamicFamilyRoutingError(f"{n} must be rectangular")
 if w is None or not c0<=w<=c1: raise DynamicFamilyRoutingError(f"{n} column count is outside the governed range")
 return len(rows),w

def _ctx(i:Mapping[str,Any],name:str)->None:
 v=i.get(name)
 if v is not None and not isinstance(v,Mapping): raise DynamicFamilyRoutingError(f"{name} must be an object when supplied")

def _validate(f:str,o:str,m:str,i:Mapping[str,Any])->None:
 if f=="scenario-decision":
  if not _s(i.get("scenarios"),"inputs.scenarios") or not isinstance(i.get("model"),Mapping): raise DynamicFamilyRoutingError("scenario-decision requires model and scenarios")
 elif f=="time-series":
  if len(_s(i.get("data"),"inputs.data"))<5: raise DynamicFamilyRoutingError("time-series requires at least five observations")
 elif f=="causal-policy":
  if m not in {"backdoor_adjustment","propensity_weighting"}: raise DynamicFamilyRoutingError("unsupported causal-policy mode")
  t=_s(i.get("treatment"),"inputs.treatment"); y=_s(i.get("outcome"),"inputs.outcome")
  if len(t)<8 or len(t)!=len(y) or not isinstance(i.get("confounders"),Mapping) or not i.get("confounders"): raise DynamicFamilyRoutingError("invalid causal-policy inputs")
 elif f=="bayesian-network":
  if m!="bayesian_parameter_estimation": raise DynamicFamilyRoutingError("unsupported bayesian-network mode")
  d=i.get("data")
  if not isinstance(d,Mapping) or not d: raise DynamicFamilyRoutingError("bayesian-network requires data")
  nodes={str(x) for x in d}; q=_s(i.get("query_variables"),"inputs.query_variables")
  if not q or any(str(x) not in nodes for x in q): raise DynamicFamilyRoutingError("invalid query_variables")
  for j,e0 in enumerate(_s(i.get("edges",[]),"inputs.edges")):
   e=_s(e0,f"inputs.edges[{j}]")
   if len(e)!=2 or str(e[0]) not in nodes or str(e[1]) not in nodes: raise DynamicFamilyRoutingError("invalid bayesian edge")
 elif f=="indirect-intelligence":
  if o!="finance_decision_analysis" or m!="indirect_intelligence_analysis" or not str(i.get("hypothesis") or "").strip() or not _s(i.get("evidence"),"inputs.evidence"): raise DynamicFamilyRoutingError("invalid indirect-intelligence inputs")
 elif f=="state-estimation":
  if o!="finance_decision_analysis" or m!="bounded_linear_kalman_filter": raise DynamicFamilyRoutingError("invalid state-estimation mode")
  for n in ("transition_matrix","observation_matrix","process_covariance","observation_covariance","initial_covariance","initial_state","observations"):
   if not _s(i.get(n),f"inputs.{n}"): raise DynamicFamilyRoutingError(f"state-estimation requires {n}")
 elif f=="reliability":
  c=i.get("reliability_context")
  if o!="descriptive_statistics" or len(_s(i.get("data"),"inputs.data"))<2 or not isinstance(c,Mapping) or "threshold" not in c or str(c.get("tail") or "lower").lower() not in {"lower","upper"}: raise DynamicFamilyRoutingError("invalid reliability inputs")
 elif f=="optimization":
  if o!="finance_decision_analysis" or m!="mixed_integer_optimization" or not 1<=len(_s(i.get("variables"),"inputs.variables"))<=200 or len(_s(i.get("constraints",[]),"inputs.constraints"))>1000: raise DynamicFamilyRoutingError("invalid optimization inputs")
 elif f=="system-dynamics":
  z=i.get("steps",100)
  if o!="system_dynamics_simulation" or m not in SYSTEM_DYNAMICS_MODES or isinstance(z,bool) or not isinstance(z,int) or not 1<=z<=10000: raise DynamicFamilyRoutingError("invalid system-dynamics inputs")
 elif f=="game-theory":
  if o!="finance_decision_analysis" or m!="open_spiel_policy_evaluation" or str(i.get("game_id") or "matrix_rps") not in {"matrix_rps","matrix_pd"}: raise DynamicFamilyRoutingError("invalid game-theory inputs")
 elif f=="drift":
  if o!="finance_decision_analysis" or m!="evidently_data_drift": raise DynamicFamilyRoutingError("invalid drift mode")
  _,a=_m(i.get("reference"),"inputs.reference",20,5000,2,30); _,b=_m(i.get("current"),"inputs.current",20,5000,2,30)
  if a!=b: raise DynamicFamilyRoutingError("drift column counts must match")
  _ctx(i,"drift_context")
 elif f=="policy-simulation":
  if o!="finance_decision_analysis" or m!="policy_microsimulation" or not 10<=len(_s(i.get("incomes"),"inputs.incomes"))<=100000 or len(_s(i.get("tax_brackets"),"inputs.tax_brackets"))>100: raise DynamicFamilyRoutingError("invalid policy-simulation inputs")
  _ctx(i,"policy_context")
 elif f=="control-response":
  p=i.get("points",101)
  if o!="finance_decision_analysis" or m!="control_step_response" or not 1<=len(_s(i.get("numerator"),"inputs.numerator"))<=10 or not 2<=len(_s(i.get("denominator"),"inputs.denominator"))<=10 or isinstance(p,bool) or not isinstance(p,int) or not 10<=p<=1000: raise DynamicFamilyRoutingError("invalid control-response inputs")
  _ctx(i,"control_context")
 elif f=="calibration":
  x=_s(i.get("x"),"inputs.x"); y=_s(i.get("y"),"inputs.y")
  if o!="finance_decision_analysis" or m!="lmfit_exponential_calibration" or len(x)!=len(y) or not 5<=len(x)<=5000: raise DynamicFamilyRoutingError("invalid calibration inputs")
  _ctx(i,"calibration_context")
 elif f=="process-mining":
  cases=_s(i.get("cases"),"inputs.cases")
  if o!="finance_decision_analysis" or m!="pm4py_directly_follows" or not 1<=len(cases)<=2000: raise DynamicFamilyRoutingError("invalid process-mining inputs")
  events=0
  for j,c in enumerate(cases):
   if not isinstance(c,Mapping): raise DynamicFamilyRoutingError(f"inputs.cases[{j}] must be an object")
   a=_s(c.get("activities"),f"inputs.cases[{j}].activities")
   if not 1<=len(a)<=200: raise DynamicFamilyRoutingError("invalid process activities")
   events+=len(a)
  if events>10000: raise DynamicFamilyRoutingError("too many process events")
  _ctx(i,"process_context")
 elif f=="robust-allocation":
  if o!="finance_decision_analysis" or m!="rsome_robust_allocation": raise DynamicFamilyRoutingError("invalid robust-allocation mode")
  _m(i.get("scenario_returns"),"inputs.scenario_returns",2,500,2,50); _ctx(i,"robust_allocation_context")
 elif f=="conformal-prediction":
  if o!="finance_decision_analysis" or m!="mapie_conformal_interval": raise DynamicFamilyRoutingError("invalid conformal-prediction mode")
  nr,nc=_m(i.get("train_x"),"inputs.train_x",20,5000,1,30); pr,pc=_m(i.get("predict_x"),"inputs.predict_x",1,1000,1,30)
  if nc!=pc or len(_s(i.get("train_y"),"inputs.train_y"))!=nr: raise DynamicFamilyRoutingError("conformal feature/target dimensions do not align")
  _ctx(i,"conformal_context"); c=i.get("conformal_context")
  if isinstance(c,Mapping) and "validation_observed" in c and len(_s(c.get("validation_observed"),"conformal_context.validation_observed"))!=pr: raise DynamicFamilyRoutingError("validation_observed must match predict_x rows")
 else: raise DynamicFamilyRoutingError(f"unsupported dynamic family: {f}")

def resolve_dynamic_family(t:Mapping[str,Any])->str:
 if not is_dynamic_request(t): raise DynamicFamilyRoutingError("ticket does not request the dynamic production contract")
 o=str(t.get("operation") or ""); i=t.get("inputs")
 if not isinstance(i,Mapping): raise DynamicFamilyRoutingError("dynamic ticket inputs must be an object")
 m=str(i.get("mode") or ""); f=FAMILY_BY_OPERATION_MODE.get((o,m)) or FAMILY_BY_OPERATION.get(o)
 if f is None: raise DynamicFamilyRoutingError(f"dynamic operation/mode is not admitted to any capability family: {o or '<empty>'}/{m or '<none>'}")
 _validate(f,o,m,i); return f

_META={
"causal-policy":("causal_policy_evaluation","dynamic-causal-policy.json","dynamic-causal-capability-graph.json","3.13",["requirements-causal.txt"]),
"bayesian-network":("bayesian_network_inference","dynamic-bayesian-policy.json","dynamic-bayesian-capability-graph.json","3.12",["requirements-bayesian-network.txt"]),
"indirect-intelligence":("finance_decision_analysis:indirect_intelligence_analysis","indirect-intelligence-mode-registry.json","dynamic-indirect-intelligence-capability-graph.json","3.12",INDIRECT_INTELLIGENCE_REQUIREMENTS),
"state-estimation":("finance_decision_analysis:bounded_linear_kalman_filter","dynamic-state-estimation-policy.json","dynamic-state-estimation-capability-graph.json","3.12",[]),
"reliability":("descriptive_statistics:sample-normal-reliability","dynamic-reliability-policy.json","dynamic-reliability-capability-graph.json","3.12",["requirements-global-openturns.txt"]),
"optimization":("finance_decision_analysis:mixed_integer_optimization","dynamic-optimization-policy.json","dynamic-optimization-capability-graph.json","3.12",["requirements-ortools.txt","requirements-thinktank-decision.txt"]),
"system-dynamics":("system_dynamics_simulation:<fixed-mode>","dynamic-system-dynamics-policy.json","dynamic-system-dynamics-capability-graph.json","3.12",["requirements-ortools.txt"]),
"game-theory":("finance_decision_analysis:open_spiel_policy_evaluation","dynamic-game-theory-policy.json","dynamic-game-theory-capability-graph.json","3.12",["requirements-ortools.txt","requirements-strategy-open-spiel.txt","requirements-strategy-pygambit.txt"]),
"policy-simulation":("finance_decision_analysis:policy_microsimulation","dynamic-policy-simulation-policy.json","dynamic-policy-simulation-capability-graph.json","3.12",["requirements-ortools.txt"]),
"control-response":("finance_decision_analysis:control_step_response","dynamic-control-response-policy.json","dynamic-control-response-capability-graph.json","3.12",["requirements-ortools.txt","requirements-global-control.txt"]),
"calibration":("finance_decision_analysis:lmfit_exponential_calibration","dynamic-calibration-policy.json","dynamic-calibration-capability-graph.json","3.12",["requirements-ortools.txt","requirements-global-lmfit.txt"]),
"process-mining":("finance_decision_analysis:pm4py_directly_follows","dynamic-process-mining-policy.json","dynamic-process-mining-capability-graph.json","3.12",["requirements-ortools.txt","requirements-global-pm4py.txt"]),
"robust-allocation":("finance_decision_analysis:rsome_robust_allocation","dynamic-robust-allocation-policy.json","dynamic-robust-allocation-capability-graph.json","3.12",["requirements-ortools.txt","requirements-global-rsome.txt"]),
"conformal-prediction":("finance_decision_analysis:mapie_conformal_interval","dynamic-conformal-prediction-policy.json","dynamic-conformal-prediction-capability-graph.json","3.12",["requirements-ortools.txt","requirements-global-mapie.txt"])}

def family_runtime_metadata(t:Mapping[str,Any])->dict[str,Any]:
 f=resolve_dynamic_family(t)
 if f=="scenario-decision": return {"family":f,"entry_contract":"scenario_compare","policy_file":"dynamic-orchestration-policy.json","graph_file":"dynamic-capability-graph.json"}
 if f=="time-series": return {"family":f,"entry_contract":"time_series_forecast","policy_file":"dynamic-time-series-policy.json","graph_file":"dynamic-time-series-capability-graph.json"}
 if f=="drift": return {"family":f,"entry_contract":"finance_decision_analysis:evidently_data_drift","policy_file":"dynamic-drift-policy.json","graph_file":"dynamic-drift-capability-graph.json","python_version":"3.12","requirements":["requirements-ortools.txt",*drift_requirements(),"requirements-thinktank-econometrics.txt"]}
 if f not in _META: raise DynamicFamilyRoutingError(f"unsupported dynamic family: {f}")
 e,p,g,v,r=_META[f]; return {"family":f,"entry_contract":e,"policy_file":p,"graph_file":g,"python_version":v,"requirements":list(r)}

def run_dynamic_family_ticket(t:Mapping[str,Any],output_dir:Path,operations:Mapping[str,Callable[[Mapping[str,Any]],dict[str,Any]]])->dict[str,Any]:
 f=resolve_dynamic_family(t)
 if f=="scenario-decision":
  from dynamic_pipeline_planner import run_dynamic_pipeline_ticket as run
 elif f=="time-series":
  from dynamic_time_series_planner import run_dynamic_time_series_ticket as run
 elif f=="causal-policy":
  from dynamic_causal_policy_planner import run_dynamic_causal_policy_ticket as run
 elif f=="bayesian-network":
  from dynamic_bayesian_network_planner import run_dynamic_bayesian_network_ticket as run
 elif f=="indirect-intelligence":
  from dynamic_indirect_intelligence_planner import run_dynamic_indirect_intelligence_ticket as run
 elif f=="state-estimation":
  from dynamic_state_estimation_planner import run_dynamic_state_estimation_ticket as run
 elif f=="reliability":
  from dynamic_reliability_planner import run_dynamic_reliability_ticket as run
 elif f=="optimization":
  from dynamic_optimization_planner import run_dynamic_optimization_ticket as run
 elif f=="system-dynamics":
  from dynamic_system_dynamics_planner import run_dynamic_system_dynamics_ticket as run
 elif f=="game-theory":
  from dynamic_game_theory_planner import run_dynamic_game_theory_ticket as run
 elif f=="drift":
  from dynamic_drift_planner import run_dynamic_drift_ticket as run
 elif f=="policy-simulation":
  from dynamic_policy_simulation_planner import run_dynamic_policy_simulation_ticket as run
 elif f=="control-response":
  from dynamic_control_response_planner import run_dynamic_control_response_ticket as run
 elif f=="calibration":
  from dynamic_calibration_planner import run_dynamic_calibration_ticket as run
 elif f=="process-mining":
  from dynamic_process_mining_planner import run_dynamic_process_mining_ticket as run
 elif f=="robust-allocation":
  from dynamic_robust_allocation_planner import run_dynamic_robust_allocation_ticket as run
 elif f=="conformal-prediction":
  from dynamic_conformal_prediction_planner import run_dynamic_conformal_prediction_ticket as run
 else: raise DynamicFamilyRoutingError(f"unsupported dynamic family: {f}")
 return run(t,output_dir,operations)
