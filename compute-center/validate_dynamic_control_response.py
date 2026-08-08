#!/usr/bin/env python3
from __future__ import annotations
import json, math, shutil, tempfile
from pathlib import Path
from compute_runner import descriptive_statistics
from decision_intelligence_gateway import finance_decision_analysis
from dynamic_control_response_planner import run_dynamic_control_response_ticket

def main() -> None:
    ticket={"task_id":"dynamic-control-response-validator","objective":"Validate repository-controlled control-response orchestration without objective-text routing.","operation":"finance_decision_analysis","inputs":{"mode":"control_step_response","numerator":[1.0],"denominator":[1.0,1.0],"time_end":10.0,"points":201,"control_context":{"tail_profile_requested":True,"tail_fraction":0.2,"maximum_tail_standard_deviation":0.001,"dc_gain_consistency_requested":True,"dc_gain_tolerance":1e-4,"maximum_overshoot_percent":0.1,"minimum_final_value":0.999,"maximum_final_value":1.001}},"pipeline":{"pipeline_id":"dynamic-auto-v1","stage_id":"dynamic","sequence_reason":"real dynamic control-response validation","upstream_refs":[]},"quality_profile":{"decision_class":"exploratory","publication_policy":"status_only"}}
    root=Path(tempfile.mkdtemp(prefix="validate-dynamic-control-response-"))
    try:
        result=run_dynamic_control_response_ticket(ticket,root,{"finance_decision_analysis":finance_decision_analysis,"descriptive_statistics":descriptive_statistics})
        expected=["control_step_response","tail_response_statistics","tail_stability_audit","dc_gain_consistency_audit","control_target_audit"]
        primary=result["results"]["final_result"]; validation=result["results"]["validation_results"]
        assert result["status"]=="success" and result["results"]["stage_order"]==expected
        assert result["results"]["optimization"]["solver_status"]=="OPTIMAL" and result["results"]["optimization"]["objective_value"]==685
        assert result["results"]["optimization"]["global_optimal_proven"] is True and result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert abs(primary["final_value"]-(1.0-math.exp(-10.0)))<=1e-10
        assert abs(primary["overshoot_percent"])<=1e-12
        assert validation["tail_response_statistics"]["standard_deviation_population"]<0.001
        assert validation["tail_stability_audit"]["status"]=="PASS" and validation["dc_gain_consistency_audit"]["status"]=="PASS" and validation["control_target_audit"]["status"]=="PASS"
        assert result["execution"]["network_used"] is False and result["execution"]["model_calls"]==0 and result["execution"]["automatic_parallel_execution"] is False and result["execution"]["graph_contains_branching"] is True
        print(json.dumps({"status":"PASS","stage_order":expected,"selector_status":"OPTIMAL","selector_objective":685,"final_value":primary["final_value"],"overshoot_percent":primary["overshoot_percent"],"tail_std":validation["tail_response_statistics"]["standard_deviation_population"],"dc_gain_audit":validation["dc_gain_consistency_audit"]["status"],"targets":validation["control_target_audit"]["status"],"branching":True,"network_used":False,"model_calls":0},sort_keys=True))
    finally: shutil.rmtree(root,ignore_errors=True)
if __name__=="__main__": main()
