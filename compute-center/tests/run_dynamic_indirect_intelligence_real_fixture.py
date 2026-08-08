#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import resolve_dynamic_family, run_dynamic_family_ticket
from tool_registry import managed_runtime_plan


def build_ticket() -> dict:
    return {
        "task_id": "dynamic-indirect-real",
        "operation": "finance_decision_analysis",
        "objective": "Validate the governed indirect intelligence dynamic family",
        "inputs": {
            "mode": "indirect_intelligence_analysis",
            "analysis_depth": "deep",
            "hypothesis": "技术A已经进入实际司法应用",
            "prior_probability": 0.4,
            "scope": {
                "time_window": "2025-2026",
                "geographic_scope": "测试地区X/Y",
                "institution_scope": "公开机构样本",
            },
            "evidence": [
                {
                    "evref": "ev-training",
                    "analysis_class": "DIRECT",
                    "stance": "support",
                    "reliability": 0.82,
                    "entity": "机构甲",
                    "institution": "机构甲",
                    "geography": "地区X",
                    "aliases": ["甲机构"],
                    "tokens": ["培训", "技术A", "课程"],
                    "p_if_true": 0.8,
                    "p_if_false": 0.3,
                    "case_id": "case-x",
                    "activity": "培训",
                    "timestamp": "2025-01-01",
                },
                {
                    "evref": "ev-procurement",
                    "analysis_class": "DIRECT",
                    "stance": "support",
                    "reliability": 0.9,
                    "entity": "机构乙",
                    "institution": "机构乙",
                    "geography": "地区Y",
                    "aliases": ["乙机构"],
                    "tokens": ["采购", "技术A", "部署"],
                    "p_if_true": 0.9,
                    "p_if_false": 0.2,
                    "case_id": "case-x",
                    "activity": "部署",
                    "timestamp": "2025-06-01",
                },
                {
                    "evref": "ev-counter",
                    "analysis_class": "DIRECT",
                    "stance": "contradict",
                    "reliability": 0.25,
                    "entity": "公开审计",
                    "institution": "公开审计",
                    "geography": "地区X",
                    "tokens": ["试点", "限制", "技术A"],
                    "p_if_true": 0.3,
                    "p_if_false": 0.7,
                },
            ],
            "entity_records": [
                {"name": "机构乙", "institution": "机构乙", "geography": "地区Y"},
                {"name": "乙机构", "institution": "机构乙", "geography": "地区Y"},
            ],
            "entity_fields": ["name", "institution", "geography"],
            "relations": [
                {"subject": "机构甲", "predicate": "培训", "object": "技术A"},
                {"subject": "机构乙", "predicate": "部署", "object": "技术A"},
                {"subject": "技术A", "predicate": "用于", "object": "案件实践"},
            ],
            "path_targets": ["机构乙", "技术A", "案件实践"],
            "process_cases": [
                {
                    "case_id": "lifecycle",
                    "activities": ["培训", "采购", "部署", "案件实践"],
                }
            ],
            "rules": [
                {
                    "name": "deployment-signal",
                    "required_evrefs": ["ev-training", "ev-procurement"],
                }
            ],
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real indirect intelligence dynamic family validation",
            "upstream_refs": [],
        },
    }


def main() -> int:
    ticket = build_ticket()
    assert resolve_dynamic_family(ticket) == "indirect-intelligence"
    plan = managed_runtime_plan(ticket)
    assert plan["dynamic_family"] == "indirect-intelligence", plan
    assert plan["mode"] == "indirect_intelligence_analysis", plan
    assert plan["network_policy"] == "deny", plan
    assert len(plan["requirements"]) == 11, plan

    transfer = run_dynamic_family_ticket(
        ticket,
        Path("dynamic-indirect-validation/runtime"),
        {"finance_decision_analysis": finance_decision_analysis},
    )
    result = transfer["results"]
    final = result["final_result"]
    expected = [
        "name_normalization",
        "similarity_collision",
        "entity_resolution",
        "knowledge_graph",
        "graph_analysis",
        "process_mining",
        "probabilistic_inference",
        "contradiction_check",
    ]
    assert transfer["status"] == "success", transfer
    assert result["dynamic_family"] == "indirect-intelligence", result
    assert result["stage_order"] == expected, result
    assert final["stage_plan"]["solver_status"] == "OPTIMAL", final["stage_plan"]
    assert final["analysis_class"] in {"INFERRED", "CONTRADICTED"}, final
    assert final["inference_not_fact"] is True, final
    assert final["scope_extrapolation_allowed"] is False, final
    assert final["network_used"] is False, final
    assert final["external_data_fetches"] == 0, final
    assert final["model_calls"] == 0, final
    assert transfer["execution"]["network_used"] is False, transfer
    assert transfer["execution"]["model_calls"] == 0, transfer
    summary = {
        "status": "PASS",
        "dynamic_family": result["dynamic_family"],
        "stage_order": result["stage_order"],
        "analysis_class": final["analysis_class"],
        "posterior_probability": final["posterior_probability"],
        "confidence": final["confidence"],
        "network_used": transfer["execution"]["network_used"],
        "model_calls": transfer["execution"]["model_calls"],
        "dependency_packs": len(plan["requirements"]),
    }
    Path("dynamic-indirect-validation").mkdir(parents=True, exist_ok=True)
    Path("dynamic-indirect-validation/result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
