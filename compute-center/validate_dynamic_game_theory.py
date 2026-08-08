#!/usr/bin/env python3
"""Real production-dispatch validation for the dynamic game-theory family."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_game_theory_planner import run_dynamic_game_theory_ticket


def main() -> None:
    ticket = {
        "task_id": "dynamic-game-theory-validator",
        "objective": "Validate repository-controlled matrix-game orchestration without objective-text routing.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "open_spiel_policy_evaluation",
            "game_id": "matrix_rps",
            "row_policy": [1.0 / 3.0] * 3,
            "column_policy": [1.0 / 3.0] * 3,
            "game_context": {
                "equilibrium_analysis_requested": True,
                "expected_pure_equilibrium_count": 0,
                "expected_policy_utility": [0.0, 0.0],
                "utility_tolerance": 1e-9,
            },
        },
        "pipeline": {
            "pipeline_id": "dynamic-auto-v1",
            "stage_id": "dynamic",
            "sequence_reason": "real dynamic game-theory validation",
            "upstream_refs": [],
        },
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }
    root = Path(tempfile.mkdtemp(prefix="validate-dynamic-game-theory-"))
    try:
        result = run_dynamic_game_theory_ticket(
            ticket,
            root,
            {"finance_decision_analysis": finance_decision_analysis},
        )
        expected = [
            "policy_evaluation",
            "pure_equilibria",
            "equilibrium_count_audit",
            "expected_utility_audit",
        ]
        assert result["status"] == "success"
        assert result["results"]["stage_order"] == expected
        assert result["results"]["optimization"]["solver_status"] == "OPTIMAL"
        assert result["results"]["optimization"]["global_optimal_proven"] is True
        assert result["results"]["optimization"]["exhaustive_cross_check"]["passed"] is True
        assert result["results"]["final_result"]["expected_utility"] == [0.0, 0.0]
        assert result["results"]["validation_results"]["pure_equilibria"]["pure_equilibria"] == []
        assert result["results"]["validation_results"]["equilibrium_count_audit"]["status"] == "PASS"
        assert result["results"]["validation_results"]["expected_utility_audit"]["status"] == "PASS"
        assert result["execution"]["network_used"] is False
        assert result["execution"]["model_calls"] == 0
        assert result["execution"]["automatic_parallel_execution"] is False
        assert result["execution"]["graph_contains_branching"] is True
        print(json.dumps({
            "status": "PASS",
            "stage_order": expected,
            "selector_status": result["results"]["optimization"]["solver_status"],
            "selector_objective": result["results"]["optimization"]["objective_value"],
            "expected_utility": result["results"]["final_result"]["expected_utility"],
            "pure_equilibrium_count": len(result["results"]["validation_results"]["pure_equilibria"]["pure_equilibria"]),
            "branching": result["execution"]["graph_contains_branching"],
            "network_used": result["execution"]["network_used"],
            "model_calls": result["execution"]["model_calls"],
        }, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
