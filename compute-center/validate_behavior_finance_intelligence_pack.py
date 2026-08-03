#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from behavior_finance_intelligence_operations import HANDLERS


FIXTURES = {
    "quantlib_option_greeks": {
        "evaluation_date": "2026-08-03",
        "maturity_days": 365,
        "spot": 100.0,
        "strike": 100.0,
        "risk_free_rate": 0.03,
        "dividend_yield": 0.0,
        "volatility": 0.2,
        "option_type": "call",
    },
    "quantlib_bond_duration": {
        "evaluation_date": "2026-08-03",
        "maturity_years": 5,
        "settlement_days": 2,
        "face_value": 100.0,
        "coupon_rate": 0.04,
        "market_yield": 0.05,
    },
    "active_inference_policy_choice": {
        "observation_likelihood": [[0.9, 0.2], [0.1, 0.8]],
        "transition_matrices": [
            [[0.9, 0.2], [0.1, 0.8]],
            [[0.2, 0.9], [0.8, 0.1]],
        ],
        "preferences": [1.0, -1.0],
        "prior": [0.5, 0.5],
        "observation": 0,
    },
    "pyod_anomaly_screen": {
        "records": [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.05, 0.0], [8.0, 8.0]],
        "detector": "iforest",
        "contamination": 0.2,
        "seed": 7,
    },
    "market_basket_association_rules": {
        "transactions": [
            ["bread", "milk"],
            ["bread", "milk", "eggs"],
            ["bread", "eggs"],
            ["milk", "eggs"],
        ],
        "min_support": 0.5,
        "min_confidence": 0.5,
        "max_len": 3,
    },
    "replicator_dynamics": {
        "payoff_matrix": [[3.0, 0.0], [5.0, 1.0]],
        "initial_population": [0.5, 0.5],
        "steps": 50,
        "dt": 0.05,
    },
    "finite_population_fixation": {
        "population_size": 100,
        "initial_mutants": 1,
        "relative_fitness": 1.1,
    },
    "prospect_theory_choice": {
        "reference_point": 0.0,
        "options": [
            {"name": "certain", "outcomes": [40.0], "probabilities": [1.0]},
            {"name": "risky", "outcomes": [100.0, 0.0], "probabilities": [0.5, 0.5]},
        ],
    },
    "collective_action_threshold": {
        "thresholds": [0.1, 0.2, 0.4, 0.6, 0.8],
        "initial_adopters": [True, False, False, False, False],
        "external_support": 0.05,
        "steps": 20,
    },
    "rumor_correction_dynamics": {
        "initial_rumor": 0.1,
        "initial_correction": 0.05,
        "rumor_spread_rate": 1.2,
        "correction_spread_rate": 0.9,
        "correction_conversion_rate": 1.0,
        "rumor_forgetting_rate": 0.1,
        "steps": 100,
        "dt": 0.05,
    },
    "trust_reputation_update": {
        "actors": ["A", "B", "C"],
        "initial_trust": [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        "learning_rate": 0.25,
        "events": [
            {"source": "A", "target": "B", "outcome": 1.0},
            {"source": "C", "target": "B", "outcome": 0.8},
        ],
    },
    "group_consensus_pressure": {
        "initial_opinions": [-0.8, -0.2, 0.3, 0.9],
        "private_confidence": [0.8, 0.7, 0.6, 0.9],
        "conformity": [0.2, 0.5, 0.6, 0.1],
        "leader_index": 3,
        "leader_weight": 2.0,
        "steps": 20,
        "dissent_threshold": 0.25,
    },
}


def validate(mode: str, result: dict) -> None:
    assert result["mode"] == mode
    if mode == "quantlib_option_greeks":
        assert result["npv"] > 0 and 0 < result["delta"] < 1
    elif mode == "quantlib_bond_duration":
        assert result["clean_price_per_100"] > 0 and result["modified_duration"] > 0
    elif mode == "active_inference_policy_choice":
        assert result["chosen_action"] in {0, 1} and result["autonomous_loop_used"] is False
    elif mode == "pyod_anomaly_screen":
        assert result["anomaly_count"] == 1 and result["top_rows"][0]["row_index"] == 4
        assert result["agentic_engine_used"] is False and result["mcp_used"] is False
    elif mode == "market_basket_association_rules":
        assert result["frequent_itemsets"] and result["rules"]
    elif mode == "replicator_dynamics":
        assert abs(sum(result["final_population"]) - 1.0) < 1e-9
    elif mode == "finite_population_fixation":
        assert 0 <= result["fixation_probability"] <= 1
    elif mode == "prospect_theory_choice":
        assert len(result["ranking"]) == 2 and result["individual_prediction_allowed"] is False
    elif mode == "collective_action_threshold":
        assert result["final_fraction"] >= result["initial_fraction"]
    elif mode == "rumor_correction_dynamics":
        assert result["persuasion_targeting_allowed"] is False
    elif mode == "trust_reputation_update":
        assert len(result["reputation_ranking"]) == 3
    elif mode == "group_consensus_pressure":
        assert result["psychological_diagnosis_allowed"] is False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(FIXTURES), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = HANDLERS[args.mode](FIXTURES[args.mode])
    validate(args.mode, result)
    receipt = {
        "status": "PASS",
        "mode": args.mode,
        "results": result,
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_used": False,
        "remote_data_used": False,
        "brokerage_execution_used": False,
        "individual_psychological_diagnosis_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
