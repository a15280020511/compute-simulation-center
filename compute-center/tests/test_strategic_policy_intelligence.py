from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategic_policy_intelligence_operations import HANDLERS, strategic_policy_analysis  # noqa: E402


class StrategicPolicyGovernanceTests(unittest.TestCase):
    def test_fixed_mode_catalog(self) -> None:
        self.assertEqual(len(HANDLERS), 30)
        expected = {
            "open_spiel_policy_evaluation",
            "pygambit_pure_equilibria",
            "negmas_bilateral_bargaining",
            "scml_supply_chain_competition",
            "pyblp_price_counterfactual",
            "pymc_marketing_budget_allocation",
            "taxcalc_policy_counterfactual",
            "splink_entity_resolution",
            "rdflib_claim_evidence_graph",
            "pyshacl_graph_validation",
            "igraph_link_analysis",
            "claim_evidence_contradiction",
            "red_team_challenge_matrix",
        }
        self.assertTrue(expected <= set(HANDLERS))

    def test_source_has_no_network_or_dynamic_execution(self) -> None:
        source = (ROOT / "strategic_policy_intelligence_operations.py").read_text(encoding="utf-8")
        for forbidden in (
            "import requests",
            "import socket",
            "urllib.request",
            "subprocess.",
            "pickle.loads",
        ):
            self.assertNotIn(forbidden, source)
        tree = ast.parse(source)
        forbidden_calls = {"eval", "exec", "compile"}
        observed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(forbidden_calls.isdisjoint(observed))

    def test_issue_tree_and_value_driver_modes(self) -> None:
        issue = strategic_policy_analysis({
            "mode": "issue_tree_coverage",
            "root": "profit",
            "branches": [
                {"name": "revenue", "weight": 0.6, "evidence_count": 2},
                {"name": "cost", "weight": 0.4, "evidence_count": 0},
            ],
        })
        self.assertAlmostEqual(issue["weighted_coverage"], 0.6)
        self.assertEqual(issue["uncovered_branches"], ["cost"])
        value = strategic_policy_analysis({
            "mode": "value_driver_tree",
            "base_value": 100,
            "drivers": [
                {"name": "price", "change": 5, "multiplier": 2},
                {"name": "cost", "change": -3, "multiplier": 1},
            ],
        })
        self.assertAlmostEqual(value["projected_value"], 107)

    def test_public_intelligence_methods_are_bounded(self) -> None:
        contradiction = strategic_policy_analysis({
            "mode": "claim_evidence_contradiction",
            "claims": ["C1"],
            "evidence": [
                {"claim": "C1", "stance": "support", "weight": 2},
                {"claim": "C1", "stance": "contradict", "weight": 1},
            ],
        })
        self.assertTrue(contradiction["claims"][0]["unresolved"])
        timeline = strategic_policy_analysis({
            "mode": "event_timeline_collision",
            "events": [
                {"id": "E1", "entity": "X", "start": 1, "end": 3, "location": "A"},
                {"id": "E2", "entity": "X", "start": 2, "end": 4, "location": "B"},
            ],
        })
        self.assertEqual(timeline["collision_count"], 1)
        red = strategic_policy_analysis({
            "mode": "red_team_challenge_matrix",
            "assumptions": [
                {"name": "demand", "impact": 0.9, "uncertainty": 0.8, "reversibility": 0.2},
                {"name": "cost", "impact": 0.4, "uncertainty": 0.3, "reversibility": 0.8},
            ],
        })
        self.assertEqual(red["highest_priority"], "demand")
        self.assertTrue(red["offline_execution"])


if __name__ == "__main__":
    unittest.main()
