#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest.mock import patch

import capability_manager
import indirect_intelligence_operations as indirect
from decision_intelligence_gateway import (
    ALL_SUPPORTED_MODES,
    CONTROLLED_PREVIEW_OVERLAY_MODES,
    finance_decision_analysis,
)

# A connector-authored commit on this test path intentionally re-triggers the
# feature-branch integration workflow after bot-applied semantic fixes.


class IndirectIntelligenceRegistryTests(unittest.TestCase):
    def test_mode_is_registered_as_controlled_preview_overlay(self) -> None:
        registry = capability_manager.load_registry()
        groups = [
            row for row in registry["groups"] if row.get("id") == "decision-intelligence"
        ]
        self.assertEqual(len(groups), 1)
        mode = groups[0]["modes"][indirect.MODE]
        self.assertEqual(mode["maturity"], "controlled-preview")
        self.assertEqual(mode["network_policy"], "deny")
        self.assertTrue(mode["deterministic"])
        self.assertEqual(mode["limits"]["max_stages"], 8)
        self.assertIn(indirect.MODE, CONTROLLED_PREVIEW_OVERLAY_MODES)
        self.assertNotIn(indirect.MODE, ALL_SUPPORTED_MODES)

    def test_requirement_bundle_reuses_existing_pinned_packs(self) -> None:
        ticket = {
            "operation": "finance_decision_analysis",
            "inputs": {"mode": indirect.MODE},
        }
        requirements = [
            path.rsplit("/", 1)[-1]
            for path in capability_manager.requirements_for_ticket(ticket)
        ]
        self.assertIn("requirements-ortools.txt", requirements)
        self.assertIn("requirements-intelligence-splink.txt", requirements)
        self.assertIn("requirements-bayesian-network.txt", requirements)
        self.assertIn("requirements-global-pm4py.txt", requirements)
        self.assertIn("requirements-intelligence-problog.txt", requirements)


class IndirectIntelligenceContractTests(unittest.TestCase):
    def _base_inputs(self) -> dict:
        return {
            "mode": indirect.MODE,
            "hypothesis": "技术A已经进入实际应用",
            "prior_probability": 0.4,
            "scope": {
                "time_window": "2025-2026",
                "geographic_scope": "X/Y/Z",
                "institution_scope": "公开司法机构样本",
            },
            "assumptions": ["公开信息存在选择性披露"],
            "evidence": [
                {
                    "evref": "ev-training",
                    "analysis_class": "DIRECT",
                    "stance": "support",
                    "reliability": 0.8,
                    "entity": "机构甲",
                },
                {
                    "evref": "ev-case",
                    "analysis_class": "LINKED",
                    "stance": "support",
                    "reliability": 0.9,
                    "entity": "机构乙",
                },
                {
                    "evref": "ev-counter",
                    "analysis_class": "DIRECT",
                    "stance": "contradict",
                    "reliability": 0.2,
                },
            ],
        }

    def test_output_never_promotes_linked_or_inferred_result_to_fact(self) -> None:
        inputs = self._base_inputs()
        fake_plan = {
            "selected_stages": ["contradiction_check"],
            "signals": {},
            "solver_status": "OPTIMAL",
            "objective_value": 45,
            "selection_engine": "ortools-cp-sat",
            "graph_engine": "networkx",
            "serial_execution": True,
            "automatic_parallel_execution": False,
            "maximum_stages": 8,
        }
        with patch.object(
            indirect,
            "_select_stages",
            return_value=fake_plan,
        ), patch.object(
            indirect,
            "_contradiction",
            return_value={"claims": []},
        ):
            result = indirect.indirect_intelligence_analysis(inputs)
        self.assertIn(result["analysis_class"], indirect.ANALYSIS_CLASSES)
        self.assertFalse(result["scope_extrapolation_allowed"])
        self.assertFalse(result["network_used"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["external_data_fetches"], 0)
        self.assertEqual(result["supporting_evrefs"], ["ev-training", "ev-case"])
        self.assertEqual(result["contradicting_evrefs"], ["ev-counter"])
        self.assertIn("publication_boundary", result)

    def test_posterior_forces_inferred_class_even_with_direct_inputs(self) -> None:
        inputs = self._base_inputs()
        inputs["evidence"][0].update({"p_if_true": 0.8, "p_if_false": 0.2})
        fake_plan = {
            "selected_stages": ["probabilistic_inference", "contradiction_check"],
            "signals": {},
            "solver_status": "OPTIMAL",
            "objective_value": 85,
            "selection_engine": "ortools-cp-sat",
            "graph_engine": "networkx",
            "serial_execution": True,
            "automatic_parallel_execution": False,
            "maximum_stages": 8,
        }
        with patch.object(
            indirect,
            "_select_stages",
            return_value=fake_plan,
        ), patch.object(
            indirect,
            "_probabilistic_inference",
            return_value={
                "bayesian": {"posterior_probability": 0.79},
                "problog_rules": [],
            },
        ), patch.object(
            indirect,
            "_contradiction",
            return_value={"claims": []},
        ):
            result = indirect.indirect_intelligence_analysis(inputs)
        self.assertEqual(result["analysis_class"], "INFERRED")
        self.assertAlmostEqual(result["posterior_probability"], 0.79)
        self.assertTrue(result["inference_not_fact"])

    def test_rule_only_probabilistic_stage_stays_inferred(self) -> None:
        inputs = self._base_inputs()
        inputs["rules"] = [
            {
                "name": "support-rule",
                "required_evrefs": ["ev-training", "ev-case"],
            }
        ]
        fake_plan = {
            "selected_stages": ["probabilistic_inference", "contradiction_check"],
            "signals": {},
            "solver_status": "OPTIMAL",
            "objective_value": 85,
            "selection_engine": "ortools-cp-sat",
            "graph_engine": "networkx",
            "serial_execution": True,
            "automatic_parallel_execution": False,
            "maximum_stages": 8,
        }
        with patch.object(
            indirect,
            "_select_stages",
            return_value=fake_plan,
        ), patch.object(
            indirect,
            "_probabilistic_inference",
            return_value={"bayesian": None, "problog_rules": [{"joint_probability": 0.72}]},
        ), patch.object(
            indirect,
            "_contradiction",
            return_value={"claims": []},
        ):
            result = indirect.indirect_intelligence_analysis(inputs)
        self.assertEqual(result["analysis_class"], "INFERRED")
        self.assertIsNone(result["posterior_probability"])
        self.assertTrue(result["inference_not_fact"])

    def test_gateway_dispatches_new_mode(self) -> None:
        inputs = self._base_inputs()
        fake_plan = {
            "selected_stages": ["contradiction_check"],
            "signals": {},
            "solver_status": "OPTIMAL",
            "objective_value": 45,
            "selection_engine": "ortools-cp-sat",
            "graph_engine": "networkx",
            "serial_execution": True,
            "automatic_parallel_execution": False,
            "maximum_stages": 8,
        }
        with patch.object(
            indirect,
            "_select_stages",
            return_value=fake_plan,
        ), patch.object(
            indirect,
            "_contradiction",
            return_value={"claims": []},
        ):
            result = finance_decision_analysis(inputs)
        self.assertEqual(result["mode"], indirect.MODE)
        self.assertEqual(result["runtime_registration"], "controlled-preview-overlay")
        self.assertTrue(result["decision_support_only"])
        self.assertEqual(result["external_data_fetches"], 0)


if __name__ == "__main__":
    unittest.main()
