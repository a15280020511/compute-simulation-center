from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assumption_library import assess_assumptions, load_library
from credibility_engine import build_credibility_case
from experiment_assurance import assess_experiment


class AssumptionCredibilityGovernanceTests(unittest.TestCase):
    def assumption(self, assumption_id="demand-growth-001"):
        return {
            "assumption_id": assumption_id,
            "type": "parameter",
            "statement": "Demand growth is uncertain.",
            "source_type": "historical",
            "basis": "Frozen historical snapshot.",
            "confidence": "medium",
            "minimum": 0.0,
            "maximum": 0.2,
            "distribution": "uniform",
            "falsification_test": "Compare with realized demand.",
            "invalid_when": "Observed growth is outside the registered interval.",
            "sensitivity_rank": 1,
            "sensitivity_required": True,
            "calibration_status": "calibrated",
            "uncertainty_type": "epistemic",
            "criticality": "high",
            "evidence_strength": "moderate",
            "evidence_sha256": "a" * 64,
            "status": "approved",
        }

    def test_library_is_empty_by_design_and_valid(self):
        library = load_library()
        self.assertFalse(library["policy"]["domain_assumptions_may_be_prepopulated"])
        self.assertEqual(library["assumptions"], [])

    def test_assumption_risk_and_snapshot_are_deterministic(self):
        ticket = {"operation": "monte_carlo", "quality_profile": {"decision_class": "formal"}, "assumption_register": [self.assumption()]}
        first = assess_assumptions(ticket)
        second = assess_assumptions(ticket)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(len(first["resolved_snapshot_sha256"]), 64)
        self.assertEqual(first["risk_ranking"][0]["assumption_id"], "demand-growth-001")

    def test_dependency_cycle_blocks(self):
        a = self.assumption("a-001")
        b = self.assumption("b-001")
        a["dependencies"] = ["b-001"]
        b["dependencies"] = ["a-001"]
        report = assess_assumptions({"operation": "monte_carlo", "assumption_register": [a, b]})
        self.assertEqual(report["status"], "BLOCKED")

    def test_high_stakes_stochastic_requires_experiment_profile(self):
        report = assess_experiment({"operation": "monte_carlo", "quality_profile": {"decision_class": "high_stakes"}, "inputs": {"seed": 1}})
        self.assertEqual(report["status"], "BLOCKED")

    def test_high_stakes_credibility_has_no_magic_score_and_requires_typed_evidence(self):
        assumption = assess_assumptions({"operation": "descriptive_statistics", "quality_profile": {"decision_class": "high_stakes"}, "assumption_register": [self.assumption()]})
        experiment = {"status": "NOT_REQUIRED"}
        factor_ids = [
            "intended_use", "conceptual_model", "data_pedigree", "assumption_governance",
            "implementation_verification", "solution_verification", "empirical_validation",
            "uncertainty_characterization", "results_robustness", "input_pedigree",
            "technical_review", "process_management",
        ]
        profile = {
            "use_statement": "Support a bounded high-stakes descriptive decision.",
            "model_influence": "high",
            "decision_consequence": "high",
            "factor_levels": {key: 4 for key in factor_ids},
            "evidence": [{
                "factor_ids": factor_ids,
                "sha256": "b" * 64,
                "source": "independent operational credibility evidence",
                "evidence_type": "operational_feedback",
                "independent": True,
            }],
        }
        case = build_credibility_case(
            {"operation": "descriptive_statistics", "objective": "bounded use", "quality_profile": {"decision_class": "high_stakes"}, "credibility_profile": profile},
            {"model_id": "descriptive_statistics-registered-v1", "version": "1.0.0"},
            assumption,
            experiment,
        )
        self.assertEqual(case["status"], "PASS")
        self.assertIsNone(case["single_weighted_score"])
        self.assertTrue(case["declared_levels_are_targets_only"])

    def test_high_stakes_self_declared_levels_do_not_pass_without_typed_evidence(self):
        factor_ids = [
            "intended_use", "conceptual_model", "data_pedigree", "assumption_governance",
            "implementation_verification", "solution_verification", "empirical_validation",
            "uncertainty_characterization", "results_robustness", "input_pedigree",
            "technical_review", "process_management",
        ]
        assumption = assess_assumptions({"operation": "descriptive_statistics", "quality_profile": {"decision_class": "high_stakes"}, "assumption_register": [self.assumption()]})
        case = build_credibility_case(
            {
                "operation": "descriptive_statistics",
                "objective": "bounded use",
                "quality_profile": {"decision_class": "high_stakes"},
                "credibility_profile": {
                    "use_statement": "Self-declared target only.",
                    "model_influence": "high",
                    "decision_consequence": "high",
                    "factor_levels": {key: 4 for key in factor_ids},
                    "evidence": [],
                },
            },
            {"model_id": "descriptive_statistics-registered-v1", "version": "1.0.0"},
            assumption,
            {"status": "NOT_REQUIRED"},
        )
        self.assertEqual(case["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
