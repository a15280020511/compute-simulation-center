from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assumption_runtime import build_assumption_plan  # noqa: E402


class AssumptionDistributionCompletionTests(unittest.TestCase):
    def _ticket(self, assumption: dict) -> dict:
        return {
            "task_id": "assumption-completion-001",
            "operation": "monte_carlo",
            "quality_profile": {"decision_class": "formal"},
            "data_context": {
                "variables": [
                    {
                        "name": "tail_loss",
                        "missing": True,
                        "confidence": "low",
                        "source_type": "proxy",
                    }
                ]
            },
            "assumption_register": [assumption],
        }

    def test_parametric_distribution_is_accepted_when_complete(self) -> None:
        plan = build_assumption_plan(self._ticket({
            "assumption_id": "tail_loss",
            "linked_parameter": "tail_loss",
            "status": "approved",
            "basis": "Frozen historical tail sample.",
            "invalid_when": "Tail index changes beyond the declared monitoring band.",
            "distribution": "student_t",
            "distribution_parameters": {
                "degrees_of_freedom": 5.0,
                "location": 0.0,
                "scale": 1.0,
            },
            "dependence_model": "independent",
        }))
        self.assertEqual(plan["status"], "PASS")
        self.assertEqual(plan["approved_assumption_count"], 1)

    def test_copula_requires_correlation_group(self) -> None:
        incomplete = build_assumption_plan(self._ticket({
            "assumption_id": "tail_loss",
            "linked_parameter": "tail_loss",
            "status": "approved",
            "basis": "Frozen historical tail sample.",
            "invalid_when": "Dependence changes.",
            "distribution": "student_t",
            "distribution_parameters": {
                "degrees_of_freedom": 5.0,
                "location": 0.0,
                "scale": 1.0,
            },
            "dependence_model": "t_copula",
        }))
        self.assertEqual(incomplete["status"], "BLOCKED")
        complete_assumption = {
            "assumption_id": "tail_loss",
            "linked_parameter": "tail_loss",
            "status": "approved",
            "basis": "Frozen historical tail sample.",
            "invalid_when": "Dependence changes.",
            "distribution": "student_t",
            "distribution_parameters": {
                "degrees_of_freedom": 5.0,
                "location": 0.0,
                "scale": 1.0,
            },
            "dependence_model": "t_copula",
            "correlation_group": "market-tail-group",
        }
        complete = build_assumption_plan(self._ticket(complete_assumption))
        self.assertEqual(complete["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
