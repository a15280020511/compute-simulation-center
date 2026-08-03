from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behavior_finance_intelligence_operations import HANDLERS  # noqa: E402


class BehaviorFinanceIntelligenceGovernanceTests(unittest.TestCase):
    def test_fixed_catalog(self) -> None:
        self.assertEqual(len(HANDLERS), 12)
        self.assertEqual(
            set(HANDLERS),
            {
                "quantlib_option_greeks",
                "quantlib_bond_duration",
                "active_inference_policy_choice",
                "pyod_anomaly_screen",
                "market_basket_association_rules",
                "replicator_dynamics",
                "finite_population_fixation",
                "prospect_theory_choice",
                "collective_action_threshold",
                "rumor_correction_dynamics",
                "trust_reputation_update",
                "group_consensus_pressure",
            },
        )

    def test_source_has_no_network_dynamic_execution_or_agent_loop(self) -> None:
        source = (ROOT / "behavior_finance_intelligence_operations.py").read_text(encoding="utf-8")
        for forbidden in (
            "import requests",
            "import socket",
            "urllib.request",
            "subprocess.",
            "pickle.loads",
            "sample_action(",
            "from pyod.models.adengine",
            "from pyod.models.agent",
            "import mcp",
            "MCPServer",
        ):
            self.assertNotIn(forbidden, source)
        tree = compile(source, str(ROOT / "behavior_finance_intelligence_operations.py"), "exec", flags=0, dont_inherit=True)
        self.assertIsNotNone(tree)

    def test_evolutionary_and_behavior_modes_are_reproducible(self) -> None:
        first = HANDLERS["replicator_dynamics"](
            {
                "payoff_matrix": [[3.0, 0.0], [5.0, 1.0]],
                "initial_population": [0.5, 0.5],
                "steps": 20,
                "dt": 0.05,
            }
        )
        second = HANDLERS["replicator_dynamics"](
            {
                "payoff_matrix": [[3.0, 0.0], [5.0, 1.0]],
                "initial_population": [0.5, 0.5],
                "steps": 20,
                "dt": 0.05,
            }
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first["final_population"]), 1.0)

        prospect = HANDLERS["prospect_theory_choice"](
            {
                "options": [
                    {"name": "certain", "outcomes": [40.0], "probabilities": [1.0]},
                    {"name": "risky", "outcomes": [100.0, 0.0], "probabilities": [0.5, 0.5]},
                ]
            }
        )
        self.assertFalse(prospect["individual_prediction_allowed"])

    def test_collective_trust_and_correction_modes_are_bounded(self) -> None:
        cascade = HANDLERS["collective_action_threshold"](
            {
                "thresholds": [0.1, 0.2, 0.4],
                "initial_adopters": [True, False, False],
                "steps": 10,
            }
        )
        self.assertGreaterEqual(cascade["final_fraction"], cascade["initial_fraction"])
        self.assertFalse(cascade["real_group_prediction_allowed"])

        correction = HANDLERS["rumor_correction_dynamics"](
            {
                "initial_rumor": 0.1,
                "initial_correction": 0.1,
                "rumor_spread_rate": 1.0,
                "correction_spread_rate": 1.0,
                "correction_conversion_rate": 1.0,
                "steps": 20,
                "dt": 0.05,
            }
        )
        self.assertFalse(correction["persuasion_targeting_allowed"])
        for row in correction["trajectory"]:
            self.assertAlmostEqual(row["uninformed"] + row["rumor"] + row["correction"], 1.0, places=9)

        trust = HANDLERS["trust_reputation_update"](
            {
                "actors": ["A", "B"],
                "initial_trust": [[1.0, 0.5], [0.5, 1.0]],
                "events": [{"source": "A", "target": "B", "outcome": 1.0}],
            }
        )
        self.assertFalse(trust["individual_scoring_for_enforcement_allowed"])

    def test_external_modes_are_isolated_when_installed(self) -> None:
        cases = {
            "QuantLib": "quantlib_option_greeks",
            "pymdp": "active_inference_policy_choice",
            "pyod": "pyod_anomaly_screen",
            "mlxtend": "market_basket_association_rules",
        }
        availability = {name: importlib.util.find_spec(name) is not None for name in cases}
        self.assertEqual(set(availability), set(cases))


if __name__ == "__main__":
    unittest.main()
