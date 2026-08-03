from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tool_registry  # noqa: E402


class BehaviorFinanceRegistryTests(unittest.TestCase):
    def test_mode_specific_dependency_resolution(self) -> None:
        cases = {
            "quantlib_option_greeks": "requirements-final-quantlib.txt",
            "quantlib_bond_duration": "requirements-final-quantlib.txt",
            "active_inference_policy_choice": "requirements-final-pymdp.txt",
            "pyod_anomaly_screen": "requirements-final-pyod.txt",
            "market_basket_association_rules": "requirements-final-mlxtend.txt",
        }
        for mode, expected in cases.items():
            observed = tool_registry.requirement_files_for_ticket(
                {"operation": "strategic_policy_analysis", "inputs": {"mode": mode}}
            )
            self.assertEqual([Path(item).name for item in observed], [expected])

    def test_native_modes_require_no_extra_dependency(self) -> None:
        for mode in (
            "replicator_dynamics",
            "finite_population_fixation",
            "prospect_theory_choice",
            "collective_action_threshold",
            "rumor_correction_dynamics",
            "trust_reputation_update",
            "group_consensus_pressure",
        ):
            observed = tool_registry.requirement_files_for_ticket(
                {"operation": "strategic_policy_analysis", "inputs": {"mode": mode}}
            )
            self.assertEqual(observed, [])


if __name__ == "__main__":
    unittest.main()
