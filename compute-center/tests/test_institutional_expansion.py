from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capability_manager import load_institutional_expansion, requirements_for_ticket, runtime_plan  # noqa: E402
from institutional_expansion_operations import (  # noqa: E402
    HANDLERS as INSTITUTIONAL_HANDLERS,
    deflated_sharpe_gate,
    transaction_cost_capacity,
)
from personal_finance_operations import HANDLERS as PERSONAL_FINANCE_HANDLERS  # noqa: E402


class InstitutionalExpansionRegistryTests(unittest.TestCase):
    def test_registry_is_complete_and_offline(self) -> None:
        registry = load_institutional_expansion()
        governed_handlers = set(INSTITUTIONAL_HANDLERS)
        self.assertEqual(set(registry["modes"]), governed_handlers)
        self.assertEqual(set(registry["mode_requirements"]), governed_handlers)
        self.assertEqual(registry["network_policy"], "deny")
        self.assertFalse(registry["arbitrary_code_allowed"])
        self.assertEqual(len(INSTITUTIONAL_HANDLERS), 19)
        self.assertEqual(len(PERSONAL_FINANCE_HANDLERS), 9)
        self.assertTrue(set(PERSONAL_FINANCE_HANDLERS) <= governed_handlers)

    def test_mode_specific_requirements_and_runtime_plan(self) -> None:
        cases = {
            "causal_pc_discovery": "requirements-institutional-causal.txt",
            "tigramite_pcmci_discovery": "requirements-institutional-tigramite.txt",
            "evidently_data_drift": "requirements-institutional-evidently.txt",
            "river_adwin_drift": "requirements-institutional-river.txt",
            "skfolio_walk_forward_portfolio": "requirements-institutional-skfolio.txt",
            "black_litterman_allocation": "requirements-finance.txt",
        }
        for mode, filename in cases.items():
            ticket = {"operation": "finance_decision_analysis", "inputs": {"mode": mode}}
            requirements = requirements_for_ticket(ticket)
            self.assertEqual(len(requirements), 1)
            self.assertTrue(requirements[0].endswith(filename))
            plan = runtime_plan(ticket)
            self.assertEqual(plan["capability_pack"], "institutional-expansion")
            self.assertEqual(plan["network_policy"], "deny")
            self.assertEqual(plan["maturity"], "controlled-preview")
            self.assertFalse(plan["arbitrary_code_allowed"])
        native_modes = {
            "deflated_sharpe_gate",
            "transaction_cost_capacity",
            *PERSONAL_FINANCE_HANDLERS.keys(),
        }
        for mode in native_modes:
            ticket = {"operation": "finance_decision_analysis", "inputs": {"mode": mode}}
            self.assertEqual(requirements_for_ticket(ticket), [])
            plan = runtime_plan(ticket)
            self.assertEqual(plan["maturity"], "controlled-preview")
            self.assertEqual(plan["network_policy"], "deny")


class NativeRobustFinanceTests(unittest.TestCase):
    def test_deflated_sharpe_gate_is_bounded(self) -> None:
        rng = np.random.default_rng(5)
        result = deflated_sharpe_gate({
            "returns": rng.normal(0.001, 0.01, size=300).tolist(),
            "strategy_trials": 25,
            "trial_sharpe_standard_deviation": 0.2,
        })
        self.assertGreaterEqual(result["deflated_sharpe_probability"], 0.0)
        self.assertLessEqual(result["deflated_sharpe_probability"], 1.0)
        self.assertFalse(result["brokerage_execution"])

    def test_transaction_costs_reduce_gross_return(self) -> None:
        returns = np.tile([0.001, 0.0005], (40, 1))
        weights = np.tile([0.5, 0.5], (40, 1))
        weights[20:, :] = [0.8, 0.2]
        result = transaction_cost_capacity({
            "asset_returns": returns.tolist(),
            "weights": weights.tolist(),
            "capital": 1_000_000.0,
            "average_daily_volume": [50_000_000.0, 50_000_000.0],
            "commission_bps": 2.0,
            "spread_bps": 4.0,
            "impact_coefficient": 0.01,
        })
        self.assertLess(result["net_cumulative_return"], result["gross_cumulative_return"])
        self.assertIn(result["capacity_status"], {"PASS", "BREACH"})


if __name__ == "__main__":
    unittest.main()
