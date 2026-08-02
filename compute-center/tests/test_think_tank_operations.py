#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capability_manager import requirements_for_ticket, runtime_plan
from think_tank_business_operations import customer_lifetime_value, inventory_policy, process_capability
from think_tank_decision_operations import influence_diagram, policy_microsimulation, strategic_sandbox
from think_tank_operations import SUPPORTED_MODES
from think_tank_registry_validate import validate


class ThinkTankRegistryTests(unittest.TestCase):
    def test_extension_registry_and_catalog_are_consistent(self) -> None:
        result = validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["extension_modes"], 38)
        self.assertEqual(result["effective_managed_modes"], 118)
        self.assertEqual(len(SUPPORTED_MODES), 38)

    def test_mode_specific_dependency_resolution(self) -> None:
        cases = {
            "bounded_table_profile": "requirements-thinktank-data.txt",
            "panel_fixed_effects": "requirements-thinktank-econometrics.txt",
            "cvar_portfolio": "requirements-thinktank-finance.txt",
            "multiobjective_pareto": "requirements-thinktank-decision.txt",
            "hierarchical_bayesian_mean": "requirements-thinktank-bayesian.txt",
            "raster_zonal_statistics": "requirements-thinktank-geospatial.txt",
        }
        for mode, filename in cases.items():
            ticket = {"operation": "finance_decision_analysis", "inputs": {"mode": mode}}
            requirements = requirements_for_ticket(ticket)
            self.assertEqual(len(requirements), 1)
            self.assertTrue(requirements[0].endswith(filename))
            plan = runtime_plan(ticket)
            self.assertEqual(plan["network_policy"], "deny")
            self.assertEqual(plan["maturity"], "controlled-preview")
            self.assertFalse(plan["arbitrary_code_allowed"])


class ThinkTankPureModeTests(unittest.TestCase):
    def test_customer_lifetime_value(self) -> None:
        result = customer_lifetime_value(
            {
                "period_margin": 100.0,
                "retention_rate": 0.8,
                "discount_rate": 0.01,
                "acquisition_cost": 200.0,
                "periods": 12,
            }
        )
        self.assertEqual(result["mode"], "customer_lifetime_value")
        self.assertEqual(len(result["cashflows"]), 12)
        self.assertGreater(result["gross_present_value"], result["net_lifetime_value"])

    def test_inventory_policy(self) -> None:
        result = inventory_policy(
            {
                "annual_demand": 10_000.0,
                "order_cost": 100.0,
                "holding_cost_per_unit": 5.0,
                "lead_time_demand_mean": 500.0,
                "lead_time_demand_sd": 50.0,
                "service_level": 0.95,
            }
        )
        self.assertGreater(result["economic_order_quantity"], 0)
        self.assertGreater(result["reorder_point"], 500.0)

    def test_process_capability(self) -> None:
        result = process_capability(
            {
                "values": [9.8, 10.0, 10.1, 10.2, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2],
                "lower_specification": 9.0,
                "upper_specification": 11.0,
            }
        )
        self.assertGreater(result["cp"], 1.0)
        self.assertGreater(result["cpk"], 1.0)

    def test_influence_diagram(self) -> None:
        result = influence_diagram(
            {
                "actions": ["invest", "wait"],
                "states": ["growth", "recession"],
                "state_probabilities": [0.7, 0.3],
                "utilities": [[10.0, -8.0], [3.0, 2.0]],
            }
        )
        self.assertEqual(result["best_action"], "invest")
        self.assertGreaterEqual(result["perfect_information_value"], 0.0)

    def test_policy_microsimulation(self) -> None:
        result = policy_microsimulation(
            {
                "incomes": [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000],
                "tax_brackets": [
                    {"threshold": 3_000, "rate": 0.05},
                    {"threshold": 8_000, "rate": 0.10},
                ],
                "universal_transfer": 200,
                "poverty_line": 2_500,
            }
        )
        self.assertEqual(result["population"], 10)
        self.assertEqual(len(result["individual_results"]), 10)
        self.assertLessEqual(result["gini_after"], result["gini_before"])

    def test_strategic_sandbox_is_reproducible(self) -> None:
        inputs = {
            "actors": ["a", "b", "c"],
            "payoff_matrix": [[1.0, 0.2, -0.1], [0.0, 1.2, 0.1], [0.2, -0.2, 1.0]],
            "initial_resources": [1.0, 1.0, 1.0],
            "periods": 20,
            "adaptation_rate": 0.2,
            "shock_standard_deviation": 0.01,
            "seed": 42,
        }
        left = strategic_sandbox(inputs)
        right = strategic_sandbox(inputs)
        self.assertEqual(left["final_shares"], right["final_shares"])
        self.assertAlmostEqual(sum(left["final_shares"]), 1.0)


if __name__ == "__main__":
    unittest.main()
