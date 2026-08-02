#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capability_manager import load_registry, requirements_for_ticket, runtime_plan
from decision_intelligence_gateway import ALL_SUPPORTED_MODES, INSTITUTIONAL_PREVIEW_MODES
from institutional_toolkit_operations import HANDLERS
from institutional_toolkit_registry_validate import main as validate_registry


class InstitutionalToolkitContractTests(unittest.TestCase):
    def test_registry_and_handlers_are_complete_and_disjoint(self) -> None:
        self.assertEqual(validate_registry(), 0)
        self.assertEqual(len(HANDLERS), 41)
        self.assertEqual(set(HANDLERS), INSTITUTIONAL_PREVIEW_MODES)
        self.assertTrue(set(HANDLERS).issubset(set(ALL_SUPPORTED_MODES)))
        registry = load_registry()
        target = next(group for group in registry["groups"] if group.get("id") == "decision-intelligence")
        self.assertTrue(set(HANDLERS).issubset(set(target.get("modes") or {})))
        effective_count = sum(len(group.get("modes") or {}) for group in registry["groups"])
        self.assertEqual(effective_count, 148)

    def test_mode_specific_dependency_resolution(self) -> None:
        cases = {
            "double_machine_learning": "requirements-institutional-economics.txt",
            "garch_volatility": "requirements-institutional-forecasting.txt",
            "comprehensive_mcda": "requirements-institutional-decision.txt",
            "spatial_lag_regression": "requirements-institutional-spatial.txt",
            "energy_system_dispatch": "requirements-institutional-energy.txt",
            "water_network_resilience": "requirements-institutional-climate-health.txt",
            "european_option_pricing": "requirements-institutional-finance.txt",
            "shacl_graph_validation": "requirements-institutional-knowledge.txt",
            "job_shop_schedule": "requirements-institutional-engineering.txt",
            "fairness_metric_audit": "requirements-institutional-assurance.txt",
        }
        for mode, filename in cases.items():
            ticket = {"operation": "finance_decision_analysis", "inputs": {"mode": mode}}
            requirements = requirements_for_ticket(ticket)
            self.assertEqual(len(requirements), 1)
            self.assertTrue(requirements[0].endswith(filename))
            plan = runtime_plan(ticket)
            self.assertEqual(plan["network_policy"], "deny")
            self.assertEqual(plan["maturity"], "controlled-preview")
            self.assertTrue(plan["deterministic"])
            self.assertFalse(plan["arbitrary_code_allowed"])
            self.assertFalse(plan["arbitrary_requirements_allowed"])

    def test_import_does_not_load_optional_engines(self) -> None:
        forbidden = {
            "pyfixest", "doubleml", "econml", "semopy", "pyblp", "statsforecast",
            "hierarchicalforecast", "arch", "pyod", "pyextremes", "xskillscore",
            "ema_workbench", "pymcdm", "nashpy", "geopandas", "mgwr", "momepy",
            "spreg", "spopt", "movingpandas", "segregation", "pypsa", "pandapower",
            "wntr", "xclim", "starsim", "QuantLib", "pyvinecopulib", "splink",
            "rapidfuzz", "pyshacl", "rdflib", "datasketch", "control", "reliability",
            "stockpyl", "ciw", "job_shop_lib", "fairlearn", "cleanlab", "shap", "copulas",
        }
        self.assertFalse(forbidden & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
