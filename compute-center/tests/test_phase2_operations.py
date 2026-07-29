from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _missing_data_engine_available() -> bool:
    try:
        return version("statsmodels") == "0.14.6" and version("pandas") == "3.0.3"
    except PackageNotFoundError:
        return False


MISSING_DATA_AVAILABLE = _missing_data_engine_available()

dispatch_spec = importlib.util.spec_from_file_location("compute_dispatch_phase2", ROOT / "compute_dispatch.py")
assert dispatch_spec and dispatch_spec.loader
dispatch = importlib.util.module_from_spec(dispatch_spec)
sys.modules["compute_dispatch_phase2"] = dispatch
dispatch_spec.loader.exec_module(dispatch)


class Phase2OperationTests(unittest.TestCase):
    def execute(self, operation: str, inputs: dict) -> dict:
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        return dispatch.run_ticket({"task_id": f"phase2-{operation}-001", "operation": operation, "inputs": inputs}, Path(directory.name))["results"]

    @unittest.skipUnless(MISSING_DATA_AVAILABLE, "Pinned missing-data optional engine is not installed in the core environment")
    def test_missingness_and_mice(self) -> None:
        base = {"columns": ["x", "y"], "data": [[1, 2], [2, None], [3, 6], [4, None], [5, 10]]}
        profile = self.execute("missing_data_analysis", {"mode": "missingness_profile", **base})
        self.assertEqual(profile["missing_cell_count"], 2)
        mice = self.execute("missing_data_analysis", {"mode": "mice_multiple_imputation", "imputations": 3, "burn_in": 2, "seed": 7, **base})
        self.assertEqual(len(mice["imputation_records"]), 2)
        self.assertFalse(mice["original_data_overwritten"])

    def test_all_system_dynamics_modes(self) -> None:
        fixtures = {
            "stock_flow": {"stocks": [{"name": "stock", "initial": 10, "inflow": 2, "outflow_rate": 0.1, "capacity": 100}]},
            "feedback_delay": {"initial_state": 10, "exogenous_input": 1, "decay_rate": 0.1, "feedback_gain": 0.01, "delay_steps": 2},
            "policy_switch": {"initial_state": 10, "capacity": 100, "growth_rate_before": 0.1, "growth_rate_after": 0.03, "switch_step": 5},
            "coupled_capacity": {"initial_demand": 20, "initial_capacity": 15, "demand_growth": 0.01, "capacity_addition": 1, "service_rate": 0.8},
            "resource_depletion": {"initial_stock": 80, "carrying_capacity": 100, "regeneration_rate": 0.05, "extraction": 2},
            "adoption_saturation": {"initial_adoption": 0.05, "innovation_rate": 0.01, "imitation_rate": 0.2},
        }
        for mode, values in fixtures.items():
            result = self.execute("system_dynamics_simulation", {"mode": mode, "steps": 12, "dt": 1, **values})
            self.assertEqual(result["mode"], mode)

    def test_all_crisis_modes(self) -> None:
        fixtures = {
            "composite_risk_index": {"indicators": [{"name": "rain", "value": 8, "minimum": 0, "maximum": 10, "weight": 2}, {"name": "capacity", "value": 3, "minimum": 0, "maximum": 10, "weight": 1, "direction": "lower_risk"}]},
            "change_point_warning": {"values": [0, 0.1, -0.1, 0.05, 0, 0.1, -0.1, 0, 0.05, -0.05, 2.0, 2.2], "baseline_window": 10, "threshold": 3},
            "alert_threshold_optimization": {"probabilities": [0.1, 0.8, 0.4, 0.9], "outcomes": [0, 1, 0, 1], "false_positive_cost": 1, "false_negative_cost": 5},
            "scenario_escalation": {"transition_matrix": [[0.8, 0.2], [0.1, 0.9]], "initial_distribution": [1, 0], "steps": 5, "severe_states": [1]},
            "response_resource_allocation": {"available_resource": 10, "demands": [{"name": "a", "need": 8, "priority": 2}, {"name": "b", "need": 8, "priority": 1}]},
            "warning_performance_evaluation": {"probabilities": [0.1, 0.8, 0.4, 0.9], "outcomes": [0, 1, 0, 1], "threshold": 0.5, "lead_times": [0, 3, 0, 5]},
        }
        for mode, values in fixtures.items():
            result = self.execute("crisis_early_warning", {"mode": mode, **values})
            self.assertEqual(result["mode"], mode)

    def test_runtime_governance_and_constraints_are_artifacted(self) -> None:
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        ticket = {"task_id": "phase2-governance-001", "operation": "system_dynamics_simulation", "inputs": {"mode": "adoption_saturation", "steps": 5, "initial_adoption": 0.1, "innovation_rate": 0.01, "imitation_rate": 0.2}, "constraint_profile": {"hard_constraints": [{"id": "adoption-bound", "type": "probability", "field": "results.final_adoption"}], "independent_post_check": True}}
        result = dispatch.run_ticket(ticket, root)
        self.assertEqual(result["constraint_assurance"]["post_execution"]["status"], "PASS")
        for name in ("compute-model-governance.json", "compute-constraint-precheck.json", "compute-constraint-postcheck.json", "compute-calibration-assurance.json"):
            self.assertTrue((root / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
