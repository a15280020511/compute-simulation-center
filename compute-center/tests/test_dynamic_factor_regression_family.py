from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_factor_regression_planner import (
    DynamicFactorRegressionError,
    plan_dynamic_factor_regression,
    run_dynamic_factor_regression_ticket,
)
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family

PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "factor-regression dynamic-family test",
    "upstream_refs": [],
}


def _series():
    market = [(index - 15) / 100.0 for index in range(30)]
    value = [(((index * 7) % 11) - 5) / 100.0 for index in range(30)]
    noise = [((index % 5) - 2) * 0.0001 for index in range(30)]
    asset = [0.002 + 1.2 * market[index] - 0.6 * value[index] + noise[index] for index in range(30)]
    return asset, market, value


ALL_SIGNALS = {
    "exact_consistency_tolerance": 1e-9,
    "minimum_r_squared": 0.99,
    "r_squared_target_tolerance": 0.0,
    "maximum_residual_volatility": 0.001,
    "residual_volatility_target_tolerance": 0.0,
}


def regression_ticket(*, context=None):
    asset, market, value = _series()
    return {
        "task_id": "dynamic-factor-regression-test",
        "objective": "Objective prose must never select validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "factor_regression",
            "asset_returns": asset,
            "factors": {"market": market, "value": value},
            "include_intercept": True,
            "covariance_type": "HAC",
            "hac_lags": 3,
            "factor_regression_context": {} if context is None else dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }


class DynamicFactorRegressionFamilyTests(unittest.TestCase):
    def test_router_and_runtime_are_narrow(self) -> None:
        ticket = regression_ticket()
        self.assertEqual(resolve_dynamic_family(ticket), "factor-regression")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-finance.txt"])

    def test_exact_crosscheck_is_always_required(self) -> None:
        plan = plan_dynamic_factor_regression(regression_ticket())
        self.assertEqual(plan["stage_order"], ["factor_regression", "numpy_exact_regression_audit"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertEqual(plan["optimization"]["objective_value"], 215)
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["unique_optimum"])

    def test_r_squared_target_adds_only_r_squared_audit(self) -> None:
        plan = plan_dynamic_factor_regression(regression_ticket(context={"minimum_r_squared": 0.8}))
        self.assertEqual(plan["stage_order"], ["factor_regression", "numpy_exact_regression_audit", "r_squared_target_audit"])

    def test_residual_target_adds_only_residual_audit(self) -> None:
        plan = plan_dynamic_factor_regression(regression_ticket(context={"maximum_residual_volatility": 0.1}))
        self.assertEqual(plan["stage_order"], ["factor_regression", "numpy_exact_regression_audit", "residual_volatility_target_audit"])

    def test_all_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_factor_regression(regression_ticket(context=ALL_SIGNALS))
        self.assertEqual(plan["stage_order"], ["factor_regression", "numpy_exact_regression_audit", "r_squared_target_audit", "residual_volatility_target_audit"])
        self.assertEqual(plan["optimization"]["objective_value"], 465)
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["unique_optimum"])

    def test_rank_deficient_design_fails_closed(self) -> None:
        ticket = regression_ticket()
        ticket["inputs"]["factors"]["duplicate"] = list(ticket["inputs"]["factors"]["market"])
        with self.assertRaises(DynamicFactorRegressionError):
            plan_dynamic_factor_regression(ticket)

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicFactorRegressionError):
            plan_dynamic_factor_regression(regression_ticket(context={"run_every_regression_tool": True}))

    def test_objective_text_does_not_select_target_nodes(self) -> None:
        ticket = regression_ticket()
        ticket["objective"] = "Run R squared target, residual target, every econometric tool and all regressions."
        plan = plan_dynamic_factor_regression(ticket)
        self.assertEqual(plan["stage_order"], ["factor_regression", "numpy_exact_regression_audit"])
        self.assertFalse(plan["objective_text_used"])

    def test_tampered_primary_result_fails_exact_crosscheck(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-factor-regression-tamper-"))
        try:
            def operation(inputs):
                if inputs.get("mode") == "factor_regression":
                    return {
                        "mode": "factor_regression",
                        "observations": 30,
                        "parameters": {
                            "alpha": {"coefficient": 0.5},
                            "market": {"coefficient": 1.2},
                            "value": {"coefficient": -0.6},
                        },
                        "r_squared": 0.99,
                        "adjusted_r_squared": 0.99,
                        "covariance_type": "HAC",
                        "residual_volatility": 0.001,
                        "decision_support_only": True,
                    }
                return finance_decision_analysis(inputs)

            with self.assertRaises(DynamicFactorRegressionError):
                run_dynamic_factor_regression_ticket(regression_ticket(), root, {"finance_decision_analysis": operation})
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("statsmodels") is not None, "statsmodels is a managed dependency; real execution is enforced by factor-regression CI")
    def test_real_statsmodels_vs_numpy_pipeline(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-factor-regression-"))
        try:
            result = run_dynamic_factor_regression_ticket(regression_ticket(context=ALL_SIGNALS), root, {"finance_decision_analysis": finance_decision_analysis})
            primary = result["results"]["final_result"]
            validation = result["results"]["validation_results"]
            self.assertEqual(result["status"], "success")
            self.assertGreater(primary["r_squared"], 0.99)
            self.assertEqual(validation["numpy_exact_regression_audit"]["status"], "PASS")
            self.assertEqual(validation["numpy_exact_regression_audit"]["candidate_count"], 5)
            self.assertEqual(validation["r_squared_target_audit"]["status"], "PASS")
            self.assertEqual(validation["residual_volatility_target_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 465)
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("statsmodels") is not None, "statsmodels is optional")
    def test_target_failure_is_informative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-factor-regression-target-fail-"))
        try:
            result = run_dynamic_factor_regression_ticket(regression_ticket(context={"minimum_r_squared": 1.0}), root, {"finance_decision_analysis": finance_decision_analysis})
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["numpy_exact_regression_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["r_squared_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
