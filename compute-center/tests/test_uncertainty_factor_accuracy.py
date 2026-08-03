from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_runner import ComputeError  # noqa: E402
from uncertainty_factor_accuracy_operations import (  # noqa: E402
    HANDLERS,
    bayesian_linear_calibration,
    cross_validation_plan,
    distribution_fit_select,
    factor_information_analysis,
    factor_overfit_diagnostics,
    factor_selection_diagnostics,
    forecast_accuracy,
    joint_random_sample,
    probabilistic_accuracy,
    reliability_analysis,
    variable_role_validate,
)


class RegistryTests(unittest.TestCase):
    def test_distribution_and_variable_role_registries_are_extended(self) -> None:
        distributions = json.loads((ROOT / "distribution-registry.json").read_text(encoding="utf-8"))
        identifiers = {row["id"] for row in distributions["distributions"]}
        self.assertTrue({
            "bernoulli", "binomial", "poisson", "negative_binomial", "gamma",
            "exponential", "weibull", "student_t", "pareto", "gev", "gpd",
            "truncated_normal", "zero_inflated_poisson", "hurdle_poisson",
            "gaussian_mixture",
        } <= identifiers)
        self.assertEqual(
            {row["id"] for row in distributions["dependence_models"]},
            {"independent", "gaussian_copula", "t_copula"},
        )
        roles = json.loads((ROOT / "variable-role-registry.json").read_text(encoding="utf-8"))
        role_ids = {row["id"] for row in roles["roles"]}
        self.assertTrue({
            "target", "outcome", "exogenous", "endogenous", "control", "decision",
            "state", "latent", "nuisance", "constraint", "mediator", "moderator",
            "confounder", "instrument", "treatment", "exposure",
        } <= role_ids)

    def test_pack_has_exact_allowlisted_modes(self) -> None:
        self.assertEqual(len(HANDLERS), 11)
        self.assertEqual(set(HANDLERS), {
            "joint_random_sample",
            "distribution_fit_select",
            "variable_role_validate",
            "probabilistic_accuracy",
            "forecast_accuracy",
            "bayesian_linear_calibration",
            "reliability_analysis",
            "factor_information_analysis",
            "factor_selection_diagnostics",
            "factor_overfit_diagnostics",
            "cross_validation_plan",
        })


class RandomVariableTests(unittest.TestCase):
    def test_gaussian_copula_is_reproducible_and_correlated(self) -> None:
        request = {
            "variables": [
                {"name": "normal", "distribution": "normal", "parameters": {"mean": 0.0, "standard_deviation": 1.0}},
                {"name": "tail", "distribution": "student_t", "parameters": {"degrees_of_freedom": 5.0, "location": 0.0, "scale": 1.0}},
                {"name": "count", "distribution": "negative_binomial", "parameters": {"mean": 4.0, "dispersion": 2.0}},
            ],
            "sample_count": 2000,
            "seed": 17,
            "dependence": {
                "method": "gaussian_copula",
                "correlation_matrix": [[1.0, 0.7, 0.3], [0.7, 1.0, 0.2], [0.3, 0.2, 1.0]],
            },
        }
        first = joint_random_sample(request)
        second = joint_random_sample(request)
        self.assertEqual(first["sample_sha256"], second["sample_sha256"])
        self.assertGreater(first["empirical_correlation"][0][1], 0.5)
        self.assertFalse(first["network_used"])
        self.assertEqual(first["model_calls"], 0)

    def test_invalid_correlation_matrix_fails_closed(self) -> None:
        with self.assertRaises(ComputeError):
            joint_random_sample({
                "variables": [
                    {"name": "a", "distribution": "normal", "parameters": {"mean": 0.0, "standard_deviation": 1.0}},
                    {"name": "b", "distribution": "normal", "parameters": {"mean": 0.0, "standard_deviation": 1.0}},
                ],
                "sample_count": 100,
                "dependence": {"method": "gaussian_copula", "correlation_matrix": [[1.0, 1.5], [1.5, 1.0]]},
            })

    def test_distribution_selection_recovers_normal_family(self) -> None:
        values = np.random.default_rng(9).normal(2.0, 0.8, size=500)
        result = distribution_fit_select({
            "observations": values.tolist(),
            "candidates": ["normal", "student_t", "gev"],
        })
        self.assertEqual(result["selected_distribution"], "normal")
        self.assertEqual(len(result["fits"]), 3)


class VariableAndAccuracyTests(unittest.TestCase):
    def test_variable_role_graph_and_cycle_detection(self) -> None:
        result = variable_role_validate({
            "variables": [
                {"name": "income", "role": "exogenous", "dependencies": []},
                {"name": "policy", "role": "treatment", "dependencies": ["income"], "manipulable": True},
                {"name": "outcome", "role": "outcome", "dependencies": ["income", "policy"]},
            ]
        })
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["topological_order"].index("income"), result["topological_order"].index("outcome"))
        with self.assertRaises(ComputeError):
            variable_role_validate({
                "variables": [
                    {"name": "a", "role": "state", "dependencies": ["b"]},
                    {"name": "b", "role": "state", "dependencies": ["a"]},
                ]
            })

    def test_probability_metrics_include_calibration_and_classification(self) -> None:
        result = probabilistic_accuracy({
            "actual": [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            "probabilities": [0.05, 0.15, 0.85, 0.75, 0.95, 0.35, 0.65, 0.25, 0.55, 0.45],
            "bins": 5,
        })
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["pr_auc"], 1.0)
        self.assertLess(result["brier_score"], 0.15)
        self.assertGreater(result["threshold_metrics"]["matthews_correlation_coefficient"], 0.8)
        self.assertTrue(result["reliability_bins"])

    def test_forecast_metrics_intervals_quantiles_and_baseline(self) -> None:
        actual = np.arange(1.0, 31.0)
        predicted = actual + np.sin(actual) * 0.2
        result = forecast_accuracy({
            "actual": actual.tolist(),
            "predicted": predicted.tolist(),
            "baseline_predicted": (actual + 2.0).tolist(),
            "lower": (predicted - 1.0).tolist(),
            "upper": (predicted + 1.0).tolist(),
            "interval_alpha": 0.1,
            "quantile_predictions": {
                "0.1": (predicted - 0.8).tolist(),
                "0.5": predicted.tolist(),
                "0.9": (predicted + 0.8).tolist(),
            },
            "fold_ids": [str(index // 10) for index in range(30)],
        })
        self.assertLess(result["metrics"]["rmse"], 0.2)
        self.assertGreater(result["interval"]["coverage"], 0.9)
        self.assertLess(result["baseline_comparison"]["candidate_mean_squared_error"], result["baseline_comparison"]["baseline_mean_squared_error"])
        self.assertEqual(len(result["fold_metrics"]), 3)
        self.assertIsNotNone(result["quantile_crps_approximation"])


class CalibrationAndReliabilityTests(unittest.TestCase):
    def test_bayesian_calibration_recovers_linear_parameters(self) -> None:
        x = np.arange(60, dtype=float)
        y = 2.0 + 3.0 * x
        result = bayesian_linear_calibration({
            "features": x[:, None].tolist(),
            "observations": y.tolist(),
            "prediction_features": [[61.0], [62.0]],
        })
        self.assertAlmostEqual(result["posterior_mean"][0], 2.0, places=4)
        self.assertAlmostEqual(result["posterior_mean"][1], 3.0, places=6)
        self.assertLess(result["calibration_metrics"]["rmse"], 1e-4)
        self.assertEqual(len(result["prediction"]["mean"]), 2)

    def test_reliability_supports_form_and_monte_carlo(self) -> None:
        form = reliability_analysis({
            "method": "form_linear",
            "means": [0.0, 0.0],
            "covariance": [[1.0, 0.2], [0.2, 1.0]],
            "coefficients": [1.0, 1.0],
            "threshold": -1.0,
            "failure_when": "below",
        })
        self.assertGreater(form["failure_probability"], 0.0)
        self.assertLess(form["failure_probability"], 0.5)
        samples = np.linspace(-2.0, 3.0, 1000)
        monte_carlo = reliability_analysis({
            "method": "monte_carlo",
            "limit_state_values": samples.tolist(),
            "factors": np.column_stack([samples, samples**2]).tolist(),
            "factor_names": ["linear", "quadratic"],
        })
        self.assertEqual(monte_carlo["failure_count"], 400)
        self.assertAlmostEqual(monte_carlo["failure_probability"], 0.4)
        self.assertEqual(len(monte_carlo["failure_probability_wilson_95"]), 2)


class FactorResearchTests(unittest.TestCase):
    def test_factor_information_detects_signal_and_turnover(self) -> None:
        rng = np.random.default_rng(21)
        periods, assets = 24, 40
        signal = rng.normal(size=(periods, assets))
        noise = rng.normal(size=(periods, assets))
        returns = 0.35 * signal + 0.05 * noise
        result = factor_information_analysis({
            "forward_returns": returns.tolist(),
            "factors": {"signal": signal.tolist(), "noise": noise.tolist()},
            "quantiles": 5,
            "regimes": ["calm" if index < 12 else "stress" for index in range(periods)],
        })
        rows = {row["factor"]: row for row in result["factors"]}
        self.assertGreater(rows["signal"]["mean_rank_information_coefficient"], 0.8)
        self.assertGreater(rows["signal"]["quantile_monotonicity"], 0.8)
        self.assertGreaterEqual(rows["signal"]["top_quantile_turnover"], 0.0)
        self.assertIn("calm", rows["signal"]["regime_rank_ic"])

    def test_factor_selection_covers_vif_pca_lasso_and_fdr(self) -> None:
        rng = np.random.default_rng(22)
        x = rng.normal(size=(500, 6))
        x[:, 5] = x[:, 0] * 0.99 + rng.normal(scale=0.01, size=500)
        y = 1.2 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(scale=0.4, size=500)
        result = factor_selection_diagnostics({
            "factor_matrix": x.tolist(),
            "target": y.tolist(),
            "factor_names": ["value", "quality", "size", "momentum", "liquidity", "value_clone"],
            "lasso_penalty": 10.0,
            "fdr_level": 0.05,
        })
        self.assertIn("value", result["selected_by_fdr"])
        self.assertIn("quality", result["selected_by_fdr"])
        self.assertTrue({"value", "value_clone"} & set(result["high_vif_factors"]))
        self.assertLessEqual(result["pca"]["components_for_threshold"], 6)
        self.assertGreater(result["crowding_proxy_mean_absolute_correlation"], 0.05)

    def test_overfit_diagnostics_are_deterministic_and_bounded(self) -> None:
        rng = np.random.default_rng(23)
        returns = rng.normal(0.0, 0.01, size=(160, 12))
        returns[:, 0] += 0.0008
        request = {
            "strategy_returns": returns.tolist(),
            "blocks": 8,
            "bootstraps": 150,
            "seed": 7,
        }
        first = factor_overfit_diagnostics(request)
        second = factor_overfit_diagnostics(request)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["probability_of_backtest_overfitting"], 0.0)
        self.assertLessEqual(first["probability_of_backtest_overfitting"], 1.0)
        self.assertGreaterEqual(first["white_reality_check"]["bootstrap_p_value"], 0.0)
        self.assertLessEqual(first["white_reality_check"]["bootstrap_p_value"], 1.0)


class CrossValidationTests(unittest.TestCase):
    def test_rolling_spatial_and_nested_plans_have_no_leakage(self) -> None:
        rolling = cross_validation_plan({
            "rows": 100,
            "strategy": "rolling",
            "splits": 4,
            "minimum_train_size": 40,
            "test_size": 10,
        })
        spatial = cross_validation_plan({
            "rows": 60,
            "strategy": "spatial_block",
            "splits": 3,
            "spatial_blocks": [f"block-{index // 10}" for index in range(60)],
        })
        nested = cross_validation_plan({
            "rows": 60,
            "strategy": "nested",
            "outer_splits": 3,
            "inner_splits": 3,
        })
        for result in (rolling, spatial, nested):
            self.assertTrue(result["leakage_check_passed"])
            for fold in result["folds"]:
                self.assertTrue(set(fold["train_indices"]).isdisjoint(fold["test_indices"]))
        self.assertEqual(nested["fold_count"], 3)
        self.assertEqual(len(nested["folds"][0]["inner_folds"]), 3)


if __name__ == "__main__":
    unittest.main()
