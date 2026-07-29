from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _available(package: str, expected: str) -> bool:
    try:
        return version(package) == expected
    except PackageNotFoundError:
        return False


DIFFUSION_AVAILABLE = _available("ndlib", "5.1.1")
CAUSAL_AVAILABLE = _available("dowhy", "0.14")
BAYESIAN_NETWORK_AVAILABLE = _available("pgmpy", "1.1.2")

dispatch_spec = importlib.util.spec_from_file_location("compute_dispatch_phase3", ROOT / "compute_dispatch.py")
assert dispatch_spec and dispatch_spec.loader
dispatch = importlib.util.module_from_spec(dispatch_spec)
sys.modules["compute_dispatch_phase3"] = dispatch
dispatch_spec.loader.exec_module(dispatch)


class Phase3Base(unittest.TestCase):
    def execute(self, operation: str, inputs: dict) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        result = dispatch.run_ticket({"task_id": f"phase3-{operation}-001", "operation": operation, "inputs": inputs}, Path(directory.name))
        self.assertFalse(result["execution"]["network_used"])
        self.assertEqual(result["execution"]["model_calls"], 0)
        return result["results"]


@unittest.skipUnless(DIFFUSION_AVAILABLE, "Pinned NDlib method pack is not installed")
class DiffusionOperationTests(Phase3Base):
    def test_all_diffusion_modes_are_bounded_and_reproducible(self) -> None:
        edges = [[index, (index + 1) % 30] for index in range(30)] + [[index, (index + 3) % 30] for index in range(30)]
        modes = ["sir_information_spread", "threshold_adoption", "independent_cascade", "voter_opinion", "majority_rule", "bounded_confidence", "cognitive_risk_opinion"]
        for mode in modes:
            inputs = {"mode": mode, "node_count": 30, "edges": edges, "steps": 8, "seeds": [7, 11], "initial_nodes": [0, 1], "transmission_probability": 0.3, "recovery_probability": 0.1, "threshold": 0.2, "activation_probability": 0.3}
            first = self.execute("information_diffusion_analysis", inputs)
            second = self.execute("information_diffusion_analysis", inputs)
            self.assertEqual(first, second, mode)
            self.assertGreaterEqual(first["aggregate"]["mean_final_share"], 0)
            self.assertLessEqual(first["aggregate"]["mean_final_share"], 1)
            self.assertEqual(first["run_count"], 2)


@unittest.skipUnless(CAUSAL_AVAILABLE, "Pinned DoWhy method pack is not installed")
class CausalPolicyOperationTests(Phase3Base):
    @staticmethod
    def observational_data() -> dict:
        rng = np.random.default_rng(42)
        confounder = rng.normal(size=200)
        treatment_probability = 1 / (1 + np.exp(-confounder))
        treatment = (rng.random(200) < treatment_probability).astype(int)
        outcome = 2.0 * treatment + 1.5 * confounder + rng.normal(scale=0.2, size=200)
        return {"treatment": treatment.tolist(), "outcome": outcome.tolist(), "confounders": {"baseline_risk": confounder.tolist()}}

    def test_backdoor_and_propensity(self) -> None:
        data = self.observational_data()
        backdoor = self.execute("causal_policy_evaluation", {"mode": "backdoor_adjustment", **data})
        weighted = self.execute("causal_policy_evaluation", {"mode": "propensity_weighting", **data})
        self.assertTrue(backdoor["identified"])
        self.assertLess(abs(backdoor["effect"] - 2.0), 0.3)
        self.assertTrue(weighted["identified"])

    def test_refutation_modes(self) -> None:
        did = self.execute("causal_policy_evaluation", {"mode": "difference_in_differences_refuted", "treated_pre": [10, 11, 12, 13, 14, 15], "treated_post": [14, 15, 16, 17, 18, 19], "control_pre": [8, 9, 10, 11, 12, 13], "control_post": [9, 10, 11, 12, 13, 14]})
        self.assertAlmostEqual(did["effect"], 3.0)
        rng = np.random.default_rng(9)
        instrument = rng.integers(0, 2, 300)
        treatment = 2 * instrument + rng.normal(scale=0.3, size=300)
        outcome = 3 * treatment + rng.normal(scale=0.5, size=300)
        iv = self.execute("causal_policy_evaluation", {"mode": "instrumental_variable_refuted", "instrument": instrument.tolist(), "treatment": treatment.tolist(), "outcome": outcome.tolist()})
        self.assertFalse(iv["weak_instrument"])
        data = self.observational_data()
        placebo = self.execute("causal_policy_evaluation", {"mode": "placebo_policy_test", "repetitions": 50, "seed": 7, **data})
        self.assertEqual(placebo["repetitions"], 50)
        sensitivity = self.execute("causal_policy_evaluation", {"mode": "unobserved_confounding_sensitivity", "effect_estimate": 2.0, "standard_error": 0.4, "bias_strengths": [0, 1, 3, 6]})
        self.assertEqual(len(sensitivity["scenarios"]), 4)


@unittest.skipUnless(BAYESIAN_NETWORK_AVAILABLE, "Pinned pgmpy method pack is not installed")
class BayesianNetworkOperationTests(Phase3Base):
    @staticmethod
    def network() -> dict:
        return {"nodes": ["A", "B"], "edges": [["A", "B"]], "cpds": [{"variable": "A", "variable_card": 2, "values": [[0.6], [0.4]]}, {"variable": "B", "variable_card": 2, "values": [[0.9, 0.2], [0.1, 0.8]], "evidence": ["A"], "evidence_card": [2]}], "query_variables": ["B"]}

    def test_inference_prior_and_evidence_modes(self) -> None:
        fixed = self.execute("bayesian_network_inference", {"mode": "fixed_network_inference", "evidence": {"A": 1}, **self.network()})
        self.assertTrue(fixed["model_valid"])
        prior = self.execute("bayesian_network_inference", {"mode": "expert_prior_update", "states": ["low", "high"], "prior_counts": [2, 2], "observed_counts": [3, 7]})
        self.assertAlmostEqual(sum(prior["posterior_probability"]), 1.0)
        sensitivity = self.execute("bayesian_network_inference", {"mode": "evidence_sensitivity", "evidence_scenarios": [{"name": "a0", "evidence": {"A": 0}}, {"name": "a1", "evidence": {"A": 1}}], **self.network()})
        self.assertEqual(sensitivity["scenario_count"], 2)
        virtual = self.execute("bayesian_network_inference", {"mode": "virtual_evidence_update", "virtual_evidence": [{"variable": "A", "probabilities": [0.2, 0.8]}], **self.network()})
        self.assertEqual(virtual["virtual_evidence_count"], 1)

    def test_parameter_and_em_estimation(self) -> None:
        rng = np.random.default_rng(12)
        a = rng.integers(0, 2, 200)
        b = np.where(rng.random(200) < np.where(a == 1, 0.8, 0.2), 1, 0)
        parameter = self.execute("bayesian_network_inference", {"mode": "bayesian_parameter_estimation", "edges": [["A", "B"]], "data": {"A": a.tolist(), "B": b.tolist()}, "equivalent_sample_size": 5})
        self.assertTrue(parameter["model_valid"])
        em = self.execute("bayesian_network_inference", {"mode": "em_parameter_estimation", "nodes": ["A", "L", "B"], "edges": [["A", "L"], ["L", "B"]], "latent_nodes": ["L"], "latent_cards": {"L": 2}, "data": {"A": a[:80].tolist(), "B": b[:80].tolist()}, "max_iterations": 3, "seed": 4})
        self.assertTrue(em["model_valid"])
        self.assertEqual(em["latent_cards"], {"L": 2})


if __name__ == "__main__":
    unittest.main()
