from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from dynamic_bayesian_network_planner import (
    DynamicBayesianNetworkError,
    plan_dynamic_bayesian_network,
    run_dynamic_bayesian_network_ticket,
)
from dynamic_family_router import DynamicFamilyRoutingError, resolve_dynamic_family
from tool_registry import managed_runtime_plan, requirement_files_for_ticket


def dynamic_pipeline() -> dict[str, object]:
    return {
        "pipeline_id": "dynamic-auto-v1",
        "stage_id": "dynamic",
        "sequence_reason": "Bayesian dynamic family test",
        "upstream_refs": [],
    }


def bayesian_ticket(
    *,
    task_id: str,
    decision_class: str = "exploratory",
    include_scenarios: bool = False,
    request_sensitivity: bool = False,
    include_virtual: bool = False,
    request_virtual: bool = False,
) -> dict[str, object]:
    count = 80
    a = [index % 2 for index in range(count)]
    b = [a[index] if index % 5 else 1 - a[index] for index in range(count)]
    c = [b[index] if index % 7 else 1 - b[index] for index in range(count)]
    inputs: dict[str, object] = {
        "mode": "bayesian_parameter_estimation",
        "edges": [["A", "B"], ["B", "C"]],
        "data": {"A": a, "B": b, "C": c},
        "query_variables": ["C"],
        "evidence": {"A": 1},
        "equivalent_sample_size": 5.0,
    }
    if include_scenarios:
        inputs["evidence_scenarios"] = [
            {"name": "A=0", "evidence": {"A": 0}},
            {"name": "A=1", "evidence": {"A": 1}},
        ]
    if include_virtual:
        inputs["virtual_evidence"] = [
            {"variable": "B", "probabilities": [0.25, 0.75], "state_names": [0, 1]}
        ]
    context: dict[str, bool] = {}
    if request_sensitivity:
        context["evidence_sensitivity"] = True
    if request_virtual:
        context["virtual_evidence_update"] = True
    if context:
        inputs["dynamic_context"] = context
    return {
        "task_id": task_id,
        "operation": "bayesian_network_inference",
        "inputs": inputs,
        "pipeline": dynamic_pipeline(),
        "quality_profile": {
            "decision_class": decision_class,
            "probabilistic_claim": False,
        },
    }


def _engine() -> dict[str, object]:
    return {
        "name": "pgmpy-isolated-fixed-adapter",
        "version": "1.1.2",
        "network_used": False,
    }


def bayesian_stub_factory(seen: list[dict[str, Any]]):
    def bayesian_stub(inputs: Mapping[str, Any]) -> dict[str, Any]:
        captured = json.loads(json.dumps(dict(inputs), ensure_ascii=False, allow_nan=False))
        seen.append(captured)
        mode = str(inputs["mode"])
        boundary = (
            "Bayesian dependencies and posterior updates do not establish causal structure without external identification evidence."
        )
        if mode == "bayesian_parameter_estimation":
            return {
                "engine": _engine(),
                "mode": mode,
                "estimator": "bayesian_bdeu",
                "model_valid": True,
                "cpds": [
                    {
                        "variable": "A",
                        "variable_card": 2,
                        "evidence": [],
                        "cardinality": [2],
                        "values": [[0.5], [0.5]],
                        "state_names": {"A": [0, 1]},
                    },
                    {
                        "variable": "B",
                        "variable_card": 2,
                        "evidence": ["A"],
                        "cardinality": [2, 2],
                        "values": [[0.8, 0.2], [0.2, 0.8]],
                        "state_names": {"B": [0, 1], "A": [0, 1]},
                    },
                    {
                        "variable": "C",
                        "variable_card": 2,
                        "evidence": ["B"],
                        "cardinality": [2, 2],
                        "values": [[0.85, 0.15], [0.15, 0.85]],
                        "state_names": {"C": [0, 1], "B": [0, 1]},
                    },
                ],
                "observation_count": len(inputs["data"]["A"]),
                "causal_structure_claimed": False,
                "interpretation_boundary": boundary,
            }
        cpds = inputs["cpds"]
        self_card = {row["variable"]: row["evidence_card"] for row in cpds}
        if self_card != {"A": [], "B": [2], "C": [2]}:
            raise AssertionError(f"unexpected evidence_card conversion: {self_card}")
        if mode == "fixed_network_inference":
            return {
                "engine": _engine(),
                "mode": mode,
                "query": {
                    "variables": ["C"],
                    "cardinality": [2],
                    "values": [0.2, 0.8],
                    "state_names": {"C": [0, 1]},
                },
                "model_valid": True,
                "node_count": 3,
                "edge_count": 2,
                "causal_structure_claimed": False,
                "interpretation_boundary": boundary,
            }
        if mode == "evidence_sensitivity":
            return {
                "engine": _engine(),
                "mode": mode,
                "scenario_count": len(inputs["evidence_scenarios"]),
                "scenarios": [
                    {"name": row["name"], "query": {"variables": ["C"], "values": [0.5, 0.5]}}
                    for row in inputs["evidence_scenarios"]
                ],
                "causal_structure_claimed": False,
                "interpretation_boundary": boundary,
            }
        if mode == "virtual_evidence_update":
            return {
                "engine": _engine(),
                "mode": mode,
                "query": {"variables": ["C"], "cardinality": [2], "values": [0.3, 0.7]},
                "virtual_evidence_count": len(inputs["virtual_evidence"]),
                "causal_structure_claimed": False,
                "interpretation_boundary": boundary,
            }
        raise AssertionError(f"unexpected Bayesian mode: {mode}")

    return bayesian_stub


class DynamicBayesianNetworkFamilyTests(unittest.TestCase):
    def test_exploratory_selects_only_mandatory_estimation_and_posterior(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-exploratory")
        self.assertEqual(resolve_dynamic_family(ticket), "bayesian-network")
        plan = plan_dynamic_bayesian_network(ticket)
        self.assertEqual(plan["stage_order"], ["parameter_estimation", "posterior_inference"])
        self.assertEqual(plan["stage_map"]["parameter_estimation"]["depends_on"], [])
        self.assertEqual(plan["stage_map"]["posterior_inference"]["depends_on"], ["parameter_estimation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        cross = plan["optimization"]["exhaustive_cross_check"]
        self.assertTrue(cross["performed"])
        self.assertTrue(cross["passed"])
        self.assertEqual(cross["optional_node_count"], 2)
        self.assertFalse(plan["objective_text_used"])

    def test_formal_with_scenarios_selects_sensitivity_by_utility(self) -> None:
        ticket = bayesian_ticket(
            task_id="bayesian-formal",
            decision_class="formal",
            include_scenarios=True,
        )
        plan = plan_dynamic_bayesian_network(ticket)
        self.assertEqual(
            plan["stage_order"],
            ["parameter_estimation", "posterior_inference", "evidence_sensitivity"],
        )
        self.assertTrue(plan["optimization"]["selected_nodes"]["evidence_sensitivity"])
        self.assertFalse(plan["optimization"]["selected_nodes"]["virtual_evidence_update"])
        self.assertGreater(plan["optimization"]["utility_by_node"]["evidence_sensitivity"], 0)

    def test_explicit_requests_select_true_branching_dag_but_serial_execution(self) -> None:
        ticket = bayesian_ticket(
            task_id="bayesian-full",
            include_scenarios=True,
            request_sensitivity=True,
            include_virtual=True,
            request_virtual=True,
        )
        plan = plan_dynamic_bayesian_network(ticket)
        expected = [
            "parameter_estimation",
            "posterior_inference",
            "evidence_sensitivity",
            "virtual_evidence_update",
        ]
        self.assertEqual(plan["stage_order"], expected)
        self.assertEqual(plan["stage_map"]["posterior_inference"]["depends_on"], ["parameter_estimation"])
        self.assertEqual(plan["stage_map"]["evidence_sensitivity"]["depends_on"], ["parameter_estimation"])
        self.assertEqual(plan["stage_map"]["virtual_evidence_update"]["depends_on"], ["parameter_estimation"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertTrue(plan["optimization"]["global_optimal_proven"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])

        requirements = requirement_files_for_ticket(ticket)
        self.assertEqual(
            [Path(item).name for item in requirements],
            ["requirements-ortools.txt", "requirements-bayesian-network.txt"],
        )
        runtime = managed_runtime_plan(ticket)
        self.assertEqual(runtime["dynamic_family"], "bayesian-network")
        self.assertEqual(runtime["dynamic_entry_contract"], "bayesian_network_inference")
        self.assertEqual(runtime["python_version"], "3.12")
        self.assertEqual(runtime["network_policy"], "deny")
        self.assertFalse(runtime["automatic_parallel_execution"])

        seen: list[dict[str, Any]] = []
        operations = {"bayesian_network_inference": bayesian_stub_factory(seen)}
        with tempfile.TemporaryDirectory() as directory, patch(
            "dynamic_bayesian_network_planner.version", return_value="1.1.2"
        ):
            result = run_dynamic_bayesian_network_ticket(ticket, Path(directory), operations)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["dynamic_family"], "bayesian-network")
            self.assertEqual(result["results"]["stage_order"], expected)
            self.assertEqual(result["results"]["final_stage"], "posterior_inference")
            self.assertEqual(result["results"]["final_result"]["mode"], "fixed_network_inference")
            self.assertEqual(set(result["results"]["robustness_results"]), {"evidence_sensitivity", "virtual_evidence_update"})
            self.assertFalse(result["results"]["causal_structure_claimed"])
            self.assertEqual(len(result["results"]["stage_receipts"]), 4)
            self.assertTrue(all(row["status"] == "PASS" for row in result["results"]["stage_receipts"]))
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertFalse(result["execution"]["automatic_parallel_execution"])
            self.assertTrue(result["execution"]["graph_contains_branching"])
            self.assertEqual([row["mode"] for row in seen], [
                "bayesian_parameter_estimation",
                "fixed_network_inference",
                "evidence_sensitivity",
                "virtual_evidence_update",
            ])
            state = json.loads((Path(directory) / "compute-dynamic-pipeline-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "PASS")
            self.assertEqual(state["family"], "bayesian-network")
            self.assertTrue(all("input_sha256" in row and "output_sha256" in row for row in state["stages"]))

    def test_high_stakes_requires_multiple_sensitivity_scenarios(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-high-stakes", decision_class="high_stakes")
        with self.assertRaises(DynamicBayesianNetworkError):
            plan_dynamic_bayesian_network(ticket)

    def test_high_stakes_forces_sensitivity_when_scenarios_exist(self) -> None:
        ticket = bayesian_ticket(
            task_id="bayesian-high-stakes-full",
            decision_class="high_stakes",
            include_scenarios=True,
        )
        plan = plan_dynamic_bayesian_network(ticket)
        self.assertTrue(plan["optimization"]["selected_nodes"]["evidence_sensitivity"])
        self.assertIn("evidence_sensitivity", plan["stage_order"])

    def test_objective_text_does_not_select_optional_stages(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-objective-text")
        ticket["objective"] = "Run every Bayesian sensitivity and virtual evidence analysis available"
        plan = plan_dynamic_bayesian_network(ticket)
        self.assertEqual(plan["stage_order"], ["parameter_estimation", "posterior_inference"])
        self.assertFalse(plan["planning_features"]["sensitivity_requested"])
        self.assertFalse(plan["planning_features"]["virtual_evidence_requested"])
        self.assertFalse(plan["objective_text_used"])

    def test_router_rejects_non_estimation_entry_mode(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-fixed-entry")
        ticket["inputs"]["mode"] = "fixed_network_inference"
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)
        with self.assertRaises(DynamicFamilyRoutingError):
            requirement_files_for_ticket(ticket)

    def test_router_rejects_dependency_node_without_observed_data(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-missing-node")
        ticket["inputs"]["edges"] = [["A", "D"]]
        with self.assertRaises(DynamicFamilyRoutingError):
            resolve_dynamic_family(ticket)

    def test_explicit_sensitivity_request_requires_scenarios(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-missing-scenarios", request_sensitivity=True)
        with self.assertRaises(DynamicBayesianNetworkError):
            plan_dynamic_bayesian_network(ticket)

    def test_virtual_evidence_probabilities_must_sum_to_one(self) -> None:
        ticket = bayesian_ticket(task_id="bayesian-bad-virtual", include_virtual=True, request_virtual=True)
        ticket["inputs"]["virtual_evidence"][0]["probabilities"] = [0.2, 0.2]
        with self.assertRaises(DynamicBayesianNetworkError):
            plan_dynamic_bayesian_network(ticket)


if __name__ == "__main__":
    unittest.main()
