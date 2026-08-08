from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from decision_intelligence_gateway import finance_decision_analysis
from dynamic_family_router import family_runtime_metadata, resolve_dynamic_family
from dynamic_transport_routing_planner import (
    DynamicTransportRoutingError,
    plan_dynamic_transport_routing,
    run_dynamic_transport_routing_ticket,
)

PIPELINE = {
    "pipeline_id": "dynamic-auto-v1",
    "stage_id": "dynamic",
    "sequence_reason": "transport-routing dynamic-family test",
    "upstream_refs": [],
}

ALL_SIGNALS = {
    "cost_consistency_tolerance": 1e-9,
    "maximum_total_cost": 3.1,
    "cost_target_tolerance": 0.0,
    "maximum_link_count": 3,
    "link_count_tolerance": 0,
}


def routing_ticket(*, context=None):
    return {
        "task_id": "dynamic-transport-routing-test",
        "objective": "Objective prose must never select route validation stages.",
        "operation": "finance_decision_analysis",
        "inputs": {
            "mode": "aequilibrae_shortest_path",
            "links": [
                {"a_node": 1, "b_node": 2, "cost": 2.0},
                {"a_node": 2, "b_node": 5, "cost": 2.0},
                {"a_node": 1, "b_node": 3, "cost": 1.0},
                {"a_node": 3, "b_node": 4, "cost": 1.0},
                {"a_node": 4, "b_node": 5, "cost": 1.0},
                {"a_node": 2, "b_node": 4, "cost": 10.0},
                {"a_node": 1, "b_node": 5, "cost": 9.0},
                {"a_node": 1, "b_node": 3, "cost": 4.0},
            ],
            "origin": 1,
            "destination": 5,
            "transport_routing_context": {} if context is None else dict(context),
        },
        "pipeline": dict(PIPELINE),
        "quality_profile": {"decision_class": "exploratory", "publication_policy": "status_only"},
    }


class DynamicTransportRoutingFamilyTests(unittest.TestCase):
    def test_router_and_runtime_are_narrow(self) -> None:
        ticket = routing_ticket()
        self.assertEqual(resolve_dynamic_family(ticket), "transport-routing")
        metadata = family_runtime_metadata(ticket)
        self.assertEqual(metadata["python_version"], "3.12")
        self.assertEqual(metadata["requirements"], ["requirements-ortools.txt", "requirements-global-aequilibrae.txt"])

    def test_exact_crosscheck_is_always_required(self) -> None:
        plan = plan_dynamic_transport_routing(routing_ticket())
        self.assertEqual(plan["stage_order"], ["aequilibrae_shortest_path", "networkx_exact_route_audit"])
        self.assertEqual(plan["optimization"]["solver_status"], "OPTIMAL")
        self.assertEqual(plan["optimization"]["objective_value"], 215)
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["unique_optimum"])

    def test_cost_target_adds_only_cost_audit(self) -> None:
        plan = plan_dynamic_transport_routing(routing_ticket(context={"maximum_total_cost": 4.0}))
        self.assertEqual(plan["stage_order"], ["aequilibrae_shortest_path", "networkx_exact_route_audit", "route_cost_target_audit"])

    def test_hop_target_adds_only_hop_audit(self) -> None:
        plan = plan_dynamic_transport_routing(routing_ticket(context={"maximum_link_count": 4}))
        self.assertEqual(plan["stage_order"], ["aequilibrae_shortest_path", "networkx_exact_route_audit", "route_hop_target_audit"])

    def test_all_signals_produce_unique_optimum(self) -> None:
        plan = plan_dynamic_transport_routing(routing_ticket(context=ALL_SIGNALS))
        self.assertEqual(plan["stage_order"], [
            "aequilibrae_shortest_path",
            "networkx_exact_route_audit",
            "route_cost_target_audit",
            "route_hop_target_audit",
        ])
        self.assertEqual(plan["optimization"]["objective_value"], 465)
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["passed"])
        self.assertTrue(plan["optimization"]["exhaustive_cross_check"]["unique_optimum"])

    def test_nonpositive_cost_fails_closed(self) -> None:
        ticket = routing_ticket()
        ticket["inputs"]["links"][0]["cost"] = 0.0
        with self.assertRaises(Exception):
            plan_dynamic_transport_routing(ticket)

    def test_unknown_context_fails_closed(self) -> None:
        with self.assertRaises(DynamicTransportRoutingError):
            plan_dynamic_transport_routing(routing_ticket(context={"run_every_routing_tool": True}))

    def test_objective_text_does_not_select_target_nodes(self) -> None:
        ticket = routing_ticket()
        ticket["objective"] = "Run cost target, hop target, every routing engine and all route tools."
        plan = plan_dynamic_transport_routing(ticket)
        self.assertEqual(plan["stage_order"], ["aequilibrae_shortest_path", "networkx_exact_route_audit"])
        self.assertFalse(plan["objective_text_used"])

    def test_tampered_primary_result_fails_exact_crosscheck(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-transport-routing-tamper-"))
        try:
            def operation(inputs):
                if inputs.get("mode") == "aequilibrae_shortest_path":
                    return {
                        "mode": "aequilibrae_shortest_path",
                        "origin": 1,
                        "destination": 5,
                        "path_nodes": [1, 3, 4, 5],
                        "path_links": [3, 4, 5],
                        "total_cost": 4.0,
                        "link_count": 3,
                        "network_link_count": 8,
                        "engine": {"aequilibrae": "tampered-test"},
                    }
                return finance_decision_analysis(inputs)

            with self.assertRaises(DynamicTransportRoutingError):
                run_dynamic_transport_routing_ticket(routing_ticket(), root, {"finance_decision_analysis": operation})
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("aequilibrae") is not None, "AequilibraE is a managed optional dependency; real execution is enforced by transport-routing CI")
    def test_real_aequilibrae_vs_networkx_pipeline(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-transport-routing-"))
        try:
            result = run_dynamic_transport_routing_ticket(
                routing_ticket(context=ALL_SIGNALS),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            primary = result["results"]["final_result"]
            validation = result["results"]["validation_results"]
            self.assertEqual(result["status"], "success")
            self.assertAlmostEqual(primary["total_cost"], 3.0, delta=1e-9)
            self.assertEqual(primary["path_nodes"], [1, 3, 4, 5])
            self.assertEqual(primary["link_count"], 3)
            self.assertEqual(validation["networkx_exact_route_audit"]["status"], "PASS")
            self.assertEqual(validation["networkx_exact_route_audit"]["candidate_count"], 2)
            self.assertEqual(validation["route_cost_target_audit"]["status"], "PASS")
            self.assertEqual(validation["route_hop_target_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["optimization"]["objective_value"], 465)
            self.assertFalse(result["execution"]["network_used"])
            self.assertEqual(result["execution"]["model_calls"], 0)
            self.assertTrue(result["execution"]["graph_contains_branching"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("aequilibrae") is not None, "AequilibraE is optional")
    def test_target_failure_is_informative(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dynamic-transport-routing-target-fail-"))
        try:
            result = run_dynamic_transport_routing_ticket(
                routing_ticket(context={"maximum_total_cost": 2.0}),
                root,
                {"finance_decision_analysis": finance_decision_analysis},
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"]["validation_results"]["networkx_exact_route_audit"]["status"], "PASS")
            self.assertEqual(result["results"]["validation_results"]["route_cost_target_audit"]["status"], "FAIL")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
