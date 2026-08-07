import copy
import json
import tempfile
import unittest
from pathlib import Path

import compute_runner
from pipeline_engine import (
    PIPELINE_REGISTRY_PATH,
    PipelineEngineError,
    resolve_pipeline_ticket,
    run_pipeline_ticket,
    validate_registry,
)


class PipelineEngineTests(unittest.TestCase):
    def _ticket(self):
        return {
            "task_id": "pipeline-test-001",
            "operation": "scenario_compare",
            "inputs": {
                "model": {
                    "intercept": 10.0,
                    "coefficients": {"demand": 2.0, "cost": -1.0},
                },
                "scenarios": [
                    {"name": "weak", "values": {"demand": 1.0, "cost": 4.0}},
                    {"name": "base", "values": {"demand": 2.0, "cost": 3.0}},
                    {"name": "strong", "values": {"demand": 4.0, "cost": 1.0}},
                ],
            },
            "pipeline": {
                "pipeline_id": "scenario-risk-linear-v1",
                "stage_id": "pipeline",
                "sequence_reason": "fixed regression test",
                "upstream_refs": [],
            },
        }

    def test_registry_is_a_strict_serial_networkx_chain(self):
        registry = validate_registry()
        pipeline = registry["pipelines"]["scenario-risk-linear-v1"]
        self.assertEqual(
            pipeline["stage_order"], ["scenarios", "sensitivity", "risk_simulation"]
        )
        self.assertFalse(registry["automatic_parallel_execution"])
        self.assertEqual(registry["network_policy"], "deny")

    def test_pipeline_executes_three_core_operations_deterministically(self):
        ticket = self._ticket()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            result_a = run_pipeline_ticket(ticket, Path(first), compute_runner.OPERATIONS)
            result_b = run_pipeline_ticket(ticket, Path(second), compute_runner.OPERATIONS)
            self.assertEqual(result_a["result_sha256"], result_b["result_sha256"])
            self.assertEqual(
                result_a["results"]["stage_order"],
                ["scenarios", "sensitivity", "risk_simulation"],
            )
            self.assertEqual(result_a["results"]["final_result"]["iterations"], 5000)
            self.assertEqual(result_a["results"]["final_result"]["seed"], 20260807)
            self.assertFalse(result_a["execution"]["automatic_parallel_execution"])
            self.assertFalse(result_a["execution"]["network_used"])
            state = json.loads((Path(first) / "compute-pipeline-state.json").read_text())
            self.assertEqual(state["status"], "PASS")
            self.assertTrue(all(row["status"] == "PASS" for row in state["stages"]))
            self.assertEqual(len(list((Path(first) / "pipeline-stages").glob("*.json"))), 6)

    def test_unknown_single_ticket_pipeline_fails_closed(self):
        ticket = self._ticket()
        ticket["pipeline"]["pipeline_id"] = "unregistered-pipeline-v1"
        with self.assertRaises(PipelineEngineError):
            resolve_pipeline_ticket(ticket)

    def test_existing_multi_ticket_pipeline_metadata_does_not_trigger_engine(self):
        ticket = self._ticket()
        ticket["pipeline"]["stage_id"] = "stage-001"
        self.assertIsNone(resolve_pipeline_ticket(ticket))

    def test_cycle_is_rejected(self):
        document = json.loads(PIPELINE_REGISTRY_PATH.read_text())
        broken = copy.deepcopy(document)
        stages = broken["pipelines"][0]["stages"]
        stages[0]["depends_on"] = ["risk_simulation"]
        with self.assertRaises(PipelineEngineError):
            validate_registry(broken)

    def test_stage_output_contract_fails_closed(self):
        ticket = self._ticket()
        bad_operations = dict(compute_runner.OPERATIONS)
        bad_operations["scenario_compare"] = lambda inputs: {"bad": True}
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(PipelineEngineError):
                run_pipeline_ticket(ticket, Path(output), bad_operations)
            state = json.loads((Path(output) / "compute-pipeline-state.json").read_text())
            self.assertEqual(state["status"], "FAILED")
            self.assertEqual(state["stages"][0]["status"], "FAILED")

    def test_sensitivity_adapter_rejects_no_variation(self):
        ticket = self._ticket()
        for scenario in ticket["inputs"]["scenarios"]:
            scenario["values"]["cost"] = 3.0
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(PipelineEngineError):
                run_pipeline_ticket(ticket, Path(output), compute_runner.OPERATIONS)


if __name__ == "__main__":
    unittest.main()
