from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compute_dispatch  # noqa: E402
from relay_contracts import build_data_gap_plan, build_expert_review_request  # noqa: E402


class RelayContractTests(unittest.TestCase):
    def test_data_gap_plan_never_dispatches_directly(self) -> None:
        ticket = {"task_id": "relay-gap-001", "operation": "break_even_analysis", "inputs": {}}
        preflight = {
            "status": "DATA_INSUFFICIENT",
            "execution_allowed": False,
            "issues": [
                {
                    "code": "REQUIRED_DATA_MISSING",
                    "variable": "demand",
                    "blocking": True,
                    "message": "missing",
                    "remediation": "resolve",
                }
            ],
            "recommended_operations": ["sensitivity_analysis"],
        }
        plan = build_data_gap_plan(ticket, preflight)
        self.assertEqual(plan["relay_owner"], "gpts-usage-center")
        self.assertFalse(plan["center_direct_contact_allowed"])
        chain = plan["gpts_relay_requests"][0]["ordered_fallback_chain"]
        self.assertEqual(chain[0]["action"], "create_separate_api_ticket")
        self.assertTrue(all(item["actor"] == "gpts-usage-center" for item in chain))
        self.assertTrue(all(item["direct_center_call"] is False for item in chain))

    def test_expert_review_is_a_gpts_handoff_not_a_center_call(self) -> None:
        request = build_expert_review_request(
            {"task_id": "relay-expert-001", "operation": "monte_carlo"},
            {"package_sha256": "a" * 64},
            {"status": "DATA_READY", "data_summary": {}, "recommended_operations": []},
        )
        self.assertEqual(request["relay_owner"], "gpts-usage-center")
        self.assertFalse(request["direct_center_delivery_allowed"])
        self.assertFalse(request["compute_center_dispatch_performed"])
        self.assertIn("separate [execution] ticket", request["delivery_mode"])

    def test_successful_dispatch_writes_both_relay_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_path = root / "ticket.json"
            output = root / "output"
            ticket_path.write_text(
                json.dumps(
                    {
                        "task_id": "relay-success-001",
                        "objective": "Verify relay artifacts",
                        "operation": "break_even_analysis",
                        "inputs": {"fixed_cost": 1000, "unit_price": 20, "variable_cost": 12},
                    }
                ),
                encoding="utf-8",
            )
            rc = compute_dispatch.main(["--ticket", str(ticket_path), "--output-dir", str(output)])
            self.assertEqual(rc, 0)
            gap = json.loads((output / "compute-data-gap-plan.json").read_text(encoding="utf-8"))
            expert = json.loads((output / "compute-expert-review-request.json").read_text(encoding="utf-8"))
            result = json.loads((output / "compute-result.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
            paths = {row["path"] for row in manifest["files"]}
            self.assertEqual(gap["relay_owner"], "gpts-usage-center")
            self.assertFalse(expert["direct_center_delivery_allowed"])
            self.assertEqual(result["relay_contract"]["sole_relay"], "gpts-usage-center")
            self.assertIn("compute-data-gap-plan.json", paths)
            self.assertIn("compute-expert-review-request.json", paths)

    def test_blocked_dispatch_preserves_data_gap_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_path = root / "ticket.json"
            output = root / "output"
            ticket_path.write_text(
                json.dumps(
                    {
                        "task_id": "relay-blocked-001",
                        "operation": "break_even_analysis",
                        "inputs": {"fixed_cost": 1000, "unit_price": 20, "variable_cost": 12},
                        "data_context": {
                            "variables": [
                                {
                                    "name": "demand",
                                    "required": True,
                                    "source_type": "api_observation",
                                    "confidence": "high",
                                    "missing": True,
                                    "replacement_strategy": "none",
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            rc = compute_dispatch.main(["--ticket", str(ticket_path), "--output-dir", str(output)])
            self.assertEqual(rc, 2)
            plan = json.loads((output / "compute-data-gap-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["preflight_status"], "DATA_INSUFFICIENT")
            self.assertFalse(plan["execution_allowed"])
            self.assertTrue(plan["gpts_relay_requests"])


if __name__ == "__main__":
    unittest.main()
