from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compute_ticket


class ComputeTicketTests(unittest.TestCase):
    def packet(self) -> dict:
        return {
            "task_id": "compute-ticket-test-0001",
            "operation": "descriptive_statistics",
            "inputs": {"data": [1, 2, 3]},
        }

    def event(self, packet: object, *, title: str = "[compute] test") -> dict:
        return {
            "sender": {"login": "owner"},
            "issue": {
                "number": 12,
                "title": title,
                "body": json.dumps(packet),
                "user": {"login": "owner"},
            },
        }

    def run_prepare(self, event: dict) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_path = root / "event.json"
            event_path.write_text(json.dumps(event), encoding="utf-8")
            args = argparse.Namespace(event_path=str(event_path), output_dir=str(root / "out"))
            with patch.dict(
                os.environ,
                {
                    "REPOSITORY_OWNER": "owner",
                    "GITHUB_REPOSITORY": "owner/repo",
                    "GITHUB_TOKEN": "",
                    "GH_TOKEN": "",
                },
                clear=False,
            ):
                code = compute_ticket.prepare(args)
            status = json.loads((root / "out" / "ticket-status.json").read_text(encoding="utf-8"))
            return code, status

    def test_valid_ticket_is_accepted_without_network(self) -> None:
        code, status = self.run_prepare(self.event(self.packet()))
        self.assertEqual(code, 0)
        self.assertTrue(status["accepted"])
        self.assertEqual(status["operation"], "descriptive_statistics")
        self.assertRegex(status["ticket_sha256"], r"^[0-9a-f]{64}$")

    def test_wrong_title_is_rejected(self) -> None:
        code, status = self.run_prepare(self.event(self.packet(), title="test"))
        self.assertEqual(code, 2)
        self.assertFalse(status["accepted"])
        self.assertIn("Issue title must start", status["reason"])

    def test_non_object_json_is_rejected(self) -> None:
        code, status = self.run_prepare(self.event([1, 2, 3]))
        self.assertEqual(code, 2)
        self.assertFalse(status["accepted"])
        self.assertIn("JSON root must be an object", status["reason"])

    def test_duplicate_rows_detect_task_id_and_fingerprint(self) -> None:
        packet = self.packet()
        row = {"number": 9, "title": "[compute] prior", "body": json.dumps(packet)}
        reason = compute_ticket._duplicate_in_rows(
            [row],
            current_issue=12,
            task_id=packet["task_id"],
            fingerprint=compute_ticket._canonical_sha(packet),
        )
        self.assertIn("duplicate task_id", reason)

    def test_completed_comment_blocks_reopen(self) -> None:
        with patch.object(
            compute_ticket,
            "_trusted_comments",
            return_value=["## COMPUTE_COMPLETED\n"],
        ):
            self.assertEqual(
                compute_ticket._current_issue_errors("owner/repo", 12),
                ["this compute Issue already completed"],
            )


if __name__ == "__main__":
    unittest.main()
