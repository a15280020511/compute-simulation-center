from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compute_ticket


class SingleTaskGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = "owner/repo"
        self.issue_rows = [
            {"number": 10, "title": "[compute] active", "body": "{}"},
            {"number": 11, "title": "[compute] completed", "body": "{}"},
            {"number": 12, "title": "ordinary issue", "body": "{}"},
            {"number": 13, "title": "[compute] current", "body": "{}"},
            {"number": 14, "title": "[compute] pull request", "body": "{}", "pull_request": {}},
        ]

    def _comments(self, repo: str, issue_number: int):
        self.assertEqual(repo, self.repo)
        mapping = {
            10: ["## COMPUTE_ACCEPTED\n- Accepted: `true`"],
            11: [
                "## COMPUTE_ACCEPTED\n- Accepted: `true`",
                "## COMPUTE_COMPLETED\n- Status: success",
            ],
            13: ["## COMPUTE_ACCEPTED\n- Accepted: `true`"],
        }
        return mapping.get(issue_number, [])

    def test_active_issue_is_detected_and_current_issue_is_ignored(self):
        with patch.object(compute_ticket, "_trusted_comments", side_effect=self._comments):
            number = compute_ticket._active_issue_number(
                self.issue_rows,
                current_issue=13,
                repo=self.repo,
            )
        self.assertEqual(number, 10)

    def test_terminal_tasks_do_not_hold_the_slot(self):
        rows = [{"number": 11, "title": "[compute] completed", "body": "{}"}]
        with patch.object(compute_ticket, "_trusted_comments", side_effect=self._comments):
            number = compute_ticket._active_issue_number(
                rows,
                current_issue=13,
                repo=self.repo,
            )
        self.assertIsNone(number)

    def test_untrusted_or_unaccepted_issue_does_not_hold_the_slot(self):
        rows = [{"number": 20, "title": "[compute] waiting", "body": "{}"}]
        with patch.object(compute_ticket, "_trusted_comments", return_value=[]):
            number = compute_ticket._active_issue_number(
                rows,
                current_issue=13,
                repo=self.repo,
            )
        self.assertIsNone(number)

    def test_active_task_reason_scans_open_issue_pages(self):
        def api(url: str):
            if "state=open" in url and "page=1" in url:
                return self.issue_rows
            raise AssertionError(url)

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}, clear=False),
            patch.object(compute_ticket, "_api_json", side_effect=api),
            patch.object(compute_ticket, "_trusted_comments", side_effect=self._comments),
        ):
            reason = compute_ticket._active_task_reason(self.repo, 13)
        self.assertEqual(
            reason,
            "another compute task is already accepted and active in Issue #10",
        )

    def test_no_token_keeps_local_validation_network_free(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(compute_ticket._active_task_reason(self.repo, 13), "")


if __name__ == "__main__":
    unittest.main()
