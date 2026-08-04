from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import compute_ticket


class AbandonedTicketReleaseTests(unittest.TestCase):
    def _row(self, *, state: str = "closed") -> dict[str, object]:
        return {
            "number": 81,
            "title": "[compute] abandoned admission test",
            "body": json.dumps(
                {
                    "task_id": "shared-task",
                    "operation": "descriptive_statistics",
                }
            ),
            "state": state,
        }

    def test_closed_ticket_without_trusted_receipt_releases_identifier(self) -> None:
        with patch.object(compute_ticket, "_trusted_comments", return_value=[]):
            result = compute_ticket._duplicate_in_rows(
                [self._row()],
                current_issue=82,
                task_id="shared-task",
                fingerprint="different",
                retry_issue=77,
            )
        self.assertEqual(result, "")

    def test_open_ticket_without_receipt_still_reserves_identifier(self) -> None:
        with patch.object(compute_ticket, "_trusted_comments", return_value=[]):
            result = compute_ticket._duplicate_in_rows(
                [self._row(state="open")],
                current_issue=82,
                task_id="shared-task",
                fingerprint="different",
                retry_issue=77,
            )
        self.assertIn("duplicate task_id", result)

    def test_closed_accepted_ticket_still_reserves_identifier(self) -> None:
        with patch.object(
            compute_ticket,
            "_trusted_comments",
            return_value=["## COMPUTE_ACCEPTED"],
        ):
            result = compute_ticket._duplicate_in_rows(
                [self._row()],
                current_issue=82,
                task_id="shared-task",
                fingerprint="different",
                retry_issue=77,
            )
        self.assertIn("duplicate task_id", result)


if __name__ == "__main__":
    unittest.main()
