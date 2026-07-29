import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

LITERATURE_DIR = Path(__file__).resolve().parents[1] / "literature-evidence"
sys.path.insert(0, str(LITERATURE_DIR))

import literature_ticket


class LiteratureTicketTests(unittest.TestCase):
    def _event(self, root: Path, body: dict) -> Path:
        path = root / "event.json"
        path.write_text(
            json.dumps(
                {
                    "issue": {
                        "number": 7,
                        "html_url": "https://github.example/issues/7",
                        "body": json.dumps(body, ensure_ascii=False),
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_prepare_accepts_bounded_ticket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts"
            event = self._event(
                root,
                {
                    "task_id": "literature-20260729-001",
                    "query": "urban mobility demand forecasting",
                    "per_page": 5,
                    "research_context": {"geography": "China"},
                },
            )
            github_output = root / "github-output.txt"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output)}):
                self.assertEqual(literature_ticket.prepare(event, output), 0)
            ticket = json.loads((output / "ticket.json").read_text(encoding="utf-8"))
            self.assertEqual(ticket["network_policy"], "allowlisted-literature-only")
            self.assertFalse(ticket["numeric_dispatch_allowed"])
            self.assertEqual(len(ticket["semantic_fingerprint"]), 64)
            self.assertIn("accepted=true", github_output.read_text(encoding="utf-8"))

    def test_prepare_rejects_forbidden_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts"
            event = self._event(
                root,
                {
                    "task_id": "literature-20260729-002",
                    "query": "file://etc/passwd",
                },
            )
            self.assertEqual(literature_ticket.prepare(event, output), 2)
            status = json.loads((output / "ticket-status.json").read_text(encoding="utf-8"))
            self.assertFalse(status["accepted"])

    def test_execute_freezes_candidate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts"
            ticket = {
                "task_id": "literature-20260729-003",
                "query": "policy evaluation",
                "per_page": 2,
                "research_context": {"outcome": "employment"},
                "semantic_fingerprint": "a" * 64,
                "network_policy": "allowlisted-literature-only",
                "numeric_dispatch_allowed": False,
            }
            ticket_path = root / "ticket.json"
            ticket_path.write_text(json.dumps(ticket), encoding="utf-8")
            fake_package = {
                "schema_version": "literature-evidence-package-v1",
                "records": [{"doi": "10.1/example", "parameter_status": "literature-raw-result-only"}],
                "sources": ["OpenAlex", "Crossref"],
                "automatic_parameter_promotion_allowed": False,
            }
            with patch.object(literature_ticket, "build", return_value=fake_package):
                self.assertEqual(literature_ticket.execute(ticket_path, output), 0)
            package = json.loads(
                (output / "literature-evidence-package.json").read_text(encoding="utf-8")
            )
            self.assertEqual(package["evidence_state"], "frozen-candidate-evidence")
            self.assertFalse(package["numeric_dispatch_allowed"])
            self.assertFalse(package["automatic_parameter_promotion_allowed"])
            self.assertEqual(len(package["frozen_package_sha256"]), 64)
            manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(manifest["files"]), 3)


if __name__ == "__main__":
    unittest.main()
