from __future__ import annotations

import unittest
from pathlib import Path


class ComputeWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = Path('.github/workflows/compute-ticket.yml').read_text(encoding='utf-8')

    def test_ownership_restored_between_isolation_and_manifest(self) -> None:
        execute = self.text.index('name: Execute compute inside an OS network namespace')
        restore = self.text.index('name: Restore runner ownership after isolated execution')
        refresh = self.text.index('name: Refresh Artifact manifest after console capture')
        self.assertLess(execute, restore)
        self.assertLess(restore, refresh)

    def test_symlinks_are_rejected_before_recursive_chown(self) -> None:
        self.assertIn('find compute-artifacts -type l -print -quit', self.text)
        self.assertIn('symbolic links are forbidden in compute artifacts', self.text)
        self.assertIn('sudo chown -R -- "$(id -u):$(id -g)" compute-artifacts', self.text)

    def test_delivery_status_includes_ownership_restore(self) -> None:
        self.assertGreaterEqual(self.text.count("steps.restore_ownership.outcome == 'success'"), 1)
        self.assertGreaterEqual(self.text.count("steps.restore_ownership.outcome != 'success'"), 2)
        self.assertIn('OWNERSHIP_OUTCOME: ${{ steps.restore_ownership.outcome }}', self.text)
        self.assertIn('Ownership restore outcome', self.text)


if __name__ == '__main__':
    unittest.main()
