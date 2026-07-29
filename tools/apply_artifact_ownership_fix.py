#!/usr/bin/env python3
"""Apply the reviewed compute Artifact ownership fix to the ticket workflow."""
from __future__ import annotations

from pathlib import Path

WORKFLOW = Path('.github/workflows/compute-ticket.yml')
TEST = Path('compute-center/tests/test_compute_workflow_contract.py')
SELF = Path('tools/apply_artifact_ownership_fix.py')


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'expected exactly one workflow fragment, found {count}: {old[:100]!r}')
    return text.replace(old, new, 1)


def main() -> int:
    text = WORKFLOW.read_text(encoding='utf-8')
    marker = '      - name: Refresh Artifact manifest after console capture\n'
    ownership_step = '''      - name: Restore runner ownership after isolated execution
        id: restore_ownership
        if: always() && steps.prepare.outputs.accepted == 'true'
        continue-on-error: true
        shell: bash
        run: |
          set -euo pipefail
          test -d compute-artifacts
          if find compute-artifacts -type l -print -quit | grep -q .; then
            echo "symbolic links are forbidden in compute artifacts" >&2
            exit 1
          fi
          sudo chown -R -- "$(id -u):$(id -g)" compute-artifacts
          test -w compute-artifacts
          if [[ -e compute-artifacts/artifact-manifest.json ]]; then
            test -w compute-artifacts/artifact-manifest.json
          fi

'''
    text = replace_once(text, marker, ownership_step + marker)
    text = replace_once(
        text,
        "          steps.execute.outcome == 'success' &&\n          steps.refresh_manifest.outcome == 'success' &&",
        "          steps.execute.outcome == 'success' &&\n          steps.restore_ownership.outcome == 'success' &&\n          steps.refresh_manifest.outcome == 'success' &&",
    )
    text = replace_once(
        text,
        "          (steps.execute.outcome != 'success' ||\n           steps.refresh_manifest.outcome != 'success' ||",
        "          (steps.execute.outcome != 'success' ||\n           steps.restore_ownership.outcome != 'success' ||\n           steps.refresh_manifest.outcome != 'success' ||",
    )
    text = replace_once(
        text,
        "          EXECUTE_OUTCOME: ${{ steps.execute.outcome }}\n          MANIFEST_OUTCOME: ${{ steps.refresh_manifest.outcome }}",
        "          EXECUTE_OUTCOME: ${{ steps.execute.outcome }}\n          OWNERSHIP_OUTCOME: ${{ steps.restore_ownership.outcome }}\n          MANIFEST_OUTCOME: ${{ steps.refresh_manifest.outcome }}",
    )
    text = replace_once(
        text,
        "              f\"- Execute outcome: `{os.environ['EXECUTE_OUTCOME']}`\",\n              f\"- Manifest outcome: `{os.environ['MANIFEST_OUTCOME']}`\",",
        "              f\"- Execute outcome: `{os.environ['EXECUTE_OUTCOME']}`\",\n              f\"- Ownership restore outcome: `{os.environ['OWNERSHIP_OUTCOME']}`\",\n              f\"- Manifest outcome: `{os.environ['MANIFEST_OUTCOME']}`\",",
    )
    text = replace_once(
        text,
        "           steps.execute.outcome != 'success' ||\n           steps.refresh_manifest.outcome != 'success' ||",
        "           steps.execute.outcome != 'success' ||\n           steps.restore_ownership.outcome != 'success' ||\n           steps.refresh_manifest.outcome != 'success' ||",
    )
    WORKFLOW.write_text(text, encoding='utf-8')

    TEST.write_text('''from __future__ import annotations

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
''', encoding='utf-8')
    SELF.unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
