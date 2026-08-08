from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "compute-all-operations-validate.yml"


class CapabilityEnvironmentCIContractTests(unittest.TestCase):
    def test_exhaustive_capability_ci_keeps_causal_runtime_isolated(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("compute-center/requirements-causal.txt", text)
        self.assertNotIn(
            "python -m pip install --disable-pip-version-check --no-input -r compute-center/requirements-causal.txt",
            text,
        )
        self.assertIn("Prepare fixed isolated causal capability environment", text)
        self.assertIn("compute-center/capability_environment.py prepare", text)
        self.assertIn("causal-capability-environment.json", text)
        self.assertIn("'dowhy': '0.14'", text)
        self.assertIn("'scipy': '1.15.3'", text)
        self.assertIn("'causal_runtime_isolation': 'fixed-venv'", text)


if __name__ == "__main__":
    unittest.main()
