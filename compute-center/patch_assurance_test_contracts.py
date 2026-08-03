#!/usr/bin/env python3
"""One-shot synchronization of assurance-mode aggregate test contracts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> int:
    think_test = ROOT / "compute-center/tests/test_think_tank_operations.py"
    replace_exact(
        think_test,
        "from capability_manager import requirements_for_ticket, runtime_plan\n",
        "from assurance_operations import HANDLERS as ASSURANCE_HANDLERS\nfrom capability_manager import requirements_for_ticket, runtime_plan\n",
    )
    replace_exact(
        think_test,
        "        self.assertEqual(len(SUPPORTED_MODES), EXPECTED_EXTENSION_MODES)\n",
        "        self.assertTrue(set(SUPPORTED_MODES).isdisjoint(set(ASSURANCE_HANDLERS)))\n        self.assertEqual(len(set(SUPPORTED_MODES) | set(ASSURANCE_HANDLERS)), EXPECTED_EXTENSION_MODES)\n",
    )

    gateway_test = ROOT / "compute-center/tests/test_decision_intelligence_gateway.py"
    replace_exact(
        gateway_test,
        "from decision_intelligence_gateway import (  # noqa: E402\n",
        "from assurance_operations import HANDLERS as ASSURANCE_HANDLERS  # noqa: E402\nfrom decision_intelligence_gateway import (  # noqa: E402\n",
    )
    replace_exact(
        gateway_test,
        "        self.assertTrue(set(THINK_TANK_MODES) <= set(ALL_SUPPORTED_MODES))\n        self.assertTrue(set(SUPPORTED_MODES).isdisjoint(set(THINK_TANK_MODES)))\n        self.assertEqual(len(SUPPORTED_MODES), 22)\n        self.assertEqual(len(THINK_TANK_MODES), 53)\n        self.assertEqual(len(ALL_SUPPORTED_MODES), 75)\n",
        "        self.assertTrue(set(THINK_TANK_MODES) <= set(ALL_SUPPORTED_MODES))\n        self.assertTrue(set(ASSURANCE_HANDLERS) <= set(ALL_SUPPORTED_MODES))\n        self.assertTrue(set(SUPPORTED_MODES).isdisjoint(set(THINK_TANK_MODES)))\n        self.assertTrue(set(SUPPORTED_MODES).isdisjoint(set(ASSURANCE_HANDLERS)))\n        self.assertTrue(set(THINK_TANK_MODES).isdisjoint(set(ASSURANCE_HANDLERS)))\n        self.assertEqual(len(SUPPORTED_MODES), 22)\n        self.assertEqual(len(THINK_TANK_MODES), 53)\n        self.assertEqual(len(ASSURANCE_HANDLERS), 8)\n        self.assertEqual(len(ALL_SUPPORTED_MODES), 83)\n",
    )
    print("assurance aggregate test contracts synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
