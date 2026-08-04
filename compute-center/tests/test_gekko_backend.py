#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from think_tank_decision_operations import algebraic_resource_optimization


def _gekko_installed() -> bool:
    try:
        version("gekko")
    except PackageNotFoundError:
        return False
    return True


@unittest.skipUnless(
    _gekko_installed(),
    "GEKKO optional engine is validated in its isolated decision-pack workflow",
)
class GekkoBackendTests(unittest.TestCase):
    def test_local_resource_optimization(self) -> None:
        result = algebraic_resource_optimization(
            {
                "objective": [3.0, 2.0],
                "constraint_matrix": [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                "constraint_bounds": [4.0, 2.0, 3.0],
                "maximize": True,
                "solver_engine": "gekko",
            }
        )
        self.assertEqual(result["mode"], "algebraic_resource_optimization")
        self.assertEqual(result["termination"], "optimal-local")
        self.assertIs(result["engines"]["remote"], False)
        self.assertAlmostEqual(result["objective_value"], 10.0, places=4)
        self.assertAlmostEqual(result["decision"][0], 2.0, places=4)
        self.assertAlmostEqual(result["decision"][1], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
