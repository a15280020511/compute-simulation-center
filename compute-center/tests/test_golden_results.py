from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("compute_runner_golden", ROOT / "compute_runner.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def resolve_path(value, path: str):
    current = value
    for token in path.split("."):
        if isinstance(current, list):
            current = current[int(token)]
        else:
            current = current[token]
    return current


class GoldenResultTests(unittest.TestCase):
    def test_all_golden_cases_match_within_tolerance(self) -> None:
        fixture = json.loads((ROOT / "golden-results.json").read_text(encoding="utf-8"))
        tolerance = float(fixture["absolute_tolerance"])
        self.assertEqual(fixture["version"], 1)
        self.assertGreaterEqual(len(fixture["cases"]), 4)
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]), tempfile.TemporaryDirectory() as directory:
                result = runner.run_ticket(case["ticket"], Path(directory))
                for path, expected in case["expected"].items():
                    actual = resolve_path(result, path)
                    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                        self.assertTrue(
                            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance),
                            f"{case['name']} {path}: expected {expected!r}, got {actual!r}",
                        )
                    else:
                        self.assertEqual(actual, expected, f"{case['name']} {path}")


if __name__ == "__main__":
    unittest.main()
