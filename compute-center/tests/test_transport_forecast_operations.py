from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compute_runner import ComputeError  # noqa: E402
from transport_forecast_operations import (  # noqa: E402
    statsforecast_batch,
    sumo_micro_simulation,
)


def daily_series() -> list[dict[str, object]]:
    return [
        {"unique_id": "region-a", "ds": f"2026-01-{day:02d}", "y": float(day)}
        for day in range(1, 16)
    ]


class TransportForecastTests(unittest.TestCase):
    def test_statsforecast_naive_forecast_is_bounded(self) -> None:
        result = statsforecast_batch(
            {
                "series": daily_series(),
                "frequency": "D",
                "horizon": 3,
                "season_length": 1,
                "models": ["Naive"],
                "levels": [80],
            }
        )
        self.assertEqual(result["mode"], "statsforecast_batch")
        self.assertEqual(result["series_count"], 1)
        self.assertEqual(result["horizon"], 3)
        self.assertEqual(len(result["forecast_rows"]), 3)
        self.assertEqual(result["network_policy"], "deny")
        self.assertEqual(result["model_calls"], 0)

    def test_statsforecast_rejects_unknown_models(self) -> None:
        with self.assertRaises(ComputeError):
            statsforecast_batch(
                {
                    "series": daily_series(),
                    "frequency": "D",
                    "horizon": 2,
                    "season_length": 1,
                    "models": ["UnregisteredModel"],
                    "levels": [80],
                }
            )

    def test_sumo_tiny_corridor_runs_without_external_paths(self) -> None:
        result = sumo_micro_simulation(
            {
                "nodes": [
                    {"id": "n0", "x": 0.0, "y": 0.0},
                    {"id": "n1", "x": 100.0, "y": 0.0},
                    {"id": "n2", "x": 200.0, "y": 0.0},
                ],
                "edges": [
                    {"id": "e0", "from": "n0", "to": "n1", "lanes": 1, "speed_mps": 13.89},
                    {"id": "e1", "from": "n1", "to": "n2", "lanes": 1, "speed_mps": 13.89},
                ],
                "routes": [{"id": "r0", "edges": ["e0", "e1"]}],
                "flows": [
                    {
                        "id": "f0",
                        "route": "r0",
                        "begin": 0,
                        "end": 120,
                        "vehicles_per_hour": 360,
                    }
                ],
                "duration_seconds": 180,
                "seed": 42,
                "timeout_seconds": 60,
            }
        )
        self.assertEqual(result["mode"], "sumo_micro_simulation")
        self.assertGreater(result["completed_trips"], 0)
        self.assertEqual(result["network_policy"], "deny")
        self.assertFalse(result["arbitrary_commands_allowed"])
        self.assertFalse(result["arbitrary_paths_allowed"])


if __name__ == "__main__":
    unittest.main()
