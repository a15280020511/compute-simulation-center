from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from large_scale_data_intelligence_operations import large_scale_data_intelligence  # noqa: E402


class LargeScaleDataIntelligenceTests(unittest.TestCase):
    def run_mode(self, mode: str, **inputs):
        return large_scale_data_intelligence({"mode": mode, **inputs})

    def test_preflight_selects_extended_without_unbounded_work(self):
        result = self.run_mode(
            "complexity_preflight",
            workload="entity_collision",
            rows=50_000,
            columns=5,
            candidate_pairs=900_000,
        )
        self.assertEqual(result["selected_profile"], "extended")
        self.assertTrue(result["requires_blocking"])
        self.assertEqual(result["model_calls"], 0)

    def test_blocked_entity_collision_avoids_cross_bucket_cartesian_product(self):
        records = [
            {"region": "福建", "name": "福州永辉超市", "address": "鼓楼区"},
            {"region": "福建", "name": "永辉超市福州店", "address": "福州市鼓楼区"},
            {"region": "上海", "name": "永辉超市", "address": "浦东新区"},
        ]
        result = self.run_mode(
            "blocked_entity_collision",
            records=records,
            fields=["name", "address", "region"],
            block_fields=["region"],
            threshold=0.45,
        )
        self.assertEqual(result["candidate_pairs_evaluated"], 1)
        self.assertFalse(result["cartesian_product_allowed"])
        self.assertEqual(result["matched_pair_count"], 1)

    def test_event_collision_uses_sorted_sweep(self):
        result = self.run_mode(
            "sorted_event_collision",
            events=[
                {"id": "a", "entity": "x", "start": 1, "end": 4, "location": "A"},
                {"id": "b", "entity": "x", "start": 2, "end": 3, "location": "B"},
                {"id": "c", "entity": "x", "start": 5, "end": 6, "location": "C"},
            ],
        )
        self.assertEqual(result["collision_count"], 1)
        self.assertEqual(result["algorithm"], "entity-partitioned-sort-and-sweep")

    def test_numeric_profile_is_single_pass_and_missing_aware(self):
        result = self.run_mode(
            "chunked_numeric_profile",
            records=[[1, 2], [3, None], [5, 6]],
        )
        self.assertEqual(result["row_count"], 3)
        self.assertAlmostEqual(result["columns"][0]["mean"], 3.0)
        self.assertEqual(result["columns"][1]["missing"], 1)

    def test_graph_summary_is_sparse_and_bounded(self):
        result = self.run_mode(
            "large_graph_summary",
            nodes=["a", "b", "c", "d"],
            edges=[["a", "b"], ["b", "c"]],
            top_k=2,
        )
        self.assertEqual(result["component_count"], 2)
        self.assertEqual(len(result["ranking"]), 2)
        self.assertTrue(result["betweenness_skipped_for_scale"])


if __name__ == "__main__":
    unittest.main()
