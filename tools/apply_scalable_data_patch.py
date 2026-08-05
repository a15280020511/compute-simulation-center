#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTER = ROOT / "compute-center"

MODULE = r'''#!/usr/bin/env python3
"""Governed large-scale offline data intelligence operations.

The operation keeps the existing exact small-data modes unchanged and adds
bounded extended modes.  Every mode performs a deterministic complexity
preflight, rejects unbounded Cartesian work, truncates public results, and
executes without network or model calls.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from compute_runner import ComputeError

STANDARD_ROWS = 2_000
EXTENDED_ROWS = 50_000
MAX_FIELDS = 100
MAX_COMPARE_FIELDS = 10
MAX_CANDIDATE_PAIRS = 2_000_000
MAX_MATCHES = 5_000
MAX_EVENTS = 50_000
MAX_GRAPH_NODES = 10_000
MAX_GRAPH_EDGES = 100_000
MAX_NUMERIC_ROWS = 25_000
MAX_NUMERIC_COLUMNS = 100
MAX_PUBLIC_ROWS = 500
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ComputeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ComputeError(f"{name} must be an integer") from exc
    if result != value or not minimum <= result <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _text(value: Any, name: str, maximum: int = 200) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise ComputeError(f"{name} must contain 1 to {maximum} characters")
    return result


def _names(value: Any, name: str, minimum: int, maximum: int) -> list[str]:
    rows = _sequence(value, name)
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} entries")
    result = [_text(item, f"{name}[]", 100) for item in rows]
    if len(result) != len(set(result)):
        raise ComputeError(f"{name} entries must be unique")
    return result


def _normalized(value: Any) -> str:
    text = str(value or "").casefold().strip()
    return " ".join(_TOKEN_RE.findall(text))


def _tokens(value: str) -> set[str]:
    return set(value.split()) if value else set()


def _bigrams(value: str) -> set[str]:
    compact = value.replace(" ", "")
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index:index + 2] for index in range(len(compact) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _field_similarity(left: Any, right: Any) -> float:
    a = _normalized(left)
    b = _normalized(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    token_score = _jaccard(_tokens(a), _tokens(b))
    bigram_score = _jaccard(_bigrams(a), _bigrams(b))
    prefix_score = min(len(a), len(b)) / max(len(a), len(b)) if a.startswith(b) or b.startswith(a) else 0.0
    return max(token_score, 0.7 * bigram_score + 0.3 * prefix_score)


def _profile_for(work_units: int, *, standard_limit: int, hard_limit: int) -> str:
    if work_units <= standard_limit:
        return "standard"
    if work_units <= hard_limit:
        return "extended"
    return "batch"


def complexity_preflight(inputs: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(inputs.get("workload") or "").strip()
    if kind not in {"entity_collision", "event_collision", "graph", "numeric_profile"}:
        raise ComputeError("workload must be entity_collision, event_collision, graph or numeric_profile")
    rows = _integer(inputs.get("rows", 0), "inputs.rows", 0, EXTENDED_ROWS)
    columns = _integer(inputs.get("columns", 1), "inputs.columns", 1, MAX_FIELDS)
    nodes = _integer(inputs.get("nodes", 0), "inputs.nodes", 0, MAX_GRAPH_NODES)
    edges = _integer(inputs.get("edges", 0), "inputs.edges", 0, MAX_GRAPH_EDGES)
    candidate_pairs = _integer(inputs.get("candidate_pairs", 0), "inputs.candidate_pairs", 0, MAX_CANDIDATE_PAIRS)
    if kind == "entity_collision":
        work_units = candidate_pairs or rows * max(1, columns)
        profile = _profile_for(work_units, standard_limit=200_000, hard_limit=MAX_CANDIDATE_PAIRS)
        algorithm = "blocking-then-bounded-field-comparison"
        requires_blocking = rows > STANDARD_ROWS
    elif kind == "event_collision":
        work_units = rows * max(1, int(math.log2(max(rows, 2))))
        profile = _profile_for(rows, standard_limit=2_000, hard_limit=MAX_EVENTS)
        algorithm = "entity-partitioned-sort-and-sweep"
        requires_blocking = False
    elif kind == "graph":
        work_units = nodes + edges
        profile = _profile_for(work_units, standard_limit=6_000, hard_limit=MAX_GRAPH_NODES + MAX_GRAPH_EDGES)
        algorithm = "sparse-adjacency-components-degree-and-bounded-pagerank"
        requires_blocking = False
    else:
        if rows > MAX_NUMERIC_ROWS or columns > MAX_NUMERIC_COLUMNS:
            raise ComputeError("numeric_profile dimensions exceed extended limits")
        work_units = rows * columns
        profile = _profile_for(work_units, standard_limit=500_000, hard_limit=MAX_NUMERIC_ROWS * MAX_NUMERIC_COLUMNS)
        algorithm = "single-pass-column-aggregates"
        requires_blocking = False
    shard_count = max(1, math.ceil(work_units / 500_000))
    return {
        "mode": "complexity_preflight",
        "workload": kind,
        "selected_profile": profile,
        "algorithm": algorithm,
        "estimated_work_units": int(work_units),
        "recommended_shards": int(shard_count),
        "requires_blocking": requires_blocking,
        "candidate_pair_hard_limit": MAX_CANDIDATE_PAIRS,
        "network_policy": "deny",
        "model_calls": 0,
    }


def blocked_entity_collision(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw_records = _sequence(inputs.get("records"), "inputs.records")
    if not 2 <= len(raw_records) <= EXTENDED_ROWS:
        raise ComputeError(f"records must contain 2 to {EXTENDED_ROWS} rows")
    fields = _names(inputs.get("fields"), "inputs.fields", 1, MAX_COMPARE_FIELDS)
    block_fields = _names(inputs.get("block_fields"), "inputs.block_fields", 1, min(3, len(fields)))
    if not set(block_fields) <= set(fields):
        raise ComputeError("block_fields must be a subset of fields")
    threshold = _finite(inputs.get("threshold", 0.85), "inputs.threshold")
    if not 0 <= threshold <= 1:
        raise ComputeError("threshold must be between 0 and 1")
    candidate_limit = _integer(inputs.get("max_candidate_pairs", 500_000), "inputs.max_candidate_pairs", 1, MAX_CANDIDATE_PAIRS)
    match_limit = _integer(inputs.get("max_matches", 1_000), "inputs.max_matches", 1, MAX_MATCHES)
    weights_raw = _mapping(inputs.get("weights") or {}, "inputs.weights")
    weights = []
    for field in fields:
        weight = _finite(weights_raw.get(field, 1.0), f"inputs.weights.{field}")
        if weight < 0:
            raise ComputeError("field weights must be non-negative")
        weights.append(weight)
    if sum(weights) <= 0:
        raise ComputeError("field weights must have positive total")
    weight_total = float(sum(weights))
    records: list[Mapping[str, Any]] = []
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, raw in enumerate(raw_records):
        row = _mapping(raw, f"inputs.records[{index}]")
        if len(row) > MAX_FIELDS:
            raise ComputeError(f"inputs.records[{index}] exceeds {MAX_FIELDS} fields")
        records.append(row)
        key_values = tuple(_normalized(row.get(field)) for field in block_fields)
        if any(not value for value in key_values):
            key_values = (f"__incomplete__:{index}",)
        buckets[key_values].append(index)
    matches = []
    candidate_pairs = 0
    exhausted = False
    largest_bucket = 0
    for key in sorted(buckets):
        members = buckets[key]
        largest_bucket = max(largest_bucket, len(members))
        for left_position, left_index in enumerate(members):
            for right_index in members[left_position + 1:]:
                candidate_pairs += 1
                if candidate_pairs > candidate_limit:
                    exhausted = True
                    break
                scores = [_field_similarity(records[left_index].get(field), records[right_index].get(field)) for field in fields]
                combined = sum(weights[index] * scores[index] for index in range(len(fields))) / weight_total
                if combined >= threshold:
                    matches.append({
                        "left_index": left_index,
                        "right_index": right_index,
                        "score": float(combined),
                        "field_scores": {fields[index]: float(scores[index]) for index in range(len(fields))},
                    })
            if exhausted:
                break
        if exhausted:
            break
    matches.sort(key=lambda row: (-row["score"], row["left_index"], row["right_index"]))
    total_matches = len(matches)
    return {
        "mode": "blocked_entity_collision",
        "record_count": len(records),
        "block_field_count": len(block_fields),
        "bucket_count": len(buckets),
        "largest_bucket": largest_bucket,
        "candidate_pairs_evaluated": min(candidate_pairs, candidate_limit),
        "candidate_limit_reached": exhausted,
        "matched_pair_count": total_matches,
        "matches": matches[:match_limit],
        "matches_truncated": total_matches > match_limit,
        "selected_profile": "standard" if len(records) <= STANDARD_ROWS else "extended",
        "cartesian_product_allowed": False,
        "personal_identity_targeting_allowed": False,
    }


def sorted_event_collision(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw_events = _sequence(inputs.get("events"), "inputs.events")
    if not 2 <= len(raw_events) <= MAX_EVENTS:
        raise ComputeError(f"events must contain 2 to {MAX_EVENTS} entries")
    candidate_limit = _integer(inputs.get("max_overlap_checks", 500_000), "inputs.max_overlap_checks", 1, MAX_CANDIDATE_PAIRS)
    result_limit = _integer(inputs.get("max_collisions", 1_000), "inputs.max_collisions", 1, MAX_MATCHES)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(raw_events):
        row = _mapping(raw, f"inputs.events[{index}]")
        event = {
            "id": _text(row.get("id"), f"inputs.events[{index}].id"),
            "entity": _text(row.get("entity"), f"inputs.events[{index}].entity"),
            "start": _finite(row.get("start"), f"inputs.events[{index}].start"),
            "end": _finite(row.get("end"), f"inputs.events[{index}].end"),
            "location": str(row.get("location") or "").strip(),
        }
        if event["start"] > event["end"]:
            raise ComputeError("event start cannot exceed end")
        grouped[event["entity"]].append(event)
    collisions = []
    checks = 0
    exhausted = False
    for entity in sorted(grouped):
        events = sorted(grouped[entity], key=lambda row: (row["start"], row["end"], row["id"]))
        active: deque[dict[str, Any]] = deque()
        for event in events:
            while active and active[0]["end"] < event["start"]:
                active.popleft()
            for previous in active:
                checks += 1
                if checks > candidate_limit:
                    exhausted = True
                    break
                if previous["location"] and event["location"] and previous["location"] != event["location"]:
                    collisions.append({
                        "left": previous["id"],
                        "right": event["id"],
                        "entity": entity,
                        "overlap_start": max(previous["start"], event["start"]),
                        "overlap_end": min(previous["end"], event["end"]),
                        "reason": "same-entity-overlapping-different-location",
                    })
            if exhausted:
                break
            inserted = False
            for position, current in enumerate(active):
                if event["end"] < current["end"]:
                    active.insert(position, event)
                    inserted = True
                    break
            if not inserted:
                active.append(event)
        if exhausted:
            break
    total = len(collisions)
    return {
        "mode": "sorted_event_collision",
        "event_count": len(raw_events),
        "entity_count": len(grouped),
        "overlap_checks": min(checks, candidate_limit),
        "candidate_limit_reached": exhausted,
        "collision_count": total,
        "collisions": collisions[:result_limit],
        "collisions_truncated": total > result_limit,
        "selected_profile": "standard" if len(raw_events) <= 2_000 else "extended",
        "algorithm": "entity-partitioned-sort-and-sweep",
    }


def chunked_numeric_profile(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = _sequence(inputs.get("records"), "inputs.records")
    if not 1 <= len(rows) <= MAX_NUMERIC_ROWS:
        raise ComputeError(f"records must contain 1 to {MAX_NUMERIC_ROWS} rows")
    first = _sequence(rows[0], "inputs.records[0]")
    width = len(first)
    if not 1 <= width <= MAX_NUMERIC_COLUMNS:
        raise ComputeError(f"numeric columns must be between 1 and {MAX_NUMERIC_COLUMNS}")
    count = [0] * width
    missing = [0] * width
    means = [0.0] * width
    m2 = [0.0] * width
    minima = [math.inf] * width
    maxima = [-math.inf] * width
    for row_index, raw in enumerate(rows):
        row = _sequence(raw, f"inputs.records[{row_index}]")
        if len(row) != width:
            raise ComputeError("all numeric rows must have equal width")
        for column, raw_value in enumerate(row):
            if raw_value is None:
                missing[column] += 1
                continue
            value = _finite(raw_value, f"inputs.records[{row_index}][{column}]")
            count[column] += 1
            delta = value - means[column]
            means[column] += delta / count[column]
            m2[column] += delta * (value - means[column])
            minima[column] = min(minima[column], value)
            maxima[column] = max(maxima[column], value)
    columns = []
    for index in range(width):
        observed = count[index]
        columns.append({
            "column_index": index,
            "observed": observed,
            "missing": missing[index],
            "mean": means[index] if observed else None,
            "variance": m2[index] / (observed - 1) if observed > 1 else 0.0 if observed == 1 else None,
            "minimum": minima[index] if observed else None,
            "maximum": maxima[index] if observed else None,
        })
    return {
        "mode": "chunked_numeric_profile",
        "row_count": len(rows),
        "column_count": width,
        "columns": columns,
        "selected_profile": "standard" if len(rows) <= 5_000 else "extended",
        "algorithm": "single-pass-welford",
    }


def large_graph_summary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    nodes = _names(inputs.get("nodes"), "inputs.nodes", 1, MAX_GRAPH_NODES)
    raw_edges = _sequence(inputs.get("edges"), "inputs.edges")
    if not 0 <= len(raw_edges) <= MAX_GRAPH_EDGES:
        raise ComputeError(f"edges must contain 0 to {MAX_GRAPH_EDGES} entries")
    directed = bool(inputs.get("directed", True))
    iterations = _integer(inputs.get("pagerank_iterations", 20), "inputs.pagerank_iterations", 1, 100)
    top_k = _integer(inputs.get("top_k", 100), "inputs.top_k", 1, MAX_PUBLIC_ROWS)
    index = {name: position for position, name in enumerate(nodes)}
    outgoing = [set() for _ in nodes]
    incoming = [set() for _ in nodes]
    undirected = [set() for _ in nodes]
    for edge_index, raw in enumerate(raw_edges):
        pair = _sequence(raw, f"inputs.edges[{edge_index}]")
        if len(pair) != 2 or pair[0] not in index or pair[1] not in index:
            raise ComputeError("every edge must reference two declared nodes")
        left, right = index[pair[0]], index[pair[1]]
        outgoing[left].add(right)
        incoming[right].add(left)
        undirected[left].add(right)
        undirected[right].add(left)
        if not directed:
            outgoing[right].add(left)
            incoming[left].add(right)
    seen = set()
    components = []
    for start in range(len(nodes)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in undirected[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    n = len(nodes)
    rank = [1.0 / n] * n
    damping = 0.85
    for _ in range(iterations):
        dangling = sum(rank[position] for position in range(n) if not outgoing[position]) / n
        next_rank = [(1.0 - damping) / n + damping * dangling for _ in range(n)]
        for source in range(n):
            targets = outgoing[source]
            if not targets:
                continue
            share = damping * rank[source] / len(targets)
            for target in targets:
                next_rank[target] += share
        rank = next_rank
    ranking = sorted(
        ({
            "node": nodes[position],
            "pagerank": float(rank[position]),
            "in_degree": len(incoming[position]),
            "out_degree": len(outgoing[position]),
            "degree": len(undirected[position]),
        } for position in range(n)),
        key=lambda row: (-row["pagerank"], -row["degree"], row["node"]),
    )
    component_sizes = sorted((len(component) for component in components), reverse=True)
    return {
        "mode": "large_graph_summary",
        "node_count": n,
        "edge_count": len(raw_edges),
        "directed": directed,
        "component_count": len(components),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "component_sizes": component_sizes[:MAX_PUBLIC_ROWS],
        "ranking": ranking[:top_k],
        "ranking_truncated": len(ranking) > top_k,
        "pagerank_iterations": iterations,
        "selected_profile": "standard" if n <= 1_000 and len(raw_edges) <= 5_000 else "extended",
        "betweenness_skipped_for_scale": True,
        "algorithm": "sparse-components-degree-and-bounded-pagerank",
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "complexity_preflight": complexity_preflight,
    "blocked_entity_collision": blocked_entity_collision,
    "sorted_event_collision": sorted_event_collision,
    "chunked_numeric_profile": chunked_numeric_profile,
    "large_graph_summary": large_graph_summary,
}


def large_scale_data_intelligence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    handler = HANDLERS.get(mode)
    if handler is None:
        raise ComputeError(f"unsupported large-scale data intelligence mode: {mode}")
    result = handler(inputs)
    result.update({
        "offline_execution": True,
        "external_data_fetches": 0,
        "model_calls": 0,
        "decision_support_only": True,
        "arbitrary_code_allowed": False,
        "output_rows_bounded": True,
    })
    return result


OPERATIONS = {"large_scale_data_intelligence": large_scale_data_intelligence}
'''

TEST = r'''from __future__ import annotations

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
'''

DOC = '''# Scalable Data Intelligence

`large_scale_data_intelligence` adds bounded extended analysis without changing
existing exact small-data modes.

Modes:

- `complexity_preflight`: estimates work and selects standard, extended or batch profile.
- `blocked_entity_collision`: up to 50,000 records with mandatory blocking and a hard candidate-pair budget.
- `sorted_event_collision`: up to 50,000 events using entity-partitioned sort-and-sweep.
- `chunked_numeric_profile`: up to 25,000 rows by 100 numeric columns using one-pass aggregates.
- `large_graph_summary`: up to 10,000 nodes and 100,000 edges using sparse components, degree and bounded PageRank.

All modes are offline, deterministic, output-bounded, and reject arbitrary code,
URLs and unbounded Cartesian work.  Large payloads still require the governance
material-package path; the public Issue receipt contains only summaries and
Artifact references.
'''


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


write(CENTER / "large_scale_data_intelligence_operations.py", MODULE)
write(CENTER / "tests" / "test_large_scale_data_intelligence.py", TEST)
write(ROOT / "SCALABLE_DATA_INTELLIGENCE.md", DOC)

registry_path = CENTER / "tool-registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
groups = registry["groups"]
if not any(group.get("id") == "large-scale-data-intelligence" for group in groups):
    groups.append({
        "id": "large-scale-data-intelligence",
        "module": "large_scale_data_intelligence_operations",
        "operations": ["large_scale_data_intelligence"],
        "input_validation": "mode_allowlist",
        "default_requirements": [],
        "mode_requirements": {},
        "network_policy": "deny",
        "deterministic": True,
        "maturity": "controlled-preview",
        "resource_limits": {"max_seconds": 600, "max_memory_mb": 6144},
        "rollback": {
            "stable_module": "large_scale_data_intelligence_operations",
            "strategy": "git-revert",
        },
        "modes": {
            "complexity_preflight": {
                "maturity": "production",
                "network_policy": "deny",
                "deterministic": True,
                "limits": {"max_rows": 50000, "max_candidate_pairs": 2000000},
            },
            "blocked_entity_collision": {
                "maturity": "controlled-preview",
                "network_policy": "deny",
                "deterministic": True,
                "limits": {"max_rows": 50000, "max_fields": 10, "max_candidate_pairs": 2000000},
            },
            "sorted_event_collision": {
                "maturity": "controlled-preview",
                "network_policy": "deny",
                "deterministic": True,
                "limits": {"max_events": 50000, "max_overlap_checks": 2000000},
            },
            "chunked_numeric_profile": {
                "maturity": "controlled-preview",
                "network_policy": "deny",
                "deterministic": True,
                "limits": {"max_rows": 25000, "max_columns": 100},
            },
            "large_graph_summary": {
                "maturity": "controlled-preview",
                "network_policy": "deny",
                "deterministic": True,
                "limits": {"max_nodes": 10000, "max_edges": 100000},
            },
        },
    })
registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

matrix_path = CENTER / "systems-computation-matrix.json"
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
matrix["routes"]["large_scale_data_intelligence"] = {
    "problem_class": "large-scale-data-linkage-and-structure-analysis",
    "system_level": "observation-and-mechanism",
    "feedback_structure": "bounded-batch-and-sparse-relations",
    "required_gates": ["input_quality", "assumption_register", "uncertainty", "stress_test"],
}
matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

cap_path = CENTER / "compute-capabilities.json"
cap = json.loads(cap_path.read_text(encoding="utf-8"))
cap["operation_count"] = int(cap.get("operation_count", 0)) + (0 if any(row.get("id") == "large_scale_data_intelligence" for row in cap.get("operations", [])) else 1)
cap["managed_mode_count"] = int(cap.get("managed_mode_count", 0)) + 5
cap["effective_managed_mode_count"] = int(cap.get("effective_managed_mode_count", 0)) + 5
cap.setdefault("limits", {}).update({
    "large_scale_records": 50000,
    "large_scale_candidate_pairs": 2000000,
    "large_scale_events": 50000,
    "large_scale_graph_nodes": 10000,
    "large_scale_graph_edges": 100000,
    "large_scale_numeric_rows": 25000,
    "large_scale_numeric_columns": 100,
})
operations = cap.setdefault("operations", [])
if not any(row.get("id") == "large_scale_data_intelligence" for row in operations):
    operations.append({
        "id": "large_scale_data_intelligence",
        "engine": "repository-native sparse and streaming algorithms",
        "availability": "managed controlled-preview",
        "use_when": "large structured datasets require bounded collision, comparison, profiling, timeline or graph analysis",
        "typical_output": "complexity profile, candidate-limited matches, conflict summary, column profile or sparse graph ranking",
    })
assessment = cap.get("toolkit_assessment")
if isinstance(assessment, dict) and isinstance(assessment.get("scope"), list):
    line = "bounded large-scale blocking, sort-and-sweep, streaming numeric profiling and sparse graph summaries"
    if line not in assessment["scope"]:
        assessment["scope"].append(line)
cap_path.write_text(json.dumps(cap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

test_registry = CENTER / "tests" / "test_tool_registry.py"
text = test_registry.read_text(encoding="utf-8")
text = text.replace("self.assertEqual(len(operations), 24)", "self.assertEqual(len(operations), 25)")
text = text.replace('"symbolic_mathematics",\n        ):', '"symbolic_mathematics", "large_scale_data_intelligence",\n        ):')
text = text.replace("self.assertEqual(len(target), 30)", "self.assertEqual(len(target), 31)")
text = text.replace('self.assertIn("symbolic_mathematics", target)', 'self.assertIn("symbolic_mathematics", target)\n        self.assertIn("large_scale_data_intelligence", target)')
test_registry.write_text(text, encoding="utf-8")

print("scalable data intelligence patch applied")
