#!/usr/bin/env python3
"""Governed indirect-intelligence fusion for already-relayed structured evidence.

This mode is collection-free. It never reaches the evidence center, opens URLs,
calls models, or accepts ticket-supplied code. OR-Tools chooses among a fixed,
repository-controlled stage catalog under a dynamic cost budget; NetworkX validates
the resulting serial DAG. Every conclusion remains explicitly typed as DIRECT,
LINKED, INFERRED, or CONTRADICTED. Probabilistic output is never promoted to fact.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from bayesian_network_operations import bayesian_network_inference
from compute_runner import ComputeError
from strategic_policy_intelligence_operations import (
    claim_evidence_contradiction,
    datasketch_set_similarity,
    igraph_link_analysis,
    owlready2_ontology_summary,
    problog_evidence_probability,
    pyshacl_graph_validation,
    rapidfuzz_record_collision,
    rdflib_claim_evidence_graph,
    splink_entity_resolution,
)
from think_tank_global_operations import pm4py_directly_follows

MODE = "indirect_intelligence_analysis"
ANALYSIS_CLASSES = {"DIRECT", "LINKED", "INFERRED", "CONTRADICTED"}
MAX_EVIDENCE = 500
MAX_ENTITY_RECORDS = 500
MAX_RELATIONS = 5_000
MAX_CASES = 1_000
MAX_STAGES = 8
MAX_DATASKETCH_SETS = 100
MAX_PROBLOG_FACTS = 100
MAX_RULES = 50
MAX_IGRAPH_NODES = 1_000
STAGE_ORDER = (
    "name_normalization",
    "similarity_collision",
    "entity_resolution",
    "knowledge_graph",
    "graph_analysis",
    "process_mining",
    "probabilistic_inference",
    "contradiction_check",
)
STAGE_POLICY = {
    "name_normalization": {"utility": 24, "cost": 1},
    "similarity_collision": {"utility": 18, "cost": 2},
    "entity_resolution": {"utility": 31, "cost": 4},
    "knowledge_graph": {"utility": 24, "cost": 3},
    "graph_analysis": {"utility": 27, "cost": 3},
    "process_mining": {"utility": 30, "cost": 4},
    "probabilistic_inference": {"utility": 43, "cost": 5},
    "contradiction_check": {"utility": 50, "cost": 2},
}
MAX_STAGE_COST = sum(int(row["cost"]) for row in STAGE_POLICY.values())


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _text(value: Any, name: str, maximum: int = 500) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise ComputeError(f"{name} must contain 1 to {maximum} characters")
    return result


def _probability(value: Any, name: str, *, allow_zero_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    if allow_zero_one:
        if not 0.0 <= result <= 1.0:
            raise ComputeError(f"{name} must be between 0 and 1")
    elif not 0.0 < result < 1.0:
        raise ComputeError(f"{name} must be strictly between 0 and 1")
    return result


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _evidence_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = _sequence(inputs.get("evidence"), "inputs.evidence")
    if not 1 <= len(raw_rows) <= MAX_EVIDENCE:
        raise ComputeError(f"inputs.evidence must contain 1 to {MAX_EVIDENCE} rows")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, f"inputs.evidence[{index}]")
        evref = _text(row.get("evref"), f"inputs.evidence[{index}].evref", 160)
        if evref in seen:
            raise ComputeError(f"duplicate evidence evref: {evref}")
        seen.add(evref)
        stance = str(row.get("stance") or "neutral").strip().lower()
        if stance not in {"support", "contradict", "neutral"}:
            raise ComputeError("evidence stance must be support, contradict, or neutral")
        evidence_class = str(row.get("analysis_class") or "DIRECT").upper()
        if evidence_class not in ANALYSIS_CLASSES:
            raise ComputeError(f"invalid evidence analysis_class: {evidence_class}")
        reliability = _probability(
            row.get("reliability", 0.5),
            f"inputs.evidence[{index}].reliability",
        )
        normalized: dict[str, Any] = {
            "evref": evref,
            "stance": stance,
            "analysis_class": evidence_class,
            "reliability": reliability,
        }
        for field in (
            "entity",
            "institution",
            "geography",
            "activity",
            "case_id",
            "timestamp",
            "summary",
        ):
            if row.get(field) not in (None, ""):
                normalized[field] = str(row[field]).strip()[:2_000]
        aliases = row.get("aliases")
        if aliases is not None:
            normalized["aliases"] = [
                _text(item, f"inputs.evidence[{index}].aliases[]", 200)
                for item in _sequence(aliases, f"inputs.evidence[{index}].aliases")
            ][:50]
        tokens = row.get("tokens")
        if tokens is not None:
            normalized["tokens"] = [
                _text(item, f"inputs.evidence[{index}].tokens[]", 200)
                for item in _sequence(tokens, f"inputs.evidence[{index}].tokens")
            ][:500]
        p_true = row.get("p_if_true")
        p_false = row.get("p_if_false")
        if (p_true is None) != (p_false is None):
            raise ComputeError("p_if_true and p_if_false must be supplied together")
        if p_true is not None:
            normalized["p_if_true"] = _probability(
                p_true,
                f"inputs.evidence[{index}].p_if_true",
                allow_zero_one=False,
            )
            normalized["p_if_false"] = _probability(
                p_false,
                f"inputs.evidence[{index}].p_if_false",
                allow_zero_one=False,
            )
        rows.append(normalized)
    return rows


def _entity_inputs(
    inputs: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw = inputs.get("entity_records")
    if raw is None:
        generated = []
        for row in evidence:
            entity = str(row.get("entity") or "").strip()
            if entity:
                generated.append(
                    {
                        "name": entity,
                        "institution": str(row.get("institution") or "").strip(),
                        "geography": str(row.get("geography") or "").strip(),
                    }
                )
        raw_rows: Sequence[Any] = generated
    else:
        raw_rows = _sequence(raw, "inputs.entity_records")
    if len(raw_rows) > MAX_ENTITY_RECORDS:
        raise ComputeError(f"entity_records cannot exceed {MAX_ENTITY_RECORDS}")
    records = [dict(_mapping(row, "inputs.entity_records[]")) for row in raw_rows]
    fields_raw = inputs.get("entity_fields") or ["name", "institution", "geography"]
    fields = [str(item).strip() for item in _sequence(fields_raw, "inputs.entity_fields")]
    fields = [field for field in fields if field]
    if records and (not fields or len(fields) > 10):
        raise ComputeError("entity_fields must contain 1 to 10 names when entity records are present")
    return records, fields


def _relations(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _sequence(inputs.get("relations") or [], "inputs.relations")
    if len(rows) > MAX_RELATIONS:
        raise ComputeError(f"relations cannot exceed {MAX_RELATIONS}")
    result = []
    for index, item in enumerate(rows):
        row = _mapping(item, f"inputs.relations[{index}]")
        result.append(
            {
                "subject": _text(row.get("subject"), "relation.subject", 200),
                "predicate": _text(row.get("predicate"), "relation.predicate", 120),
                "object": _text(row.get("object"), "relation.object", 500),
                "object_is_entity": bool(row.get("object_is_entity", True)),
            }
        )
    return result


def _process_cases(
    inputs: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    explicit = inputs.get("process_cases")
    if explicit is not None:
        cases = [
            dict(_mapping(row, "inputs.process_cases[]"))
            for row in _sequence(explicit, "inputs.process_cases")
        ]
        if len(cases) > MAX_CASES:
            raise ComputeError(f"process_cases cannot exceed {MAX_CASES}")
        return cases
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in evidence:
        case_id = str(row.get("case_id") or "").strip()
        activity = str(row.get("activity") or "").strip()
        if not case_id or not activity:
            continue
        grouped.setdefault(case_id, []).append((str(row.get("timestamp") or ""), activity))
    return [
        {"case_id": case_id, "activities": [activity for _, activity in sorted(events)]}
        for case_id, events in sorted(grouped.items())
    ][:MAX_CASES]


def _rules(inputs: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw_rules = _sequence(inputs.get("rules") or [], "inputs.rules")
    if len(raw_rules) > MAX_RULES:
        raise ComputeError(f"rules cannot exceed {MAX_RULES}")
    by_ref = {str(row["evref"]): row for row in evidence}
    known = set(by_ref)
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        row = _mapping(raw, f"inputs.rules[{index}]")
        name = _text(row.get("name"), f"inputs.rules[{index}].name", 120)
        if name in seen:
            raise ComputeError(f"duplicate rule name: {name}")
        seen.add(name)
        refs = [
            _text(item, f"inputs.rules[{index}].required_evrefs[]", 160)
            for item in _sequence(row.get("required_evrefs"), f"inputs.rules[{index}].required_evrefs")
        ]
        if not refs or len(refs) > MAX_PROBLOG_FACTS or len(refs) != len(set(refs)):
            raise ComputeError("each rule requires 1 to 100 unique evidence references")
        unknown = sorted(set(refs) - known)
        if unknown:
            raise ComputeError(f"rule {name} references unknown evidence: {unknown[:5]}")
        non_supporting = [ref for ref in refs if by_ref[ref]["stance"] != "support"]
        if non_supporting:
            raise ComputeError(
                f"rule {name} may reference only supporting evidence: {non_supporting[:5]}"
            )
        result.append({"name": name, "required_evrefs": refs})
    return result


def _stage_signals(
    evidence: Sequence[Mapping[str, Any]],
    entity_records: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    process_cases: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, bool]:
    aliases = any(row.get("aliases") for row in evidence)
    token_sets = sum(bool(row.get("tokens")) for row in evidence) >= 2
    entity_capable = len(entity_records) >= 2
    graph_capable = len(relations) > 0
    ontology_capable = bool(inputs.get("ontology_classes")) or bool(inputs.get("shacl_data_turtle"))
    probabilistic = any("p_if_true" in row for row in evidence) or bool(inputs.get("rules"))
    contradiction = any(row["stance"] in {"support", "contradict"} for row in evidence)
    return {
        "name_normalization": aliases or entity_capable,
        "similarity_collision": token_sets,
        "entity_resolution": entity_capable,
        "knowledge_graph": graph_capable or ontology_capable,
        "graph_analysis": graph_capable,
        "process_mining": bool(process_cases),
        "probabilistic_inference": probabilistic,
        "contradiction_check": contradiction,
    }


def _dynamic_stage_budget(
    inputs: Mapping[str, Any],
    *,
    evidence_count: int,
    entity_count: int,
    relation_count: int,
    process_case_count: int,
    active_stage_count: int,
) -> tuple[int, str]:
    depth = str(inputs.get("analysis_depth") or "auto").strip().lower()
    presets = {"lean": 9, "balanced": 15, "deep": MAX_STAGE_COST}
    if depth in presets:
        return min(MAX_STAGE_COST, presets[depth]), f"explicit-{depth}"
    if depth != "auto":
        raise ComputeError("analysis_depth must be auto, lean, balanced, or deep")
    complexity = 0
    complexity += min(5, math.ceil(evidence_count / 100))
    complexity += min(4, math.ceil(entity_count / 125)) if entity_count else 0
    complexity += min(4, math.ceil(relation_count / 1_250)) if relation_count else 0
    complexity += min(3, math.ceil(process_case_count / 250)) if process_case_count else 0
    complexity += min(4, active_stage_count // 2)
    budget = max(9, min(MAX_STAGE_COST, 8 + complexity))
    return budget, "auto-complexity"


def _select_stages(signals: Mapping[str, bool], budget: int) -> dict[str, Any]:
    try:
        import networkx as nx
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ComputeError("indirect intelligence planner requires NetworkX and OR-Tools") from exc

    model = cp_model.CpModel()
    variables = {stage: model.new_bool_var(f"select_{stage}") for stage in STAGE_ORDER}
    for stage in STAGE_ORDER:
        if not signals.get(stage, False):
            model.add(variables[stage] == 0)
    if signals.get("contradiction_check", False):
        model.add(variables["contradiction_check"] == 1)
    if signals.get("probabilistic_inference", False):
        model.add(variables["probabilistic_inference"] == 1)
    model.add(sum(variables.values()) <= MAX_STAGES)
    model.add(
        sum(int(STAGE_POLICY[stage]["cost"]) * variables[stage] for stage in STAGE_ORDER)
        <= budget
    )
    model.maximize(
        sum(
            (10 * int(STAGE_POLICY[stage]["utility"]) - int(STAGE_POLICY[stage]["cost"]))
            * variables[stage]
            for stage in STAGE_ORDER
        )
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise ComputeError(
            "indirect intelligence stage planner must prove OPTIMAL; "
            f"observed={solver.status_name(status)}"
        )
    selected = [stage for stage in STAGE_ORDER if bool(solver.value(variables[stage]))]
    graph = nx.DiGraph()
    graph.add_nodes_from(selected)
    graph.add_edges_from(zip(selected, selected[1:], strict=False))
    if not nx.is_directed_acyclic_graph(graph):
        raise ComputeError("indirect intelligence stage graph must remain acyclic")
    order = list(nx.topological_sort(graph)) if selected else []
    if order != selected:
        raise ComputeError("indirect intelligence stage order is not deterministic")
    used_cost = sum(int(STAGE_POLICY[stage]["cost"]) for stage in selected)
    return {
        "selected_stages": selected,
        "signals": dict(signals),
        "solver_status": solver.status_name(status),
        "objective_value": int(round(solver.objective_value)),
        "cost_budget": budget,
        "cost_used": used_cost,
        "stage_policy": {stage: dict(STAGE_POLICY[stage]) for stage in selected},
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "serial_execution": True,
        "automatic_parallel_execution": False,
        "maximum_stages": MAX_STAGES,
    }


def _name_normalization(
    evidence: Sequence[Mapping[str, Any]],
    entity_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left: list[str] = []
    right: list[str] = []
    for row in evidence:
        entity = str(row.get("entity") or "").strip()
        if entity:
            left.append(entity)
        right.extend(str(item) for item in row.get("aliases") or [])
    for row in entity_records:
        name = str(row.get("name") or "").strip()
        if name:
            left.append(name)
    left = list(dict.fromkeys(left))[:1_000]
    right = list(dict.fromkeys(right or left))[:1_000]
    if not left:
        return {"matches": [], "skipped": "no entity names"}
    return rapidfuzz_record_collision({"left": left, "right": right, "threshold": 82.0})


def _similarity_collision(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in evidence if row.get("tokens")]
    candidates.sort(key=lambda row: (-float(row["reliability"]), str(row["evref"])))
    selected = candidates[:MAX_DATASKETCH_SETS]
    sets = {str(row["evref"]): list(row.get("tokens") or []) for row in selected}
    if len(sets) < 2:
        return {"pairwise_similarity": [], "skipped": "fewer than two token sets"}
    result = datasketch_set_similarity({"sets": sets, "num_perm": 128})
    result["input_set_count"] = len(candidates)
    result["analyzed_set_count"] = len(selected)
    result["governed_limit_applied"] = len(candidates) > len(selected)
    return result


def _entity_resolution(
    entity_records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    if len(entity_records) < 2:
        return {"matched_pairs": [], "skipped": "fewer than two entity records"}
    threshold = _probability(
        inputs.get("entity_match_threshold", 0.85),
        "inputs.entity_match_threshold",
    )
    result = splink_entity_resolution(
        {
            "records": [dict(row) for row in entity_records],
            "fields": list(fields),
            "threshold": threshold,
        }
    )
    result["implementation_note"] = (
        "The current governed Splink adapter verifies the pinned Splink capability pack "
        "but computes deterministic weighted RapidFuzz field scores. Its score is not "
        "a calibrated Splink match probability."
    )
    return result


def _knowledge_graph(
    relations: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if relations:
        result["rdf"] = rdflib_claim_evidence_graph(
            {"triples": [dict(row) for row in relations]}
        )
    classes = inputs.get("ontology_classes")
    if classes:
        result["ontology"] = owlready2_ontology_summary(
            {
                "classes": list(_sequence(classes, "inputs.ontology_classes")),
                "subclass_relations": list(
                    _sequence(
                        inputs.get("ontology_subclass_relations") or [],
                        "inputs.ontology_subclass_relations",
                    )
                ),
            }
        )
    data_turtle = inputs.get("shacl_data_turtle")
    shapes_turtle = inputs.get("shacl_shapes_turtle")
    if data_turtle is not None or shapes_turtle is not None:
        if data_turtle is None or shapes_turtle is None:
            raise ComputeError(
                "SHACL validation requires both shacl_data_turtle and shacl_shapes_turtle"
            )
        result["shacl"] = pyshacl_graph_validation(
            {"data_turtle": data_turtle, "shapes_turtle": shapes_turtle}
        )
    return result


def _graph_analysis(
    relations: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    import networkx as nx

    nodes = sorted(
        {str(row["subject"]) for row in relations}
        | {
            str(row["object"])
            for row in relations
            if row.get("object_is_entity", True)
        }
    )
    edges = [
        [str(row["subject"]), str(row["object"])]
        for row in relations
        if row.get("object_is_entity", True)
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((edge[0], edge[1]) for edge in edges)
    degree_ranking = sorted(
        (
            {"node": str(node), "degree_centrality": float(score)}
            for node, score in nx.degree_centrality(graph).items()
        ),
        key=lambda row: (-row["degree_centrality"], row["node"]),
    )[:1_000]
    if len(nodes) <= MAX_IGRAPH_NODES:
        igraph_result: dict[str, Any] = igraph_link_analysis(
            {"nodes": nodes, "edges": edges, "directed": True}
        ) if nodes else {"ranking": []}
    else:
        igraph_result = {
            "skipped": "node count exceeds governed igraph adapter limit",
            "node_count": len(nodes),
            "adapter_limit": MAX_IGRAPH_NODES,
        }
    requested = [
        str(item)
        for item in _sequence(inputs.get("path_targets") or [], "inputs.path_targets")
    ]
    paths = []
    for source in requested:
        if source not in graph:
            continue
        for target in requested:
            if source == target or target not in graph:
                continue
            try:
                path = nx.shortest_path(graph, source, target)
            except nx.NetworkXNoPath:
                continue
            paths.append({"source": source, "target": target, "path": path})
            if len(paths) >= 100:
                break
        if len(paths) >= 100:
            break
    return {
        "networkx_degree_ranking": degree_ranking,
        "igraph": igraph_result,
        "graph_paths": paths,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def _process_mining(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"directly_follows_edges": [], "skipped": "no process cases"}
    normalized = []
    total_events = 0
    for index, raw in enumerate(cases):
        case_id = _text(raw.get("case_id"), f"process_cases[{index}].case_id", 80)
        activities = [
            _text(item, f"process_cases[{index}].activities[]", 100)
            for item in _sequence(
                raw.get("activities"),
                f"process_cases[{index}].activities",
            )
        ]
        if not 1 <= len(activities) <= 200:
            raise ComputeError("each process case must contain 1 to 200 activities")
        total_events += len(activities)
        if total_events > 10_000:
            raise ComputeError("process mining input cannot exceed 10000 events")
        normalized.append({"case_id": case_id, "activities": activities})
    return pm4py_directly_follows({"cases": normalized})


def _bayesian_posterior(
    evidence: Sequence[Mapping[str, Any]],
    prior: float,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in evidence
        if "p_if_true" in row and row["stance"] != "neutral"
    ]
    if not rows:
        return None
    nodes = ["hypothesis"] + [f"e{index}" for index in range(len(rows))]
    edges = [["hypothesis", f"e{index}"] for index in range(len(rows))]
    cpds: list[dict[str, Any]] = [
        {
            "variable": "hypothesis",
            "variable_card": 2,
            "values": [[1.0 - prior], [prior]],
            "evidence": [],
            "evidence_card": [],
            "state_names": {"hypothesis": ["false", "true"]},
        }
    ]
    observed: dict[str, str] = {}
    for index, row in enumerate(rows):
        p_true = float(row["p_if_true"])
        p_false = float(row["p_if_false"])
        variable = f"e{index}"
        cpds.append(
            {
                "variable": variable,
                "variable_card": 2,
                "values": [
                    [1.0 - p_false, 1.0 - p_true],
                    [p_false, p_true],
                ],
                "evidence": ["hypothesis"],
                "evidence_card": [2],
                "state_names": {
                    variable: ["absent", "present"],
                    "hypothesis": ["false", "true"],
                },
            }
        )
        observed[variable] = "present"
    result = bayesian_network_inference(
        {
            "mode": "fixed_network_inference",
            "nodes": nodes,
            "edges": edges,
            "cpds": cpds,
            "query_variables": ["hypothesis"],
            "evidence": observed,
        }
    )
    values = result.get("query", {}).get("values")
    posterior = None
    if isinstance(values, list) and len(values) == 2:
        posterior = float(values[1])
    return {
        "prior_probability": prior,
        "posterior_probability": posterior,
        "evidence_count": len(rows),
        "pgmpy": result,
        "conditional_independence_assumption": True,
        "likelihood_semantics": (
            "p_if_true/p_if_false are P(observed evidence present | hypothesis true/false); "
            "stance is evaluated separately and is not algebraically inverted."
        ),
    }


def _problog_rule_results(
    evidence: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_ref = {str(row["evref"]): row for row in evidence}
    output = []
    for rule in rules:
        facts = []
        for index, evref in enumerate(rule["required_evrefs"]):
            row = by_ref[str(evref)]
            probability = max(0.000001, min(0.999999, float(row["reliability"])))
            facts.append({"name": f"evidence_{index}", "probability": probability})
        result = problog_evidence_probability({"facts": facts})
        output.append(
            {
                "rule": str(rule["name"]),
                "required_evrefs": list(rule["required_evrefs"]),
                "joint_probability": result["joint_probability"],
                "independence_assumption": True,
            }
        )
    return output


def _probabilistic_inference(
    evidence: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    prior = _probability(
        inputs.get("prior_probability", 0.5),
        "inputs.prior_probability",
        allow_zero_one=False,
    )
    bayesian = _bayesian_posterior(evidence, prior)
    explicit_rule_results = _problog_rule_results(evidence, rules) if rules else []
    support_rows = [row for row in evidence if row["stance"] == "support"]
    support_rows.sort(key=lambda row: (-float(row["reliability"]), str(row["evref"])))
    selected_support = support_rows[:MAX_PROBLOG_FACTS]
    default_joint = None
    if selected_support:
        facts = [
            {
                "name": f"support_{index}",
                "probability": max(
                    0.000001,
                    min(0.999999, float(row["reliability"])),
                ),
            }
            for index, row in enumerate(selected_support)
        ]
        default_joint = problog_evidence_probability({"facts": facts})
    return {
        "bayesian": bayesian,
        "problog_rules": explicit_rule_results,
        "problog_support_joint": default_joint,
        "support_fact_count": len(support_rows),
        "support_facts_analyzed": len(selected_support),
        "support_fact_limit_applied": len(support_rows) > len(selected_support),
        "probabilistic_claim": True,
        "fact_promotion_allowed": False,
    }


def _contradiction(
    evidence: Sequence[Mapping[str, Any]],
    hypothesis: str,
) -> dict[str, Any]:
    rows = [
        {
            "claim": hypothesis,
            "stance": row["stance"],
            "weight": float(row["reliability"]),
        }
        for row in evidence
    ]
    return claim_evidence_contradiction({"claims": [hypothesis], "evidence": rows})


def _final_class_and_confidence(
    evidence: Sequence[Mapping[str, Any]],
    posterior: float | None,
    has_links: bool,
    probabilistic_inference_used: bool,
) -> tuple[str, float]:
    support = sum(
        float(row["reliability"]) for row in evidence if row["stance"] == "support"
    )
    contradict = sum(
        float(row["reliability"])
        for row in evidence
        if row["stance"] == "contradict"
    )
    direct_support = any(
        row["analysis_class"] == "DIRECT" and row["stance"] == "support"
        for row in evidence
    )
    total = support + contradict
    evidence_balance = support / total if total > 0 else 0.5
    if contradict > support and contradict >= 0.75:
        analysis_class = "CONTRADICTED"
    elif posterior is not None or probabilistic_inference_used:
        analysis_class = "INFERRED"
    elif direct_support:
        analysis_class = "DIRECT"
    elif has_links:
        analysis_class = "LINKED"
    else:
        analysis_class = "INFERRED"
    base = posterior if posterior is not None else evidence_balance
    contradiction_penalty = 1.0 - min(
        0.6,
        contradict / max(total, 1e-12) * 0.6,
    )
    confidence = max(0.0, min(1.0, float(base) * contradiction_penalty))
    if analysis_class == "CONTRADICTED":
        confidence = max(
            confidence,
            min(1.0, contradict / max(total, 1e-12)),
        )
    return analysis_class, confidence


def indirect_intelligence_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode != MODE:
        raise ComputeError(f"inputs.mode must equal {MODE}")
    hypothesis = _text(inputs.get("hypothesis"), "inputs.hypothesis", 2_000)
    evidence = _evidence_rows(inputs)
    entity_records, entity_fields = _entity_inputs(inputs, evidence)
    relations = _relations(inputs)
    process_cases = _process_cases(inputs, evidence)
    rules = _rules(inputs, evidence)
    signals = _stage_signals(evidence, entity_records, relations, process_cases, inputs)
    budget, budget_reason = _dynamic_stage_budget(
        inputs,
        evidence_count=len(evidence),
        entity_count=len(entity_records),
        relation_count=len(relations),
        process_case_count=len(process_cases),
        active_stage_count=sum(bool(value) for value in signals.values()),
    )
    plan = _select_stages(signals, budget)
    plan["budget_reason"] = budget_reason

    stage_results: dict[str, Any] = {}
    methods: list[str] = []
    graph_paths: list[dict[str, Any]] = []
    entity_links: list[dict[str, Any]] = []
    posterior: float | None = None

    for stage in plan["selected_stages"]:
        if stage == "name_normalization":
            stage_results[stage] = _name_normalization(evidence, entity_records)
            methods.append("RapidFuzz")
        elif stage == "similarity_collision":
            stage_results[stage] = _similarity_collision(evidence)
            methods.append("datasketch-MinHash")
        elif stage == "entity_resolution":
            stage_results[stage] = _entity_resolution(entity_records, entity_fields, inputs)
            entity_links = list(stage_results[stage].get("matched_pairs") or [])[:1_000]
            methods.append("Splink-capability-adapter+RapidFuzz")
        elif stage == "knowledge_graph":
            stage_results[stage] = _knowledge_graph(relations, inputs)
            methods.append("RDFLib/Owlready2/pySHACL")
        elif stage == "graph_analysis":
            stage_results[stage] = _graph_analysis(relations, inputs)
            graph_paths = list(stage_results[stage].get("graph_paths") or [])[:100]
            methods.append("NetworkX+igraph")
        elif stage == "process_mining":
            stage_results[stage] = _process_mining(process_cases)
            methods.append("PM4Py")
        elif stage == "probabilistic_inference":
            stage_results[stage] = _probabilistic_inference(evidence, rules, inputs)
            bayesian = stage_results[stage].get("bayesian")
            if isinstance(bayesian, Mapping) and bayesian.get("posterior_probability") is not None:
                posterior = float(bayesian["posterior_probability"])
            methods.extend(["pgmpy", "ProbLog"])
        elif stage == "contradiction_check":
            stage_results[stage] = _contradiction(evidence, hypothesis)
            methods.append("contradiction-matrix")
        else:  # pragma: no cover
            raise ComputeError(f"unknown indirect intelligence stage: {stage}")

    analysis_class, confidence = _final_class_and_confidence(
        evidence,
        posterior,
        bool(entity_links or graph_paths or relations),
        "probabilistic_inference" in plan["selected_stages"],
    )
    supporting = [
        str(row["evref"]) for row in evidence if row["stance"] == "support"
    ]
    contradicting = [
        str(row["evref"]) for row in evidence if row["stance"] == "contradict"
    ]
    prior = _probability(
        inputs.get("prior_probability", 0.5),
        "inputs.prior_probability",
        allow_zero_one=False,
    )
    scope = _mapping(inputs.get("scope") or {}, "inputs.scope")
    inference_material = {
        "hypothesis": hypothesis,
        "supporting_evrefs": supporting,
        "contradicting_evrefs": contradicting,
        "methods": methods,
        "prior": prior,
        "posterior": posterior,
        "scope": dict(scope),
    }
    inference_id = "inf-" + _canonical_sha(inference_material)[:20]
    class_counts = {
        label: sum(row["analysis_class"] == label for row in evidence)
        for label in sorted(ANALYSIS_CLASSES)
    }

    return {
        "mode": MODE,
        "inference_id": inference_id,
        "hypothesis": hypothesis,
        "analysis_class": analysis_class,
        "supporting_evrefs": supporting,
        "contradicting_evrefs": contradicting,
        "entity_links": entity_links,
        "graph_paths": graph_paths,
        "inference_methods": list(dict.fromkeys(methods)),
        "prior_probability": prior,
        "posterior_probability": posterior,
        "confidence": confidence,
        "assumptions": [
            str(item)[:1_000]
            for item in _sequence(inputs.get("assumptions") or [], "inputs.assumptions")
        ][:100],
        "time_window": scope.get("time_window"),
        "geographic_scope": scope.get("geographic_scope"),
        "institution_scope": scope.get("institution_scope"),
        "scope_extrapolation_allowed": False,
        "stage_plan": plan,
        "stage_results": stage_results,
        "evidence_count": len(evidence),
        "evidence_class_counts": class_counts,
        "network_used": False,
        "external_data_fetches": 0,
        "model_calls": 0,
        "automatic_parallel_execution": False,
        "ticket_supplied_code_allowed": False,
        "ticket_supplied_logic_program_allowed": False,
        "decision_support_only": True,
        "expert_semantic_synthesis_required_for_publication": True,
        "governance_release_gate_required": True,
        "inference_not_fact": analysis_class in {"LINKED", "INFERRED", "CONTRADICTED"},
        "publication_boundary": (
            "Linked, probabilistic, contradicted, or model-derived outputs must remain "
            "labelled as inference. Geographic or institution-wide extrapolation requires "
            "separate coverage evidence and governance approval."
        ),
    }


HANDLERS = {MODE: indirect_intelligence_analysis}
