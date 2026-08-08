#!/usr/bin/env python3
"""Governed indirect-intelligence fusion for already-relayed structured evidence.

The mode is deliberately collection-free. It never reaches the evidence center,
opens URLs, calls models, or accepts ticket-supplied code. OR-Tools chooses among
a repository-controlled set of applicable analysis stages and NetworkX validates
the resulting serial DAG. Every conclusion remains explicitly typed as DIRECT,
LINKED, INFERRED, or CONTRADICTED; probabilistic output is never promoted to fact.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from compute_runner import ComputeError
from bayesian_network_operations import bayesian_network_inference
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
        reliability = _probability(row.get("reliability", 0.5), f"inputs.evidence[{index}].reliability")
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
        if row.get("p_if_true") is not None or row.get("p_if_false") is not None:
            normalized["p_if_true"] = _probability(
                row.get("p_if_true"),
                f"inputs.evidence[{index}].p_if_true",
                allow_zero_one=False,
            )
            normalized["p_if_false"] = _probability(
                row.get("p_if_false"),
                f"inputs.evidence[{index}].p_if_false",
                allow_zero_one=False,
            )
        rows.append(normalized)
    return rows


def _entity_inputs(inputs: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    raw = inputs.get("entity_records")
    if raw is None:
        generated = []
        for row in evidence:
            entity = str(row.get("entity") or "").strip()
            if entity:
                generated.append({
                    "name": entity,
                    "institution": str(row.get("institution") or "").strip(),
                    "geography": str(row.get("geography") or "").strip(),
                })
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
    raw = inputs.get("relations") or []
    rows = _sequence(raw, "inputs.relations")
    if len(rows) > MAX_RELATIONS:
        raise ComputeError(f"relations cannot exceed {MAX_RELATIONS}")
    result = []
    for index, item in enumerate(rows):
        row = _mapping(item, f"inputs.relations[{index}]")
        result.append({
            "subject": _text(row.get("subject"), "relation.subject", 200),
            "predicate": _text(row.get("predicate"), "relation.predicate", 120),
            "object": _text(row.get("object"), "relation.object", 500),
            "object_is_entity": bool(row.get("object_is_entity", True)),
        })
    return result


def _process_cases(inputs: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    explicit = inputs.get("process_cases")
    if explicit is not None:
        cases = [dict(_mapping(row, "inputs.process_cases[]")) for row in _sequence(explicit, "inputs.process_cases")]
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
    probabilistic = any("p_if_true" in row and "p_if_false" in row for row in evidence)
    contradiction = any(row["stance"] == "contradict" for row in evidence) or any(
        row["stance"] == "support" for row in evidence
    )
    return {
        "name_normalization": aliases or entity_capable,
        "similarity_collision": token_sets,
        "entity_resolution": entity_capable,
        "knowledge_graph": graph_capable or ontology_capable,
        "graph_analysis": graph_capable,
        "process_mining": len(process_cases) > 0,
        "probabilistic_inference": probabilistic,
        "contradiction_check": contradiction,
    }


def _select_stages(signals: Mapping[str, bool]) -> dict[str, Any]:
    try:
        import networkx as nx
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ComputeError("indirect intelligence planner requires NetworkX and OR-Tools") from exc

    model = cp_model.CpModel()
    variables = {stage: model.new_bool_var(f"select_{stage}") for stage in STAGE_ORDER}
    utilities = {
        "name_normalization": 25,
        "similarity_collision": 18,
        "entity_resolution": 30,
        "knowledge_graph": 24,
        "graph_analysis": 24,
        "process_mining": 28,
        "probabilistic_inference": 40,
        "contradiction_check": 45,
    }
    for stage in STAGE_ORDER:
        if not signals.get(stage, False):
            model.add(variables[stage] == 0)
    model.add(sum(variables.values()) <= MAX_STAGES)
    model.maximize(sum(utilities[stage] * variables[stage] for stage in STAGE_ORDER))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise ComputeError(f"indirect intelligence stage planner must prove OPTIMAL; observed={solver.status_name(status)}")
    selected = [stage for stage in STAGE_ORDER if bool(solver.value(variables[stage]))]
    graph = nx.DiGraph()
    graph.add_nodes_from(selected)
    graph.add_edges_from(zip(selected, selected[1:], strict=False))
    if not nx.is_directed_acyclic_graph(graph):
        raise ComputeError("indirect intelligence stage graph must remain acyclic")
    order = list(nx.topological_sort(graph)) if selected else []
    if order != selected:
        raise ComputeError("indirect intelligence stage order is not deterministic")
    return {
        "selected_stages": selected,
        "signals": dict(signals),
        "solver_status": solver.status_name(status),
        "objective_value": int(round(solver.objective_value)),
        "selection_engine": "ortools-cp-sat",
        "graph_engine": "networkx",
        "serial_execution": True,
        "automatic_parallel_execution": False,
        "maximum_stages": MAX_STAGES,
    }


def _name_normalization(evidence: Sequence[Mapping[str, Any]], entity_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = []
    right = []
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
    sets = {str(row["evref"]): list(row.get("tokens") or []) for row in evidence if row.get("tokens")}
    if len(sets) < 2:
        return {"pairwise_similarity": [], "skipped": "fewer than two token sets"}
    return datasketch_set_similarity({"sets": sets, "num_perm": 128})


def _entity_resolution(entity_records: Sequence[Mapping[str, Any]], fields: Sequence[str], inputs: Mapping[str, Any]) -> dict[str, Any]:
    if len(entity_records) < 2:
        return {"matched_pairs": [], "skipped": "fewer than two entity records"}
    threshold = _probability(inputs.get("entity_match_threshold", 0.85), "inputs.entity_match_threshold")
    result = splink_entity_resolution({
        "records": [dict(row) for row in entity_records],
        "fields": list(fields),
        "threshold": threshold,
    })
    result["implementation_note"] = (
        "Current governed Splink adapter uses deterministic weighted RapidFuzz field scoring while verifying the pinned Splink capability pack; "
        "do not interpret its score as a calibrated Splink match probability."
    )
    return result


def _knowledge_graph(relations: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if relations:
        result["rdf"] = rdflib_claim_evidence_graph({"triples": [dict(row) for row in relations]})
    classes = inputs.get("ontology_classes")
    if classes:
        result["ontology"] = owlready2_ontology_summary({
            "classes": list(_sequence(classes, "inputs.ontology_classes")),
            "subclass_relations": list(_sequence(inputs.get("ontology_subclass_relations") or [], "inputs.ontology_subclass_relations")),
        })
    data_turtle = inputs.get("shacl_data_turtle")
    shapes_turtle = inputs.get("shacl_shapes_turtle")
    if data_turtle is not None or shapes_turtle is not None:
        if data_turtle is None or shapes_turtle is None:
            raise ComputeError("SHACL validation requires both shacl_data_turtle and shacl_shapes_turtle")
        result["shacl"] = pyshacl_graph_validation({
            "data_turtle": data_turtle,
            "shapes_turtle": shapes_turtle,
        })
    return result


def _graph_analysis(relations: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any]) -> dict[str, Any]:
    import networkx as nx

    nodes = sorted({str(row["subject"]) for row in relations} | {str(row["object"]) for row in relations if row.get("object_is_entity", True)})
    edges = [[str(row["subject"]), str(row["object"])] for row in relations if row.get("object_is_entity", True)]
    igraph_result = igraph_link_analysis({"nodes": nodes, "edges": edges, "directed": True}) if nodes else {"ranking": []}
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((edge[0], edge[1]) for edge in edges)
    requested = [str(item) for item in _sequence(inputs.get("path_targets") or [], "inputs.path_targets")]
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
    return {"igraph": igraph_result, "graph_paths": paths, "node_count": len(nodes), "edge_count": len(edges)}


def _process_mining(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"directly_follows_edges": [], "skipped": "no process cases"}
    normalized = []
    for index, raw in enumerate(cases):
        case_id = _text(raw.get("case_id"), f"process_cases[{index}].case_id", 80)
        activities = [
            _text(item, f"process_cases[{index}].activities[]", 100)
            for item in _sequence(raw.get("activities"), f"process_cases[{index}].activities")
        ]
        if not activities:
            raise ComputeError("process case activities cannot be empty")
        normalized.append({"case_id": case_id, "activities": activities})
    return pm4py_directly_follows({"cases": normalized})


def _bayesian_posterior(evidence: Sequence[Mapping[str, Any]], prior: float) -> dict[str, Any] | None:
    rows = [row for row in evidence if "p_if_true" in row and "p_if_false" in row and row["stance"] != "neutral"]
    if not rows:
        return None
    nodes = ["hypothesis"] + [f"e{index}" for index in range(len(rows))]
    edges = [["hypothesis", f"e{index}"] for index in range(len(rows))]
    cpds: list[dict[str, Any]] = [{
        "variable": "hypothesis",
        "variable_card": 2,
        "values": [[1.0 - prior], [prior]],
        "evidence": [],
        "evidence_card": [],
        "state_names": {"hypothesis": ["false", "true"]},
    }]
    observed: dict[str, int] = {}
    for index, row in enumerate(rows):
        p_true = float(row["p_if_true"])
        p_false = float(row["p_if_false"])
        if row["stance"] == "contradict":
            p_true, p_false = 1.0 - p_true, 1.0 - p_false
        variable = f"e{index}"
        cpds.append({
            "variable": variable,
            "variable_card": 2,
            "values": [
                [1.0 - p_false, 1.0 - p_true],
                [p_false, p_true],
            ],
            "evidence": ["hypothesis"],
            "evidence_card": [2],
            "state_names": {variable: ["absent", "present"], "hypothesis": ["false", "true"]},
        })
        observed[variable] = 1
    result = bayesian_network_inference({
        "mode": "fixed_network_inference",
        "nodes": nodes,
        "edges": edges,
        "cpds": cpds,
        "query_variables": ["hypothesis"],
        "evidence": observed,
    })
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
    }


def _probabilistic_inference(evidence: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any]) -> dict[str, Any]:
    prior = _probability(inputs.get("prior_probability", 0.5), "inputs.prior_probability", allow_zero_one=False)
    bayesian = _bayesian_posterior(evidence, prior)
    support_facts = [
        {"name": f"support_{index}", "probability": max(0.000001, min(0.999999, float(row["reliability"]))) }
        for index, row in enumerate(evidence)
        if row["stance"] == "support"
    ]
    rules = problog_evidence_probability({"facts": support_facts}) if support_facts else None
    return {
        "bayesian": bayesian,
        "problog": rules,
        "probabilistic_claim": True,
        "fact_promotion_allowed": False,
    }


def _contradiction(evidence: Sequence[Mapping[str, Any]], hypothesis: str) -> dict[str, Any]:
    rows = [
        {"claim": hypothesis, "stance": row["stance"], "weight": float(row["reliability"]), "evref": row["evref"]}
        for row in evidence
    ]
    return claim_evidence_contradiction({"claims": [hypothesis], "evidence": rows})


def _final_class_and_confidence(
    evidence: Sequence[Mapping[str, Any]],
    posterior: float | None,
    has_links: bool,
) -> tuple[str, float]:
    support = sum(float(row["reliability"]) for row in evidence if row["stance"] == "support")
    contradict = sum(float(row["reliability"]) for row in evidence if row["stance"] == "contradict")
    direct_support = any(row["analysis_class"] == "DIRECT" and row["stance"] == "support" for row in evidence)
    total = support + contradict
    evidence_balance = support / total if total > 0 else 0.5
    if contradict > support and contradict >= 0.75:
        analysis_class = "CONTRADICTED"
    elif direct_support and posterior is None:
        analysis_class = "DIRECT"
    elif posterior is not None:
        analysis_class = "INFERRED"
    elif has_links:
        analysis_class = "LINKED"
    else:
        analysis_class = "INFERRED"
    base = posterior if posterior is not None else evidence_balance
    contradiction_penalty = 1.0 - min(0.6, contradict / max(total, 1e-12) * 0.6)
    confidence = max(0.0, min(1.0, float(base) * contradiction_penalty))
    if analysis_class == "CONTRADICTED":
        confidence = max(confidence, min(1.0, contradict / max(total, 1e-12)))
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
    signals = _stage_signals(evidence, entity_records, relations, process_cases, inputs)
    plan = _select_stages(signals)

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
            methods.append("Splink-adapter+RapidFuzz")
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
            stage_results[stage] = _probabilistic_inference(evidence, inputs)
            bayesian = stage_results[stage].get("bayesian")
            if isinstance(bayesian, Mapping) and bayesian.get("posterior_probability") is not None:
                posterior = float(bayesian["posterior_probability"])
            methods.extend(["pgmpy", "ProbLog"])
        elif stage == "contradiction_check":
            stage_results[stage] = _contradiction(evidence, hypothesis)
            methods.append("contradiction-matrix")
        else:  # pragma: no cover - repository-controlled stage list
            raise ComputeError(f"unknown indirect intelligence stage: {stage}")

    analysis_class, confidence = _final_class_and_confidence(
        evidence,
        posterior,
        bool(entity_links or graph_paths or relations),
    )
    supporting = [str(row["evref"]) for row in evidence if row["stance"] == "support"]
    contradicting = [str(row["evref"]) for row in evidence if row["stance"] == "contradict"]
    prior = _probability(inputs.get("prior_probability", 0.5), "inputs.prior_probability", allow_zero_one=False)
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
        "assumptions": [str(item)[:1_000] for item in _sequence(inputs.get("assumptions") or [], "inputs.assumptions")][:100],
        "time_window": scope.get("time_window"),
        "geographic_scope": scope.get("geographic_scope"),
        "institution_scope": scope.get("institution_scope"),
        "scope_extrapolation_allowed": False,
        "stage_plan": plan,
        "stage_results": stage_results,
        "evidence_count": len(evidence),
        "network_used": False,
        "external_data_fetches": 0,
        "model_calls": 0,
        "automatic_parallel_execution": False,
        "ticket_supplied_code_allowed": False,
        "decision_support_only": True,
        "inference_not_fact": analysis_class in {"LINKED", "INFERRED", "CONTRADICTED"},
        "publication_boundary": (
            "Probabilistic, linked, or contradicted outputs must remain labelled as inference; "
            "nationwide or institution-wide extrapolation requires separate evidence coverage."
        ),
    }


HANDLERS = {MODE: indirect_intelligence_analysis}
