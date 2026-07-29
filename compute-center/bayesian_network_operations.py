#!/usr/bin/env python3
"""Bounded discrete Bayesian-network inference using the pinned pgmpy method pack.

Only fixed DAGs, bounded categorical states and structured CPDs/data are accepted. Learned
associations are never automatically labeled as a causal structure.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np

from compute_runner import ComputeError

EXPECTED_PGMPY = "1.1.2"
MAX_NODES = 50
MAX_EDGES = 200
MAX_STATES = 20
MAX_ROWS = 50_000
MAX_CPDS = 50
MODES = {
    "fixed_network_inference",
    "expert_prior_update",
    "bayesian_parameter_estimation",
    "em_parameter_estimation",
    "evidence_sensitivity",
    "virtual_evidence_update",
}


def _dependencies():
    try:
        import pandas as pd
        from pgmpy.estimators import BayesianEstimator, ExpectationMaximization, MaximumLikelihoodEstimator
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        from pgmpy.models import DiscreteBayesianNetwork
    except ImportError as exc:
        raise ComputeError("Bayesian-network engine is not installed; install requirements-bayesian-network.txt") from exc
    if version("pgmpy") != EXPECTED_PGMPY:
        raise ComputeError(f"pgmpy version must be exactly {EXPECTED_PGMPY}")
    return pd, BayesianEstimator, ExpectationMaximization, MaximumLikelihoodEstimator, TabularCPD, VariableElimination, DiscreteBayesianNetwork


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _edges(value: Any) -> list[tuple[str, str]]:
    raw = _sequence(value or [], "inputs.edges")
    if len(raw) > MAX_EDGES:
        raise ComputeError(f"inputs.edges cannot exceed {MAX_EDGES}")
    rows: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        edge = _sequence(item, f"inputs.edges[{index}]")
        if len(edge) != 2:
            raise ComputeError("each Bayesian-network edge must contain source and target")
        source, target = str(edge[0]).strip(), str(edge[1]).strip()
        if not source or not target or source == target:
            raise ComputeError("Bayesian-network edges require distinct non-empty nodes")
        rows.append((source, target))
    return rows


def _cpd_from_spec(TabularCPD, raw: Mapping[str, Any]):
    variable = str(raw.get("variable") or "").strip()
    card = _integer(raw.get("variable_card"), f"cpd[{variable}].variable_card", 2, MAX_STATES)
    values = _sequence(raw.get("values"), f"cpd[{variable}].values")
    evidence = [str(item) for item in _sequence(raw.get("evidence", []), f"cpd[{variable}].evidence")]
    evidence_card = [_integer(item, f"cpd[{variable}].evidence_card", 2, MAX_STATES) for item in _sequence(raw.get("evidence_card", []), f"cpd[{variable}].evidence_card")]
    if len(evidence) != len(evidence_card):
        raise ComputeError(f"CPD evidence and evidence_card mismatch for {variable}")
    state_names_raw = raw.get("state_names")
    state_names = dict(state_names_raw) if isinstance(state_names_raw, Mapping) else {}
    try:
        return TabularCPD(variable=variable, variable_card=card, values=values, evidence=evidence or None, evidence_card=evidence_card or None, state_names=state_names)
    except Exception as exc:
        raise ComputeError(f"invalid CPD for {variable}: {type(exc).__name__}: {exc}") from exc


def _build_model(inputs: Mapping[str, Any], *, require_cpds: bool = True):
    _, _, _, _, TabularCPD, VariableElimination, DiscreteBayesianNetwork = _dependencies()
    edges = _edges(inputs.get("edges", []))
    node_names = [str(item).strip() for item in _sequence(inputs.get("nodes", []), "inputs.nodes")]
    latent_names = {str(item).strip() for item in _sequence(inputs.get("latent_nodes", []), "inputs.latent_nodes")}
    all_nodes = set(node_names) | {item for edge in edges for item in edge} | latent_names
    if not 1 <= len(all_nodes) <= MAX_NODES or any(not item for item in all_nodes):
        raise ComputeError(f"Bayesian network must contain 1 to {MAX_NODES} non-empty nodes")
    try:
        model = DiscreteBayesianNetwork(edges, latents=latent_names)
        model.add_nodes_from(sorted(all_nodes))
    except Exception as exc:
        raise ComputeError(f"invalid Bayesian-network DAG: {type(exc).__name__}: {exc}") from exc
    cpd_specs = _sequence(inputs.get("cpds", []), "inputs.cpds")
    if len(cpd_specs) > MAX_CPDS:
        raise ComputeError(f"inputs.cpds cannot exceed {MAX_CPDS}")
    cpds = [_cpd_from_spec(TabularCPD, _mapping(raw, "inputs.cpds[]")) for raw in cpd_specs]
    if cpds:
        model.add_cpds(*cpds)
    if require_cpds:
        try:
            if not model.check_model():
                raise ComputeError("Bayesian-network CPDs are incomplete or inconsistent")
        except Exception as exc:
            if isinstance(exc, ComputeError):
                raise
            raise ComputeError(f"Bayesian-network validation failed: {type(exc).__name__}: {exc}") from exc
    return model, VariableElimination, TabularCPD


def _factor_payload(factor: Any) -> dict[str, Any]:
    return {"variables": [str(item) for item in factor.variables], "cardinality": [int(item) for item in np.asarray(factor.cardinality).reshape(-1)], "values": np.asarray(factor.values, dtype=float).tolist(), "state_names": {str(key): list(value) for key, value in getattr(factor, "state_names", {}).items()}}


def _cpd_payload(cpd: Any) -> dict[str, Any]:
    return {"variable": str(cpd.variable), "variable_card": int(cpd.variable_card), "evidence": [str(item) for item in cpd.variables[1:]], "cardinality": [int(item) for item in np.asarray(cpd.cardinality).reshape(-1)], "values": np.asarray(cpd.get_values(), dtype=float).tolist(), "state_names": {str(key): list(value) for key, value in getattr(cpd, "state_names", {}).items()}}


def _query(model, VariableElimination, inputs: Mapping[str, Any], *, virtual_evidence: list[Any] | None = None) -> dict[str, Any]:
    variables = [str(item) for item in _sequence(inputs.get("query_variables"), "inputs.query_variables")]
    if not variables or any(item not in model.nodes() for item in variables):
        raise ComputeError("query_variables must contain registered network nodes")
    evidence_raw = inputs.get("evidence")
    evidence = {str(key): value for key, value in evidence_raw.items()} if isinstance(evidence_raw, Mapping) else {}
    if any(key not in model.nodes() for key in evidence):
        raise ComputeError("evidence contains an unknown node")
    try:
        result = VariableElimination(model).query(variables=variables, evidence=evidence or None, virtual_evidence=virtual_evidence, show_progress=False)
    except Exception as exc:
        raise ComputeError(f"Bayesian-network inference failed: {type(exc).__name__}: {exc}") from exc
    return _factor_payload(result)


def _data_frame(inputs: Mapping[str, Any], *, allow_missing: bool = False):
    pd, *_ = _dependencies()
    raw = _mapping(inputs.get("data"), "inputs.data")
    if not 1 <= len(raw) <= MAX_NODES:
        raise ComputeError(f"inputs.data must contain 1 to {MAX_NODES} variables")
    lengths = set()
    data: dict[str, list[Any]] = {}
    for name, values_raw in raw.items():
        values = list(_sequence(values_raw, f"inputs.data.{name}"))
        lengths.add(len(values))
        converted = []
        for index, value in enumerate(values):
            if value is None and allow_missing:
                converted.append(np.nan)
            elif isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ComputeError(f"inputs.data.{name}[{index}] must be a categorical scalar")
            else:
                converted.append(value)
        data[str(name)] = converted
    if len(lengths) != 1:
        raise ComputeError("all data columns must have equal length")
    size = next(iter(lengths))
    if not 2 <= size <= MAX_ROWS:
        raise ComputeError(f"Bayesian-network data must contain 2 to {MAX_ROWS} rows")
    return pd.DataFrame(data)


def _fixed_inference(inputs: Mapping[str, Any]) -> dict[str, Any]:
    model, VariableElimination, _ = _build_model(inputs)
    return {"query": _query(model, VariableElimination, inputs), "model_valid": True, "node_count": model.number_of_nodes(), "edge_count": model.number_of_edges()}


def _expert_prior(inputs: Mapping[str, Any]) -> dict[str, Any]:
    categories = [str(item) for item in _sequence(inputs.get("states"), "inputs.states")]
    if not 2 <= len(categories) <= MAX_STATES or len(set(categories)) != len(categories):
        raise ComputeError("states must contain 2 to 20 unique categories")
    prior = np.asarray([_finite(item, "inputs.prior_counts[]") for item in _sequence(inputs.get("prior_counts"), "inputs.prior_counts")], dtype=float)
    observed = np.asarray([_finite(item, "inputs.observed_counts[]") for item in _sequence(inputs.get("observed_counts"), "inputs.observed_counts")], dtype=float)
    if prior.shape != (len(categories),) or observed.shape != prior.shape or np.any(prior <= 0) or np.any(observed < 0):
        raise ComputeError("prior_counts must be positive and observed_counts non-negative with one value per state")
    posterior = prior + observed
    probability = posterior / np.sum(posterior)
    return {"states": categories, "prior_counts": prior.tolist(), "observed_counts": observed.tolist(), "posterior_counts": posterior.tolist(), "posterior_probability": probability.tolist(), "expert_opinion_preserved_as_prior": True}


def _parameter_estimation(inputs: Mapping[str, Any], *, bayesian: bool) -> dict[str, Any]:
    pd, BayesianEstimator, _, MaximumLikelihoodEstimator, _, _, DiscreteBayesianNetwork = _dependencies()
    data = _data_frame(inputs)
    edges = _edges(inputs.get("edges", []))
    nodes = set(data.columns) | {item for edge in edges for item in edge}
    if len(nodes) > MAX_NODES:
        raise ComputeError("parameter-estimation network exceeds node limit")
    try:
        model = DiscreteBayesianNetwork(edges)
        model.add_nodes_from(sorted(nodes))
        if bayesian:
            estimator = BayesianEstimator(model, data)
            cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=_finite(inputs.get("equivalent_sample_size", 5.0), "inputs.equivalent_sample_size"), n_jobs=1)
        else:
            estimator = MaximumLikelihoodEstimator(model, data)
            cpds = estimator.get_parameters(n_jobs=1)
        model.add_cpds(*cpds)
        valid = bool(model.check_model())
    except Exception as exc:
        raise ComputeError(f"Bayesian-network parameter estimation failed: {type(exc).__name__}: {exc}") from exc
    return {"estimator": "bayesian_bdeu" if bayesian else "maximum_likelihood", "model_valid": valid, "cpds": [_cpd_payload(cpd) for cpd in sorted(cpds, key=lambda item: str(item.variable))], "observation_count": int(data.shape[0])}


def _em_estimation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _, _, ExpectationMaximization, _, _, _, DiscreteBayesianNetwork = _dependencies()
    data = _data_frame(inputs, allow_missing=True)
    edges = _edges(inputs.get("edges", []))
    latent_cards_raw = _mapping(inputs.get("latent_cards"), "inputs.latent_cards")
    latent_cards = {str(name): _integer(card, f"inputs.latent_cards.{name}", 2, MAX_STATES) for name, card in latent_cards_raw.items()}
    if not latent_cards:
        raise ComputeError("EM estimation requires latent_cards")
    nodes = set(data.columns) | set(latent_cards) | {item for edge in edges for item in edge}
    try:
        model = DiscreteBayesianNetwork(edges, latents=set(latent_cards))
        model.add_nodes_from(sorted(nodes))
        estimator = ExpectationMaximization(model, data)
        cpds = estimator.get_parameters(latent_card=latent_cards, max_iter=_integer(inputs.get("max_iterations", 30), "inputs.max_iterations", 1, 200), atol=_finite(inputs.get("tolerance", 1e-6), "inputs.tolerance"), n_jobs=1, seed=_integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1), init_cpds="uniform", show_progress=False)
        model.add_cpds(*cpds)
        valid = bool(model.check_model())
    except Exception as exc:
        raise ComputeError(f"Bayesian-network EM estimation failed: {type(exc).__name__}: {exc}") from exc
    return {"estimator": "expectation_maximization", "latent_cards": latent_cards, "model_valid": valid, "cpds": [_cpd_payload(cpd) for cpd in sorted(cpds, key=lambda item: str(item.variable))], "observation_count": int(data.shape[0])}


def _evidence_sensitivity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    model, VariableElimination, _ = _build_model(inputs)
    scenarios = _sequence(inputs.get("evidence_scenarios"), "inputs.evidence_scenarios")
    if not 1 <= len(scenarios) <= 50:
        raise ComputeError("evidence_scenarios must contain 1 to 50 scenarios")
    rows = []
    for index, scenario_raw in enumerate(scenarios):
        scenario = _mapping(scenario_raw, f"inputs.evidence_scenarios[{index}]")
        query_inputs = dict(inputs)
        query_inputs["evidence"] = dict(scenario.get("evidence") or {})
        rows.append({"name": str(scenario.get("name") or f"scenario-{index + 1}"), "evidence": query_inputs["evidence"], "query": _query(model, VariableElimination, query_inputs)})
    return {"scenario_count": len(rows), "scenarios": rows}


def _virtual_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    model, VariableElimination, TabularCPD = _build_model(inputs)
    rows = _sequence(inputs.get("virtual_evidence"), "inputs.virtual_evidence")
    if not 1 <= len(rows) <= 20:
        raise ComputeError("virtual_evidence must contain 1 to 20 entries")
    virtual = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"inputs.virtual_evidence[{index}]")
        variable = str(row.get("variable") or "")
        probabilities = np.asarray([_finite(item, "virtual_evidence.probabilities[]") for item in _sequence(row.get("probabilities"), "virtual_evidence.probabilities")], dtype=float)
        if variable not in model.nodes() or not 2 <= probabilities.size <= MAX_STATES or np.any(probabilities < 0) or not np.isclose(np.sum(probabilities), 1, atol=1e-8):
            raise ComputeError("virtual evidence must target a known node with normalized probabilities")
        state_names_raw = row.get("state_names")
        state_names = {variable: list(state_names_raw)} if isinstance(state_names_raw, Sequence) and not isinstance(state_names_raw, (str, bytes)) else {}
        virtual.append(TabularCPD(variable=variable, variable_card=int(probabilities.size), values=probabilities.reshape(-1, 1), state_names=state_names))
    return {"query": _query(model, VariableElimination, inputs, virtual_evidence=virtual), "virtual_evidence_count": len(virtual)}


def bayesian_network_inference(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode not in MODES:
        raise ComputeError(f"inputs.mode must be one of {', '.join(sorted(MODES))}")
    _dependencies()
    if mode == "fixed_network_inference":
        result = _fixed_inference(inputs)
    elif mode == "expert_prior_update":
        result = _expert_prior(inputs)
    elif mode == "bayesian_parameter_estimation":
        result = _parameter_estimation(inputs, bayesian=True)
    elif mode == "em_parameter_estimation":
        result = _em_estimation(inputs)
    elif mode == "evidence_sensitivity":
        result = _evidence_sensitivity(inputs)
    else:
        result = _virtual_evidence(inputs)
    return {"engine": {"name": "pgmpy-isolated-fixed-adapter", "version": EXPECTED_PGMPY, "network_used": False}, "mode": mode, **result, "causal_structure_claimed": False, "interpretation_boundary": "Network edges encode a supplied dependency structure; they are not automatically learned or asserted as causal."}


OPERATIONS = {"bayesian_network_inference": bayesian_network_inference}
