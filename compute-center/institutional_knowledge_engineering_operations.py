#!/usr/bin/env python3
"""Allowlisted institutional knowledge engineering, operations and model-assurance modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError
from institutional_common import (
    MAX_GRAPH_TRIPLES,
    bounded_text,
    engine,
    integer,
    jsonable,
    matrix,
    safe_names,
    strings,
    vector,
)
from think_tank_common import finite, mapping, sequence


def deterministic_record_linkage(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("splink", "duckdb")
    import pandas as pd
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    raw_records = sequence(inputs.get("records"), "inputs.records")
    if not 2 <= len(raw_records) <= 20_000:
        raise ComputeError("records must contain 2 to 20000 rows")
    records = []
    for index, raw in enumerate(raw_records):
        row = mapping(raw, f"inputs.records[{index}]")
        records.append({str(key): value for key, value in row.items()})
    frame = pd.DataFrame(records)
    id_column = str(inputs.get("id_column") or "unique_id")
    exact_fields = safe_names(inputs.get("exact_fields"), "inputs.exact_fields", maximum=10)
    fuzzy_fields_raw = inputs.get("fuzzy_fields") or []
    fuzzy_fields = safe_names(fuzzy_fields_raw, "inputs.fuzzy_fields", maximum=10) if fuzzy_fields_raw else []
    required = {id_column, *exact_fields, *fuzzy_fields}
    if not required <= set(frame.columns):
        raise ComputeError("record linkage fields are missing")
    if frame[id_column].duplicated().any():
        raise ComputeError("id_column must be unique")
    comparisons = [cl.ExactMatch(field) for field in exact_fields]
    comparisons.extend(cl.LevenshteinAtThresholds(field, [1, 2]) for field in fuzzy_fields)
    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=id_column,
        blocking_rules_to_generate_predictions=[block_on(*exact_fields)],
        comparisons=comparisons,
        retain_matching_columns=True,
    )
    linker = Linker(frame, settings, DuckDBAPI(), set_up_basic_logging=False)
    linked = linker.inference.deterministic_link().as_pandas_dataframe()
    keep = [column for column in linked.columns if column.endswith("_l") or column.endswith("_r") or column in {"match_key"}]
    output = linked[keep].head(10_000).to_dict(orient="records")
    return {
        "mode": "deterministic_record_linkage",
        "record_count": int(len(frame)),
        "candidate_pair_count": int(len(linked)),
        "pairs": jsonable(output),
        "engine": engine("splink", "duckdb"),
    }


def fuzzy_entity_matching(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("RapidFuzz")
    from rapidfuzz import fuzz, process

    queries = strings(inputs.get("queries"), "inputs.queries", maximum=10_000)
    choices = strings(inputs.get("choices"), "inputs.choices", maximum=50_000)
    limit = integer(inputs.get("limit", 3), "inputs.limit", 1, min(100, len(choices)))
    scorer_name = str(inputs.get("scorer") or "wratio").lower()
    scorers = {
        "ratio": fuzz.ratio,
        "wratio": fuzz.WRatio,
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
    }
    if scorer_name not in scorers:
        raise ComputeError("scorer is not allowlisted")
    matches = {}
    for query in queries:
        matches[query] = [
            {"choice": str(choice), "score": float(score), "index": int(index)}
            for choice, score, index in process.extract(query, choices, scorer=scorers[scorer_name], limit=limit)
        ]
    return {
        "mode": "fuzzy_entity_matching",
        "scorer": scorer_name,
        "matches": matches,
        "engine": engine("RapidFuzz"),
    }


def shacl_graph_validation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("pyshacl", "rdflib")
    from pyshacl import validate
    from rdflib import Graph

    data_turtle = bounded_text(inputs.get("data_turtle"), "inputs.data_turtle")
    shapes_turtle = bounded_text(inputs.get("shapes_turtle"), "inputs.shapes_turtle")
    data_graph = Graph().parse(data=data_turtle, format="turtle")
    shape_graph = Graph().parse(data=shapes_turtle, format="turtle")
    if len(data_graph) > MAX_GRAPH_TRIPLES or len(shape_graph) > MAX_GRAPH_TRIPLES:
        raise ComputeError("RDF graph exceeds triple limit")
    conforms, report_graph, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shape_graph,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        advanced=False,
        js=False,
        debug=False,
    )
    return {
        "mode": "shacl_graph_validation",
        "conforms": bool(conforms),
        "data_triples": int(len(data_graph)),
        "shape_triples": int(len(shape_graph)),
        "report_triples": int(len(report_graph)),
        "report_text": str(report_text)[:20_000],
        "engine": engine("pyshacl", "rdflib"),
    }


def minhash_similarity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("datasketch")
    from datasketch import MinHash

    documents = [strings(row, f"inputs.documents[{index}]", maximum=10_000) for index, row in enumerate(sequence(inputs.get("documents"), "inputs.documents"))]
    if not 2 <= len(documents) <= 1_000:
        raise ComputeError("documents must contain 2 to 1000 token arrays")
    permutations = integer(inputs.get("permutations", 128), "inputs.permutations", 16, 1024)
    sketches = []
    for tokens in documents:
        sketch = MinHash(num_perm=permutations)
        for token in sorted(set(tokens)):
            sketch.update(token.encode("utf-8"))
        sketches.append(sketch)
    matrix_result = np.eye(len(sketches), dtype=float)
    for i in range(len(sketches)):
        for j in range(i + 1, len(sketches)):
            value = float(sketches[i].jaccard(sketches[j]))
            matrix_result[i, j] = value
            matrix_result[j, i] = value
    return {
        "mode": "minhash_similarity",
        "document_count": len(documents),
        "permutations": permutations,
        "similarity_matrix": jsonable(matrix_result),
        "engine": engine("datasketch"),
    }


def control_system_response(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("control")
    import control

    numerator = vector(inputs.get("numerator"), "inputs.numerator", minimum=1, maximum=20)
    denominator = vector(inputs.get("denominator"), "inputs.denominator", minimum=2, maximum=20)
    if denominator[0] == 0:
        raise ComputeError("leading denominator coefficient cannot be zero")
    duration = finite(inputs.get("duration", 20.0), "inputs.duration")
    points = integer(inputs.get("points", 500), "inputs.points", 20, 10_000)
    if duration <= 0:
        raise ComputeError("duration must be positive")
    system = control.TransferFunction(numerator, denominator)
    time = np.linspace(0.0, duration, points)
    response = control.step_response(system, T=time)
    response_time = np.asarray(response.time, dtype=float)
    outputs = np.asarray(response.outputs, dtype=float).reshape(-1)
    final = float(outputs[-1])
    tolerance = max(abs(final) * 0.02, 1e-9)
    outside = np.where(np.abs(outputs - final) > tolerance)[0]
    settling_time = float(response_time[outside[-1] + 1]) if outside.size and outside[-1] + 1 < response_time.size else float(response_time[0])
    return {
        "mode": "control_system_response",
        "stable": bool(np.all(np.real(control.poles(system)) < 0)),
        "poles": [{"real": float(value.real), "imag": float(value.imag)} for value in control.poles(system)],
        "final_value": final,
        "peak_value": float(np.max(outputs)),
        "peak_time": float(response_time[int(np.argmax(outputs))]),
        "settling_time_2pct": settling_time,
        "time": jsonable(response_time),
        "response": jsonable(outputs),
        "engine": engine("control"),
    }


def reliability_weibull_fit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("reliability")
    from reliability.Fitters import Fit_Weibull_2P

    failures = vector(inputs.get("failures"), "inputs.failures", minimum=3, maximum=50_000)
    right_raw = inputs.get("right_censored") or []
    right_censored = vector(right_raw, "inputs.right_censored", minimum=1, maximum=50_000) if right_raw else None
    if np.any(failures <= 0) or (right_censored is not None and np.any(right_censored <= 0)):
        raise ComputeError("failure and censoring times must be positive")
    fit = Fit_Weibull_2P(
        failures=failures,
        right_censored=right_censored,
        show_probability_plot=False,
        print_results=False,
    )
    aicc = getattr(fit, "AICc", None)
    return {
        "mode": "reliability_weibull_fit",
        "failure_count": int(failures.size),
        "right_censored_count": int(0 if right_censored is None else right_censored.size),
        "alpha": float(fit.alpha),
        "beta": float(fit.beta),
        "alpha_standard_error": float(fit.alpha_SE),
        "beta_standard_error": float(fit.beta_SE),
        "log_likelihood": float(fit.loglik),
        "aic": float(aicc) if isinstance(aicc, (int, float, np.number)) else None,
        "bic": float(fit.BIC),
        "engine": engine("reliability"),
    }


def multi_echelon_inventory(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("stockpyl")
    from stockpyl.ssm_serial import optimize_base_stock_levels

    holding_costs = vector(inputs.get("echelon_holding_cost"), "inputs.echelon_holding_cost", minimum=2, maximum=20)
    lead_times = vector(inputs.get("lead_time"), "inputs.lead_time", minimum=holding_costs.size, maximum=holding_costs.size)
    stockout_cost = finite(inputs.get("stockout_cost"), "inputs.stockout_cost")
    demand_mean = finite(inputs.get("demand_mean"), "inputs.demand_mean")
    demand_standard_deviation = finite(inputs.get("demand_standard_deviation"), "inputs.demand_standard_deviation")
    if np.any(holding_costs < 0) or np.any(lead_times < 0) or stockout_cost < 0 or demand_mean < 0 or demand_standard_deviation <= 0:
        raise ComputeError("inventory parameters are invalid")
    levels, cost = optimize_base_stock_levels(
        num_nodes=int(holding_costs.size),
        echelon_holding_cost=holding_costs.tolist(),
        lead_time=lead_times.tolist(),
        stockout_cost=stockout_cost,
        demand_mean=demand_mean,
        demand_standard_deviation=demand_standard_deviation,
        x_num=200,
        d_num=50,
    )
    return {
        "mode": "multi_echelon_inventory",
        "node_count": int(holding_costs.size),
        "echelon_base_stock_levels": jsonable(levels),
        "expected_cost": float(cost),
        "engine": engine("stockpyl"),
    }


def queueing_network_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("Ciw")
    import ciw

    arrival_rates = vector(inputs.get("arrival_rates"), "inputs.arrival_rates", minimum=1, maximum=20)
    service_rates = vector(inputs.get("service_rates"), "inputs.service_rates", minimum=arrival_rates.size, maximum=arrival_rates.size)
    servers = np.asarray([int(item) for item in sequence(inputs.get("servers"), "inputs.servers")], dtype=int)
    if servers.size != arrival_rates.size or np.any(servers < 1) or np.any(servers > 1_000):
        raise ComputeError("servers must contain a positive integer per node")
    routing = matrix(
        inputs.get("routing"),
        "inputs.routing",
        min_rows=arrival_rates.size,
        max_rows=arrival_rates.size,
        min_columns=arrival_rates.size,
        max_columns=arrival_rates.size,
    )
    if routing.shape != (arrival_rates.size, arrival_rates.size) or np.any(routing < 0) or np.any(np.sum(routing, axis=1) > 1 + 1e-9):
        raise ComputeError("routing matrix is invalid")
    duration = finite(inputs.get("duration", 1_000.0), "inputs.duration")
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    if duration <= 0 or np.any(arrival_rates < 0) or np.any(service_rates <= 0):
        raise ComputeError("queueing rates and duration are invalid")
    network = ciw.create_network(
        arrival_distributions=[ciw.dists.Exponential(rate=float(rate)) if rate > 0 else ciw.dists.NoArrivals() for rate in arrival_rates],
        service_distributions=[ciw.dists.Exponential(rate=float(rate)) for rate in service_rates],
        number_of_servers=servers.tolist(),
        routing=routing.tolist(),
    )
    ciw.seed(seed)
    simulation = ciw.Simulation(network)
    simulation.simulate_until_max_time(duration)
    records = simulation.get_all_records()
    waits_by_node: dict[int, list[float]] = {}
    for record in records:
        waits_by_node.setdefault(int(record.node), []).append(float(record.waiting_time))
    return {
        "mode": "queueing_network_simulation",
        "duration": duration,
        "completed_records": len(records),
        "mean_wait_by_node": {
            str(node): float(np.mean(values)) if values else 0.0
            for node, values in sorted(waits_by_node.items())
        },
        "maximum_wait": float(max((record.waiting_time for record in records), default=0.0)),
        "engine": engine("Ciw"),
    }


def job_shop_schedule(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("job-shop-lib")
    from job_shop_lib import JobShopInstance, Operation
    from job_shop_lib.constraint_programming import ORToolsSolver

    raw_jobs = sequence(inputs.get("jobs"), "inputs.jobs")
    if not 1 <= len(raw_jobs) <= 100:
        raise ComputeError("jobs must contain 1 to 100 jobs")
    jobs = []
    total_operations = 0
    for job_index, raw_job in enumerate(raw_jobs):
        raw_operations = sequence(raw_job, f"inputs.jobs[{job_index}]")
        if not 1 <= len(raw_operations) <= 100:
            raise ComputeError("each job must contain 1 to 100 operations")
        operations = []
        for operation_index, raw in enumerate(raw_operations):
            row = mapping(raw, f"inputs.jobs[{job_index}][{operation_index}]")
            machine = integer(row.get("machine"), "machine", 0, 99)
            duration = integer(row.get("duration"), "duration", 1, 1_000_000)
            operations.append(Operation(machines=machine, duration=duration))
        jobs.append(operations)
        total_operations += len(operations)
    if total_operations > 2_000:
        raise ComputeError("job shop exceeds operation limit")
    instance = JobShopInstance(jobs)
    solver = ORToolsSolver(max_time_in_seconds=finite(inputs.get("time_limit_seconds", 10.0), "inputs.time_limit_seconds"))
    schedule = solver(instance)
    return {
        "mode": "job_shop_schedule",
        "job_count": len(jobs),
        "operation_count": total_operations,
        "machine_count": int(instance.num_machines),
        "makespan": int(schedule.makespan()),
        "metadata": jsonable(schedule.metadata),
        "engine": engine("job-shop-lib"),
    }


def fairness_metric_audit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("fairlearn", "scikit-learn")
    from fairlearn.metrics import MetricFrame, selection_rate
    from sklearn.metrics import accuracy_score

    y_true = np.asarray([int(item) for item in sequence(inputs.get("y_true"), "inputs.y_true")], dtype=int)
    y_pred = np.asarray([int(item) for item in sequence(inputs.get("y_pred"), "inputs.y_pred")], dtype=int)
    sensitive = np.asarray(strings(inputs.get("sensitive_features"), "inputs.sensitive_features"))
    if not (len(y_true) == len(y_pred) == len(sensitive)) or len(y_true) < 10:
        raise ComputeError("fairness arrays must align and contain at least 10 rows")
    frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    return {
        "mode": "fairness_metric_audit",
        "overall": jsonable(frame.overall),
        "by_group": jsonable(frame.by_group),
        "difference": jsonable(frame.difference()),
        "ratio": jsonable(frame.ratio()),
        "group_count": int(len(set(sensitive))),
        "engine": engine("fairlearn", "scikit-learn"),
        "use_restriction": "group-level audit only; not an individual eligibility decision",
    }


def label_issue_detection(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("cleanlab")
    from cleanlab.filter import find_label_issues

    labels = np.asarray([int(item) for item in sequence(inputs.get("labels"), "inputs.labels")], dtype=int)
    probabilities = matrix(
        inputs.get("predicted_probabilities"),
        "inputs.predicted_probabilities",
        min_rows=labels.size,
        max_rows=labels.size,
        min_columns=2,
        max_columns=100,
    )
    if probabilities.shape[0] != labels.size or labels.size < 20:
        raise ComputeError("labels and predicted probabilities must align")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ComputeError("labels reference invalid classes")
    if np.any(probabilities < 0) or not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
        raise ComputeError("predicted probabilities must be non-negative and sum to one")
    ranked = np.asarray(
        find_label_issues(
            labels=labels,
            pred_probs=probabilities,
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
        ),
        dtype=int,
    )
    self_confidence = probabilities[np.arange(labels.size), labels]
    return {
        "mode": "label_issue_detection",
        "observation_count": int(labels.size),
        "issue_count": int(ranked.size),
        "ranked_issue_indices": jsonable(ranked[:1_000]),
        "issue_self_confidence": [
            {"index": int(index), "self_confidence": float(self_confidence[index])}
            for index in ranked[:1_000]
        ],
        "engine": engine("cleanlab"),
    }


def linear_model_explanation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("shap", "scikit-learn")
    import shap
    from sklearn.linear_model import LinearRegression

    x = matrix(inputs.get("x"), "inputs.x", min_rows=20, max_rows=20_000, max_columns=50)
    y = vector(inputs.get("y"), "inputs.y", minimum=x.shape[0], maximum=x.shape[0])
    explain_rows = integer(inputs.get("explain_rows", min(100, x.shape[0])), "inputs.explain_rows", 1, min(1_000, x.shape[0]))
    model = LinearRegression().fit(x, y)
    explainer = shap.LinearExplainer(model, x)
    values = explainer(x[:explain_rows])
    mean_absolute = np.mean(np.abs(np.asarray(values.values, dtype=float)), axis=0)
    return {
        "mode": "linear_model_explanation",
        "model_r2": float(model.score(x, y)),
        "base_value": jsonable(np.asarray(values.base_values, dtype=float)[:explain_rows]),
        "shap_values": jsonable(np.asarray(values.values, dtype=float)),
        "mean_absolute_shap": jsonable(mean_absolute),
        "feature_ranking": [int(index) for index in np.argsort(mean_absolute)[::-1]],
        "engine": engine("shap", "scikit-learn"),
    }


def synthetic_tabular_generation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    engine("copulas")
    import pandas as pd
    from copulas.multivariate import GaussianMultivariate

    names = safe_names(inputs.get("column_names"), "inputs.column_names", maximum=30)
    data = matrix(
        inputs.get("data"),
        "inputs.data",
        min_rows=20,
        max_rows=20_000,
        min_columns=len(names),
        max_columns=len(names),
    )
    if data.shape[1] != len(names):
        raise ComputeError("data columns must match column_names")
    rows = integer(inputs.get("rows", 100), "inputs.rows", 1, 10_000)
    seed = integer(inputs.get("seed", 0), "inputs.seed", 0, 2**32 - 1)
    frame = pd.DataFrame(data, columns=names)
    model = GaussianMultivariate()
    model.fit(frame)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        sampled = model.sample(rows)
    finally:
        np.random.set_state(state)
    correlations_real = frame.corr().to_numpy(dtype=float)
    correlations_synthetic = sampled.corr().to_numpy(dtype=float)
    return {
        "mode": "synthetic_tabular_generation",
        "generated_rows": rows,
        "columns": names,
        "synthetic_records": jsonable(sampled.to_dict(orient="records")),
        "mean_absolute_correlation_error": float(np.mean(np.abs(correlations_real - correlations_synthetic))),
        "engine": engine("copulas"),
        "privacy_warning": "synthetic data is not automatically anonymous; disclosure risk requires separate review",
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "deterministic_record_linkage": deterministic_record_linkage,
    "fuzzy_entity_matching": fuzzy_entity_matching,
    "shacl_graph_validation": shacl_graph_validation,
    "minhash_similarity": minhash_similarity,
    "control_system_response": control_system_response,
    "reliability_weibull_fit": reliability_weibull_fit,
    "multi_echelon_inventory": multi_echelon_inventory,
    "queueing_network_simulation": queueing_network_simulation,
    "job_shop_schedule": job_shop_schedule,
    "fairness_metric_audit": fairness_metric_audit,
    "label_issue_detection": label_issue_detection,
    "linear_model_explanation": linear_model_explanation,
    "synthetic_tabular_generation": synthetic_tabular_generation,
}
