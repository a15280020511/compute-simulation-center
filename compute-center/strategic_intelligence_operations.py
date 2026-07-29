#!/usr/bin/env python3
"""Deterministic strategic decision and intelligence-analysis modes.

These functions consume only ticket-supplied structured evidence. They do not
fetch intelligence, call models, or make claims of certainty.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError

MAX_ALTERNATIVES = 50
MAX_CRITERIA = 50
MAX_SCENARIOS = 100
MAX_HYPOTHESES = 30
MAX_EVIDENCE = 200


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _names(rows: Sequence[Any], name: str, minimum: int, maximum: int) -> list[str]:
    if not minimum <= len(rows) <= maximum:
        raise ComputeError(f"{name} must contain {minimum} to {maximum} entries")
    result = [str(item) for item in rows]
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ComputeError(f"{name} values must be non-empty and unique")
    return result


def weighted_mcda(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw_criteria = _sequence(inputs.get("criteria"), "inputs.criteria")
    if not 1 <= len(raw_criteria) <= MAX_CRITERIA:
        raise ComputeError(f"inputs.criteria must contain 1 to {MAX_CRITERIA} entries")
    criteria: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_criteria):
        row = _mapping(raw, f"inputs.criteria[{index}]")
        name = str(row.get("name") or "")
        direction = str(row.get("direction") or "benefit")
        weight = _finite(row.get("weight"), f"inputs.criteria[{index}].weight")
        if not name or name in seen:
            raise ComputeError("criterion names must be non-empty and unique")
        if direction not in {"benefit", "cost"}:
            raise ComputeError("criterion direction must be benefit or cost")
        if weight < 0:
            raise ComputeError("criterion weights must be non-negative")
        seen.add(name)
        criteria.append({"name": name, "direction": direction, "weight": weight})
    total_weight = sum(row["weight"] for row in criteria)
    if total_weight <= 0:
        raise ComputeError("criterion weights must sum to a positive value")
    for row in criteria:
        row["normalized_weight"] = row["weight"] / total_weight

    raw_alternatives = _sequence(inputs.get("alternatives"), "inputs.alternatives")
    if not 2 <= len(raw_alternatives) <= MAX_ALTERNATIVES:
        raise ComputeError(f"inputs.alternatives must contain 2 to {MAX_ALTERNATIVES} entries")
    names: list[str] = []
    matrix: list[list[float]] = []
    for index, raw in enumerate(raw_alternatives):
        alt = _mapping(raw, f"inputs.alternatives[{index}]")
        name = str(alt.get("name") or "")
        values = _mapping(alt.get("values"), f"inputs.alternatives[{index}].values")
        if not name or name in names:
            raise ComputeError("alternative names must be non-empty and unique")
        if set(values) != {row["name"] for row in criteria}:
            raise ComputeError("each alternative must provide exactly all criteria")
        names.append(name)
        matrix.append([_finite(values[row["name"]], f"alternative[{name}].{row['name']}") for row in criteria])

    values = np.asarray(matrix, dtype=float)
    normalized = np.zeros_like(values)
    constant_criteria: list[str] = []
    for column, criterion in enumerate(criteria):
        low = float(np.min(values[:, column]))
        high = float(np.max(values[:, column]))
        if high == low:
            normalized[:, column] = 1.0
            constant_criteria.append(criterion["name"])
        elif criterion["direction"] == "benefit":
            normalized[:, column] = (values[:, column] - low) / (high - low)
        else:
            normalized[:, column] = (high - values[:, column]) / (high - low)
    weights = np.asarray([row["normalized_weight"] for row in criteria], dtype=float)
    scores = normalized @ weights
    ranking = []
    for index in np.argsort(-scores, kind="stable"):
        ranking.append({
            "rank": len(ranking) + 1,
            "alternative": names[int(index)],
            "score": float(scores[int(index)]),
            "criterion_contributions": {
                criteria[j]["name"]: float(normalized[int(index), j] * weights[j])
                for j in range(len(criteria))
            },
        })
    return {
        "mode": "weighted_mcda",
        "ranking": ranking,
        "recommended_alternative": ranking[0]["alternative"],
        "normalized_weights": {row["name"]: row["normalized_weight"] for row in criteria},
        "constant_criteria": constant_criteria,
        "method": "min-max normalized weighted additive model",
        "decision_support_only": True,
    }


def minimax_regret(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actions = _names(_sequence(inputs.get("actions"), "inputs.actions"), "inputs.actions", 2, MAX_ALTERNATIVES)
    scenarios = _names(_sequence(inputs.get("scenarios"), "inputs.scenarios"), "inputs.scenarios", 2, MAX_SCENARIOS)
    raw = _sequence(inputs.get("payoffs"), "inputs.payoffs")
    if len(raw) != len(actions):
        raise ComputeError("inputs.payoffs row count must equal action count")
    matrix = np.asarray([
        [_finite(item, f"inputs.payoffs[{i}][{j}]") for j, item in enumerate(_sequence(row, f"inputs.payoffs[{i}]"))]
        for i, row in enumerate(raw)
    ], dtype=float)
    if matrix.shape != (len(actions), len(scenarios)):
        raise ComputeError("inputs.payoffs must be an action-by-scenario matrix")
    objective = str(inputs.get("objective") or "maximize")
    if objective not in {"maximize", "minimize"}:
        raise ComputeError("inputs.objective must be maximize or minimize")
    best = np.max(matrix, axis=0) if objective == "maximize" else np.min(matrix, axis=0)
    regret = best - matrix if objective == "maximize" else matrix - best
    worst_regret = np.max(regret, axis=1)
    average_regret = np.mean(regret, axis=1)
    probabilities_raw = inputs.get("probabilities")
    expected_values: np.ndarray | None = None
    if probabilities_raw is not None:
        probabilities = np.asarray([
            _finite(item, f"inputs.probabilities[{i}]")
            for i, item in enumerate(_sequence(probabilities_raw, "inputs.probabilities"))
        ], dtype=float)
        if probabilities.shape != (len(scenarios),) or np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ComputeError("scenario probabilities must be non-negative and match scenarios")
        probabilities = probabilities / probabilities.sum()
        expected_values = matrix @ probabilities
    order = np.lexsort((average_regret, worst_regret))
    ranking = []
    for index in order:
        row = {
            "rank": len(ranking) + 1,
            "action": actions[int(index)],
            "maximum_regret": float(worst_regret[int(index)]),
            "average_regret": float(average_regret[int(index)]),
        }
        if expected_values is not None:
            row["expected_value"] = float(expected_values[int(index)])
        ranking.append(row)
    return {
        "mode": "minimax_regret",
        "objective": objective,
        "ranking": ranking,
        "robust_action": ranking[0]["action"],
        "regret_matrix": {
            actions[i]: {scenarios[j]: float(regret[i, j]) for j in range(len(scenarios))}
            for i in range(len(actions))
        },
        "decision_support_only": True,
    }


def value_of_information(inputs: Mapping[str, Any]) -> dict[str, Any]:
    actions = _names(_sequence(inputs.get("actions"), "inputs.actions"), "inputs.actions", 2, MAX_ALTERNATIVES)
    scenarios = _names(_sequence(inputs.get("scenarios"), "inputs.scenarios"), "inputs.scenarios", 2, MAX_SCENARIOS)
    probabilities = np.asarray([
        _finite(item, f"inputs.probabilities[{i}]")
        for i, item in enumerate(_sequence(inputs.get("probabilities"), "inputs.probabilities"))
    ], dtype=float)
    if probabilities.shape != (len(scenarios),) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ComputeError("probabilities must be non-negative and match scenarios")
    probabilities = probabilities / probabilities.sum()
    matrix = np.asarray([
        [_finite(item, f"inputs.payoffs[{i}][{j}]") for j, item in enumerate(_sequence(row, f"inputs.payoffs[{i}]"))]
        for i, row in enumerate(_sequence(inputs.get("payoffs"), "inputs.payoffs"))
    ], dtype=float)
    if matrix.shape != (len(actions), len(scenarios)):
        raise ComputeError("inputs.payoffs must be an action-by-scenario matrix")
    expected = matrix @ probabilities
    best_index = int(np.argmax(expected))
    expected_with_perfect_information = float(np.sum(np.max(matrix, axis=0) * probabilities))
    expected_without = float(expected[best_index])
    return {
        "mode": "value_of_information",
        "recommended_action": actions[best_index],
        "expected_values": {actions[i]: float(expected[i]) for i in range(len(actions))},
        "expected_value_without_information": expected_without,
        "expected_value_with_perfect_information": expected_with_perfect_information,
        "expected_value_of_perfect_information": max(0.0, expected_with_perfect_information - expected_without),
        "scenario_probabilities": {scenarios[i]: float(probabilities[i]) for i in range(len(scenarios))},
        "decision_support_only": True,
    }


def competing_hypotheses(inputs: Mapping[str, Any]) -> dict[str, Any]:
    hypotheses = _names(_sequence(inputs.get("hypotheses"), "inputs.hypotheses"), "inputs.hypotheses", 2, MAX_HYPOTHESES)
    raw_evidence = _sequence(inputs.get("evidence"), "inputs.evidence")
    if not 1 <= len(raw_evidence) <= MAX_EVIDENCE:
        raise ComputeError(f"inputs.evidence must contain 1 to {MAX_EVIDENCE} entries")
    burdens = {name: 0.0 for name in hypotheses}
    support = {name: 0.0 for name in hypotheses}
    diagnostic_rows: list[dict[str, Any]] = []
    contribution_by_evidence: list[dict[str, float]] = []
    evidence_ids: set[str] = set()
    evidence_order: list[str] = []
    for index, raw in enumerate(raw_evidence):
        row = _mapping(raw, f"inputs.evidence[{index}]")
        evidence_id = str(row.get("id") or "")
        reliability = _finite(row.get("reliability", 1.0), f"evidence[{index}].reliability")
        diagnosticity = _finite(row.get("diagnosticity", 1.0), f"evidence[{index}].diagnosticity")
        ratings = _mapping(row.get("ratings"), f"evidence[{index}].ratings")
        if not evidence_id or evidence_id in evidence_ids:
            raise ComputeError("evidence ids must be non-empty and unique")
        if not 0 <= reliability <= 1 or not 0 <= diagnosticity <= 1:
            raise ComputeError("reliability and diagnosticity must be between 0 and 1")
        if set(ratings) != set(hypotheses):
            raise ComputeError("each evidence item must rate every hypothesis")
        weight = reliability * diagnosticity
        contributions: dict[str, float] = {}
        numeric_ratings: list[float] = []
        for hypothesis in hypotheses:
            rating = _finite(ratings[hypothesis], f"evidence[{evidence_id}].ratings[{hypothesis}]")
            if not -2 <= rating <= 2:
                raise ComputeError("ACH ratings must be between -2 and 2")
            numeric_ratings.append(rating)
            contribution = rating * weight
            contributions[hypothesis] = contribution
            if rating < 0:
                burdens[hypothesis] += -contribution
            else:
                support[hypothesis] += contribution
        evidence_ids.add(evidence_id)
        evidence_order.append(evidence_id)
        contribution_by_evidence.append(contributions)
        diagnostic_rows.append({
            "id": evidence_id,
            "reliability": reliability,
            "diagnosticity": diagnosticity,
            "rating_spread": float(max(numeric_ratings) - min(numeric_ratings)),
        })
    scores = {name: support[name] - burdens[name] for name in hypotheses}
    ranking = sorted(
        ({"hypothesis": name, "net_score": scores[name], "support": support[name], "inconsistency_burden": burdens[name]} for name in hypotheses),
        key=lambda row: (-row["net_score"], row["inconsistency_burden"], row["hypothesis"]),
    )
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    diagnostic_rows.sort(key=lambda row: (-row["rating_spread"] * row["reliability"] * row["diagnosticity"], row["id"]))
    base_winner = ranking[0]["hypothesis"]
    leave_one_out_changes: list[dict[str, Any]] = []
    for evidence_index, evidence_id in enumerate(evidence_order):
        adjusted = {hypothesis: scores[hypothesis] - contribution_by_evidence[evidence_index][hypothesis] for hypothesis in hypotheses}
        winner = max(hypotheses, key=lambda item: (adjusted[item], -burdens[item], item))
        if winner != base_winner:
            leave_one_out_changes.append({"removed_evidence": evidence_id, "new_leader": winner})
    return {
        "mode": "competing_hypotheses",
        "ranking": ranking,
        "leading_hypothesis": base_winner,
        "most_diagnostic_evidence": diagnostic_rows[:10],
        "leave_one_out_leader_changes": leave_one_out_changes,
        "method_warning": "Structured ACH screening does not establish truth; revise ratings when new evidence arrives.",
        "decision_support_only": True,
    }


def indicators_and_warnings(inputs: Mapping[str, Any]) -> dict[str, Any]:
    raw = _sequence(inputs.get("indicators"), "inputs.indicators")
    if not 1 <= len(raw) <= MAX_EVIDENCE:
        raise ComputeError(f"inputs.indicators must contain 1 to {MAX_EVIDENCE} entries")
    rows: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0
    seen: set[str] = set()
    for index, item in enumerate(raw):
        row = _mapping(item, f"inputs.indicators[{index}]")
        name = str(row.get("name") or "")
        current = _finite(row.get("current"), f"indicator[{name}].current")
        warning = _finite(row.get("warning_threshold"), f"indicator[{name}].warning_threshold")
        critical = _finite(row.get("critical_threshold"), f"indicator[{name}].critical_threshold")
        direction = str(row.get("direction") or "higher_is_worse")
        reliability = _finite(row.get("reliability", 1.0), f"indicator[{name}].reliability")
        importance = _finite(row.get("importance", 1.0), f"indicator[{name}].importance")
        if not name or name in seen:
            raise ComputeError("indicator names must be non-empty and unique")
        if direction not in {"higher_is_worse", "lower_is_worse"}:
            raise ComputeError("indicator direction is invalid")
        if not 0 <= reliability <= 1 or importance < 0:
            raise ComputeError("indicator reliability must be in [0,1] and importance non-negative")
        if direction == "higher_is_worse":
            if critical < warning:
                raise ComputeError("critical_threshold must be at least warning_threshold")
            level = 2 if current >= critical else 1 if current >= warning else 0
        else:
            if critical > warning:
                raise ComputeError("critical_threshold must not exceed warning_threshold")
            level = 2 if current <= critical else 1 if current <= warning else 0
        weight = reliability * importance
        weighted_sum += level * weight
        weight_total += 2.0 * weight
        seen.add(name)
        rows.append({"name": name, "current": current, "level": ["normal", "warning", "critical"][level], "reliability": reliability, "importance": importance})
    score = 0.0 if weight_total == 0 else weighted_sum / weight_total
    overall = "critical" if score >= 0.67 else "warning" if score >= 0.33 else "normal"
    rows.sort(key=lambda row: ({"critical": 2, "warning": 1, "normal": 0}[row["level"]], row["importance"] * row["reliability"]), reverse=True)
    return {
        "mode": "indicators_and_warnings",
        "overall_level": overall,
        "normalized_alert_score": float(score),
        "indicators": rows,
        "critical_count": sum(row["level"] == "critical" for row in rows),
        "warning_count": sum(row["level"] == "warning" for row in rows),
        "decision_support_only": True,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "weighted_mcda": weighted_mcda,
    "minimax_regret": minimax_regret,
    "value_of_information": value_of_information,
    "competing_hypotheses": competing_hypotheses,
    "indicators_and_warnings": indicators_and_warnings,
}
