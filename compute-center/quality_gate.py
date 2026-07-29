#!/usr/bin/env python3
"""Institutional result-quality gate for the isolated compute center.

The module never fetches data and never calls a model. GPTs supplies observed outcomes,
approved benchmark identifiers and immutable evidence hashes. Numerical execution success
is deliberately separated from permission to use a result for a formal decision.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "data-readiness-policy.json"
GOLDEN_MANIFEST_PATH = HERE / "benchmarks" / "golden" / "manifest.json"
FROZEN_MANIFEST_PATH = HERE / "benchmarks" / "frozen-real" / "manifest.json"


def _load_policy() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _approved_benchmark_ids() -> set[str]:
    approved: set[str] = set()
    for path, key in (
        (GOLDEN_MANIFEST_PATH, "cases"),
        (FROZEN_MANIFEST_PATH, "datasets"),
    ):
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document.get(key, []):
            if isinstance(row, Mapping) and row.get("id"):
                approved.add(str(row["id"]))
    return approved


def _finite_sequence(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return array


def _ece(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.size
    score = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        count = int(np.sum(mask))
        if count:
            confidence = float(np.mean(probabilities[mask]))
            accuracy = float(np.mean(outcomes[mask]))
            score += count / total * abs(confidence - accuracy)
    return float(score)


def _population_stability_index(
    reference: np.ndarray,
    recent: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        width = max(float(np.std(reference)), 1.0)
        center = float(np.mean(reference))
        edges = np.linspace(center - 3 * width, center + 3 * width, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    recent_counts, _ = np.histogram(recent, bins=edges)
    epsilon = 1e-6
    reference_share = np.maximum(reference_counts / reference.size, epsilon)
    recent_share = np.maximum(recent_counts / recent.size, epsilon)
    return float(
        np.sum(
            (recent_share - reference_share)
            * np.log(recent_share / reference_share)
        )
    )


def evaluate_feedback(
    ticket: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or _load_policy())
    feedback = ticket.get("calibration_feedback")
    if not isinstance(feedback, Mapping):
        return {
            "provided": False,
            "observation_count": 0,
            "probability_calibration": None,
            "interval_calibration": None,
            "drift": None,
            "source_snapshot_sha256": None,
            "warnings": ["No realized-outcome feedback was supplied."],
        }

    warnings: list[str] = []
    probability_result: dict[str, Any] | None = None
    interval_result: dict[str, Any] | None = None
    drift_result: dict[str, Any] | None = None
    observation_count = 0

    probabilities_raw = feedback.get("predicted_probabilities")
    outcomes_raw = feedback.get("observed_outcomes")
    if probabilities_raw is not None or outcomes_raw is not None:
        probabilities = _finite_sequence(
            probabilities_raw, "predicted_probabilities"
        )
        outcomes = _finite_sequence(outcomes_raw, "observed_outcomes")
        if probabilities.shape != outcomes.shape:
            raise ValueError(
                "predicted_probabilities and observed_outcomes must have equal length"
            )
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("predicted_probabilities must be within [0,1]")
        if np.any((outcomes != 0) & (outcomes != 1)):
            raise ValueError("observed_outcomes must contain only 0 or 1")
        observation_count = max(observation_count, int(probabilities.size))
        clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        probability_result = {
            "count": int(probabilities.size),
            "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
            "expected_calibration_error": _ece(probabilities, outcomes),
            "log_loss": float(
                -np.mean(
                    outcomes * np.log(clipped)
                    + (1 - outcomes) * np.log(1 - clipped)
                )
            ),
            "base_rate": float(np.mean(outcomes)),
            "mean_predicted_probability": float(np.mean(probabilities)),
        }

    intervals = feedback.get("prediction_intervals")
    if intervals is not None:
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("prediction_intervals must be a non-empty array")
        covered = 0
        widths: list[float] = []
        for index, row in enumerate(intervals):
            if not isinstance(row, Mapping):
                raise ValueError(f"prediction_intervals[{index}] must be an object")
            lower = float(row["lower"])
            upper = float(row["upper"])
            actual = float(row["actual"])
            if (
                not all(math.isfinite(item) for item in (lower, upper, actual))
                or lower > upper
            ):
                raise ValueError(f"prediction_intervals[{index}] is invalid")
            covered += int(lower <= actual <= upper)
            widths.append(upper - lower)
        observation_count = max(observation_count, len(intervals))
        interval_result = {
            "count": len(intervals),
            "empirical_coverage": covered / len(intervals),
            "mean_interval_width": float(np.mean(widths)),
        }

    reference_raw = feedback.get("reference_values")
    recent_raw = feedback.get("recent_values")
    if reference_raw is not None or recent_raw is not None:
        reference = _finite_sequence(reference_raw, "reference_values")
        recent = _finite_sequence(recent_raw, "recent_values")
        observation_count = max(observation_count, int(recent.size))
        pooled_scale = max(float(np.std(reference, ddof=0)), 1e-12)
        drift_result = {
            "reference_count": int(reference.size),
            "recent_count": int(recent.size),
            "population_stability_index": _population_stability_index(
                reference, recent
            ),
            "kolmogorov_smirnov_statistic": float(
                stats.ks_2samp(reference, recent, method="auto").statistic
            ),
            "standardized_mean_shift": float(
                abs(np.mean(recent) - np.mean(reference)) / pooled_scale
            ),
            "reference_mean": float(np.mean(reference)),
            "recent_mean": float(np.mean(recent)),
        }

    minimum = int(
        policy.get("calibration_thresholds", {}).get(
            "minimum_feedback_observations", 20
        )
    )
    if observation_count and observation_count < minimum:
        warnings.append(
            f"Feedback sample is below the institutional minimum of {minimum} observations."
        )

    return {
        "provided": True,
        "observation_count": observation_count,
        "probability_calibration": probability_result,
        "interval_calibration": interval_result,
        "drift": drift_result,
        "source_snapshot_sha256": feedback.get("source_snapshot_sha256"),
        "warnings": warnings,
    }


def build_quality_report(
    ticket: Mapping[str, Any],
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or _load_policy())
    profile = (
        ticket.get("quality_profile")
        if isinstance(ticket.get("quality_profile"), Mapping)
        else {}
    )
    decision_class = str(profile.get("decision_class") or "formal")
    profiles = policy.get("decision_profiles") or {}
    profile_policy = dict(
        profiles.get(decision_class) or profiles.get("formal") or {}
    )
    feedback = evaluate_feedback(ticket, policy)
    checks: list[dict[str, Any]] = []

    def add(
        code: str,
        status: str,
        message: str,
        *,
        blocking: bool = False,
    ) -> None:
        checks.append(
            {
                "code": code,
                "status": status,
                "blocking": blocking,
                "message": message,
            }
        )

    strict = str(preflight.get("policy", {}).get("enforcement")) == "strict"
    add(
        "STRICT_PREFLIGHT",
        "PASS" if strict else "FAIL",
        "Formal computation must use strict preflight.",
        blocking=decision_class != "exploratory" and not strict,
    )

    assumption_ratio = preflight.get("data_summary", {}).get("assumption_ratio")
    maximum_ratio = float(profile_policy.get("max_assumption_ratio", 0.25))
    if assumption_ratio is None:
        add(
            "ASSUMPTION_RATIO",
            "WARN",
            "Variable-level provenance was not declared; assumption ratio cannot be verified.",
            blocking=decision_class == "high_stakes",
        )
    elif float(assumption_ratio) <= maximum_ratio:
        add(
            "ASSUMPTION_RATIO",
            "PASS",
            f"Assumption ratio {assumption_ratio:.1%} is within {maximum_ratio:.1%}.",
        )
    else:
        add(
            "ASSUMPTION_RATIO",
            "FAIL",
            f"Assumption ratio {assumption_ratio:.1%} exceeds {maximum_ratio:.1%}.",
            blocking=True,
        )

    benchmark_ids = (
        [str(value) for value in profile.get("benchmark_ids", [])]
        if isinstance(profile.get("benchmark_ids"), list)
        else []
    )
    benchmark_required = bool(
        profile_policy.get("benchmark_evidence_required", False)
    )
    approved_benchmarks = _approved_benchmark_ids()
    unknown_benchmarks = sorted(set(benchmark_ids) - approved_benchmarks)
    if unknown_benchmarks:
        add(
            "BENCHMARK_EVIDENCE",
            "FAIL",
            f"Unapproved benchmark IDs: {', '.join(unknown_benchmarks)}.",
            blocking=decision_class == "high_stakes",
        )
    elif benchmark_ids:
        add(
            "BENCHMARK_EVIDENCE",
            "PASS",
            f"Approved benchmark evidence: {', '.join(benchmark_ids)}.",
        )
    elif benchmark_required:
        add(
            "BENCHMARK_EVIDENCE",
            "WARN",
            "No approved golden or frozen-real benchmark ID was attached.",
            blocking=decision_class == "high_stakes",
        )
    else:
        add(
            "BENCHMARK_EVIDENCE",
            "NOT_REQUIRED",
            "Benchmark evidence is not required for this profile.",
        )

    ticket_evidence = (
        ticket.get("evidence") if isinstance(ticket.get("evidence"), list) else []
    )
    hashed_evidence = [
        row
        for row in ticket_evidence
        if isinstance(row, Mapping)
        and isinstance(row.get("sha256"), str)
        and len(str(row.get("sha256"))) == 64
    ]
    independent_required = bool(
        profile_policy.get("independent_cross_check_required", False)
    )
    independent_declared = bool(profile.get("independent_cross_check_passed", False))
    cross_check_method = str(profile.get("cross_check_method") or "").strip()
    independent_verified = bool(
        independent_declared and cross_check_method and hashed_evidence
    )
    if independent_required and not independent_verified:
        add(
            "INDEPENDENT_CROSS_CHECK",
            "FAIL",
            "High-stakes result requires a passed independent method, a named cross-check method and SHA-256 evidence.",
            blocking=True,
        )
    elif independent_verified:
        add(
            "INDEPENDENT_CROSS_CHECK",
            "PASS",
            f"Independent cross-check is evidenced using {cross_check_method}.",
        )

    user_approval_required = bool(
        profile_policy.get("user_approval_required", False)
    )
    user_approved = bool(profile.get("user_approved_for_high_stakes", False))
    if user_approval_required and not user_approved:
        add(
            "HIGH_STAKES_USER_APPROVAL",
            "FAIL",
            "High-stakes decision use requires explicit user approval in the ticket.",
            blocking=True,
        )
    elif user_approval_required:
        add(
            "HIGH_STAKES_USER_APPROVAL",
            "PASS",
            "Explicit high-stakes user approval is recorded.",
        )

    probabilistic_claim = bool(profile.get("probabilistic_claim", False)) or (
        feedback.get("probability_calibration") is not None
    )
    calibration_required = bool(
        profile_policy.get("calibration_required_for_probabilistic_claims", False)
    ) and probabilistic_claim
    calibration_thresholds = policy.get("calibration_thresholds") or {}
    minimum_feedback = int(
        calibration_thresholds.get("minimum_feedback_observations", 20)
    )
    probability = feedback.get("probability_calibration")
    if calibration_required and not probability:
        add(
            "PROBABILITY_CALIBRATION",
            "FAIL",
            "Probabilistic decision claim has no realized-outcome calibration feedback.",
            blocking=decision_class == "high_stakes",
        )
    elif probability:
        brier = float(probability["brier_score"])
        ece = float(probability["expected_calibration_error"])
        block = (
            brier > float(calibration_thresholds.get("brier_score_block", 0.35))
            or ece
            > float(
                calibration_thresholds.get(
                    "expected_calibration_error_block", 0.2
                )
            )
        )
        warn = (
            brier > float(calibration_thresholds.get("brier_score_warning", 0.25))
            or ece
            > float(
                calibration_thresholds.get(
                    "expected_calibration_error_warning", 0.1
                )
            )
            or int(probability["count"]) < minimum_feedback
        )
        add(
            "PROBABILITY_CALIBRATION",
            "FAIL" if block else "WARN" if warn else "PASS",
            f"Brier={brier:.4f}, ECE={ece:.4f}, n={probability['count']}.",
            blocking=block,
        )

    if decision_class == "high_stakes" and feedback.get("provided"):
        feedback_provenance_ok = bool(feedback.get("source_snapshot_sha256"))
        add(
            "FEEDBACK_PROVENANCE",
            "PASS" if feedback_provenance_ok else "FAIL",
            "High-stakes realized outcomes require an immutable source snapshot SHA-256.",
            blocking=not feedback_provenance_ok,
        )

    interval = feedback.get("interval_calibration")
    if interval:
        target = float(
            calibration_thresholds.get("interval_coverage_target", 0.9)
        )
        tolerance = float(
            calibration_thresholds.get("interval_coverage_tolerance", 0.1)
        )
        coverage = float(interval["empirical_coverage"])
        add(
            "INTERVAL_COVERAGE",
            "PASS" if abs(coverage - target) <= tolerance else "WARN",
            f"Empirical interval coverage={coverage:.1%}, target={target:.1%}.",
        )

    drift = feedback.get("drift")
    drift_thresholds = policy.get("drift_thresholds") or {}
    if drift:
        psi = float(drift["population_stability_index"])
        ks = float(drift["kolmogorov_smirnov_statistic"])
        shift = float(drift["standardized_mean_shift"])
        block = (
            psi
            > float(
                drift_thresholds.get("population_stability_index_block", 0.25)
            )
            or ks
            > float(
                drift_thresholds.get("kolmogorov_smirnov_block", 0.3)
            )
            or shift
            > float(
                drift_thresholds.get("standardized_mean_shift_block", 1.0)
            )
        )
        warn = (
            psi
            > float(
                drift_thresholds.get("population_stability_index_warning", 0.1)
            )
            or ks
            > float(
                drift_thresholds.get("kolmogorov_smirnov_warning", 0.15)
            )
            or shift
            > float(
                drift_thresholds.get("standardized_mean_shift_warning", 0.5)
            )
        )
        add(
            "DATA_DRIFT",
            "FAIL" if block else "WARN" if warn else "PASS",
            f"PSI={psi:.4f}, KS={ks:.4f}, standardized mean shift={shift:.4f}.",
            blocking=block,
        )

    execution = (
        result.get("execution")
        if isinstance(result.get("execution"), Mapping)
        else {}
    )
    reproducible = (
        execution.get("network_used") is False
        and int(execution.get("model_calls", 0)) == 0
    )
    add(
        "REPRODUCIBILITY",
        "PASS" if reproducible else "FAIL",
        "Decision result must be produced with zero network and zero model calls.",
        blocking=not reproducible,
    )

    blocking = [
        row for row in checks if row["blocking"] and row["status"] == "FAIL"
    ]
    warnings = [row for row in checks if row["status"] == "WARN"]
    nonblocking_failures = [
        row
        for row in checks
        if not row["blocking"] and row["status"] == "FAIL"
    ]
    if blocking:
        release_status = "DECISION_BLOCKED"
    elif warnings or nonblocking_failures:
        release_status = "DECISION_CONDITIONAL"
    else:
        release_status = "DECISION_RELEASED"

    return {
        "schema_version": "compute-quality-report-v2",
        "task_id": str(ticket.get("task_id") or ""),
        "operation": str(ticket.get("operation") or ""),
        "decision_class": decision_class,
        "release_status": release_status,
        "decision_grade": release_status == "DECISION_RELEASED",
        "checks": checks,
        "calibration_feedback": feedback,
        "constraints": {
            "compute_result_may_be_read": True,
            "formal_decision_use_allowed": release_status == "DECISION_RELEASED",
            "must_collect_more_feedback": calibration_required
            and (
                not probability
                or int(probability.get("count", 0)) < minimum_feedback
            ),
            "must_recalibrate_before_reuse": any(
                row["code"] in {"PROBABILITY_CALIBRATION", "DATA_DRIFT"}
                and row["status"] == "FAIL"
                for row in checks
            ),
            "must_resolve_evidence_gaps": any(
                row["code"]
                in {
                    "BENCHMARK_EVIDENCE",
                    "INDEPENDENT_CROSS_CHECK",
                    "HIGH_STAKES_USER_APPROVAL",
                    "FEEDBACK_PROVENANCE",
                }
                and row["status"] == "FAIL"
                for row in checks
            ),
        },
        "evidence_registry": {
            "approved_benchmark_ids": sorted(approved_benchmarks),
            "submitted_benchmark_ids": benchmark_ids,
            "hashed_evidence_count": len(hashed_evidence),
        },
        "feedback_loop": [
            "GPTs records the decision-time prediction, model version, data snapshot hashes and applicable scope.",
            "After the real outcome is known, GPTs submits predicted probabilities, observed outcomes, intervals and recent/reference samples in a new independent compute ticket.",
            "The compute center calculates calibration and drift without fetching data or changing prior results.",
            "If thresholds fail, GPTs must revise assumptions or model parameters and create a new ticket; old results remain immutable evidence.",
        ],
    }
