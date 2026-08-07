#!/usr/bin/env python3
"""Deterministic, offline public-justice trend metrics.

The operation only quantifies structured evidence supplied in the ticket. Scores
are transparent bounded evidence indices, not claims about secret capabilities,
absolute institutional strength, or nationwide deployment.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Mapping, Sequence


class JusticeTrendError(ValueError):
    pass


def _seq(value: Any, name: str, maximum: int = 100_000) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise JusticeTrendError(f"{name} must be an array")
    if len(value) > maximum:
        raise JusticeTrendError(f"{name} exceeds maximum length {maximum}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JusticeTrendError(f"{name} must be an object")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JusticeTrendError(f"{name} must be a non-negative integer")
    return value


def _date(value: Any, name: str, allow_none: bool = True) -> date | None:
    if value in (None, "") and allow_none:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise JusticeTrendError(f"{name} must be an ISO date") from exc


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return round(max(low, min(high, value)), 3)


def _sat(count: int, target: int) -> float:
    if target <= 0:
        return 0.0
    return min(1.0, max(0, count) / target)


def _freshness_points(latest: date | None, as_of: date) -> tuple[float, int | None]:
    if latest is None:
        return 0.0, None
    age = max(0, (as_of - latest).days)
    if age <= 30:
        return 10.0, age
    if age <= 90:
        return 8.0, age
    if age <= 180:
        return 5.0, age
    if age <= 365:
        return 2.0, age
    return 0.0, age


def _confidence(primary_count: int, institution_count: int, region_count: int, distinct_dates: int, conflict_count: int) -> str:
    if conflict_count > 0:
        return "CONTESTED_OR_LOW"
    if primary_count >= 5 and institution_count >= 3 and (region_count >= 2 or distinct_dates >= 4):
        return "HIGH"
    if primary_count >= 3 and institution_count >= 2 and distinct_dates >= 2:
        return "MEDIUM_HIGH"
    if primary_count >= 2:
        return "MEDIUM"
    return "LOW"


def _growth(current: int, previous: int) -> float | None:
    if previous <= 0:
        return None if current <= 0 else 1.0
    return (current - previous) / previous


def _window_counts(rows: list[Mapping[str, Any]], as_of: date, windows: list[int]) -> dict[str, Any]:
    parsed = []
    for row in rows:
        event_date = _date(row.get("event_date"), "signal_rows.event_date")
        if event_date is None or event_date > as_of:
            continue
        parsed.append((event_date, row))
    result: dict[str, Any] = {}
    for days in windows:
        recent_start = as_of.fromordinal(as_of.toordinal() - days + 1)
        prior_end = recent_start.fromordinal(recent_start.toordinal() - 1)
        prior_start = prior_end.fromordinal(prior_end.toordinal() - days + 1)
        current = sum(1 for d, _ in parsed if recent_start <= d <= as_of)
        previous = sum(1 for d, _ in parsed if prior_start <= d <= prior_end)
        result[str(days)] = {
            "current_count": current,
            "previous_equal_window_count": previous,
            "growth_rate": _growth(current, previous),
            "growth_interpretation": "NEW_FROM_ZERO_OR_NO_PRIOR_BASE" if previous == 0 and current > 0 else "COMPARABLE",
        }
    return result


def _simple_change_points(rows: list[Mapping[str, Any]], as_of: date, windows: list[int]) -> list[dict[str, Any]]:
    dates = [_date(row.get("event_date"), "signal_rows.event_date") for row in rows]
    dates = [d for d in dates if d is not None and d <= as_of]
    out: list[dict[str, Any]] = []
    for days in windows:
        if days < 14:
            continue
        start = as_of.fromordinal(as_of.toordinal() - days + 1)
        in_window = [d for d in dates if d >= start]
        if len(in_window) < 4:
            out.append({"window_days": days, "status": "INSUFFICIENT_DATA", "event_count": len(in_window)})
            continue
        half = days // 2
        midpoint = as_of.fromordinal(as_of.toordinal() - half + 1)
        earlier = sum(1 for d in in_window if d < midpoint)
        later = sum(1 for d in in_window if d >= midpoint)
        ratio = None if earlier == 0 else later / earlier
        material = earlier >= 2 and (later >= earlier * 2 or later * 2 <= earlier)
        out.append({
            "window_days": days,
            "status": "POSSIBLE_RATE_CHANGE" if material else "NO_MATERIAL_RATE_CHANGE_DETECTED",
            "earlier_half_count": earlier,
            "later_half_count": later,
            "later_to_earlier_ratio": ratio,
            "method": "deterministic_two-half_count_ratio",
            "not_a_causal_claim": True,
        })
    return out


def _capability_score(row: Mapping[str, Any], as_of: date, signal_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    cap = str(row.get("capability_id") or "")
    if not cap:
        raise JusticeTrendError("capability_rows.capability_id is required")
    primary_cases = _nonnegative_int(row.get("primary_case_count", 0), f"{cap}.primary_case_count")
    independent = _nonnegative_int(row.get("independent_institution_count", 0), f"{cap}.independent_institution_count")
    institution_types = _nonnegative_int(row.get("institution_type_count", 0), f"{cap}.institution_type_count")
    regions = _nonnegative_int(row.get("region_count", 0), f"{cap}.region_count")
    standards = _nonnegative_int(row.get("standard_signal_count", 0), f"{cap}.standard_signal_count")
    training = _nonnegative_int(row.get("training_signal_count", 0), f"{cap}.training_signal_count")
    research = _nonnegative_int(row.get("research_signal_count", 0), f"{cap}.research_signal_count")
    procurement = _nonnegative_int(row.get("procurement_signal_count", 0), f"{cap}.procurement_signal_count")
    deployment = _nonnegative_int(row.get("deployment_signal_count", 0), f"{cap}.deployment_signal_count")
    outcomes = _nonnegative_int(row.get("judicial_outcome_support_count", 0), f"{cap}.judicial_outcome_support_count")
    conflicts = _nonnegative_int(row.get("conflict_count", 0), f"{cap}.conflict_count")
    first_seen = _date(row.get("first_seen"), f"{cap}.first_seen")
    latest_seen = _date(row.get("latest_seen"), f"{cap}.latest_seen")
    freshness, age = _freshness_points(latest_seen, as_of)
    source_support = standards + training + research + procurement + deployment

    ces_components = {
        "primary_case": 30 * _sat(primary_cases, 5),
        "institution_diversity": 15 * _sat(independent, 5),
        "region_diversity": 15 * _sat(regions, 5),
        "source_view_support": 15 * _sat(source_support, 5),
        "judicial_outcome_support": 15 * _sat(outcomes, 3),
        "freshness": freshness,
        "conflict_penalty": -min(30.0, conflicts * 10.0),
    }
    ces = _clip(sum(ces_components.values()))

    pss_components = {
        "current_standard_support": 30 * _sat(standards, 2),
        "repeat_practice": 25 * _sat(primary_cases, 3),
        "judicial_review_support": 25 * _sat(outcomes, 3),
        "institutional_repeatability": 20 * _sat(independent, 3),
        "conflict_penalty": -min(30.0, conflicts * 10.0),
    }
    pss = _clip(sum(pss_components.values()))

    cap_signals = [r for r in signal_rows if str(r.get("capability_id") or "") == cap]
    dates = sorted({_date(r.get("event_date"), f"signal[{cap}].event_date") for r in cap_signals})
    dates = [d for d in dates if d is not None]
    recent90 = sum(1 for d in dates if 0 <= (as_of - d).days < 90)
    prior90 = sum(1 for d in dates if 90 <= (as_of - d).days < 180)
    longitudinal = len(dates) >= 3 and (max(dates) - min(dates)).days >= 30 if dates else False
    longitudinal_growth = _growth(recent90, prior90) if longitudinal else None
    growth_points = 0.0
    if longitudinal_growth is not None:
        growth_points = 20.0 * max(-1.0, min(1.0, longitudinal_growth))
    recency_points = 10.0 if age is not None and age <= 90 else (5.0 if age is not None and age <= 180 else 0.0)
    tms_components = {
        "recent_verified_signal_activity": 15 * _sat(recent90, 5),
        "longitudinal_growth": growth_points,
        "new_region_diffusion": 10 * _sat(regions, 4),
        "new_institution_diffusion": 10 * _sat(independent, 4),
        "training_signal": 7.5 * _sat(training, 2),
        "standard_signal": 10 * _sat(standards, 2),
        "procurement_signal": 7.5 * _sat(procurement, 2),
        "deployment_signal": 10 * _sat(deployment, 2),
        "freshness": recency_points,
        "conflict_penalty": -min(25.0, conflicts * 10.0),
    }
    tms = _clip(sum(tms_components.values()))

    persistence_days = 0 if first_seen is None or latest_seen is None else max(0, (latest_seen - first_seen).days)
    di_components = {
        "region": 35 * _sat(regions, 5),
        "institution_type": 25 * _sat(institution_types, 4),
        "independent_institution": 25 * _sat(independent, 6),
        "time_persistence": 15 * min(1.0, persistence_days / 365),
    }
    di = _clip(sum(di_components.values()))

    confidence = _confidence(primary_cases + len(cap_signals), independent, regions, len(dates), conflicts)
    return {
        "capability_id": cap,
        "capability_name": row.get("capability_name"),
        "first_seen": row.get("first_seen"),
        "latest_seen": row.get("latest_seen"),
        "latest_age_days": age,
        "raw_counts": {
            "primary_case_count": primary_cases,
            "independent_institution_count": independent,
            "institution_type_count": institution_types,
            "region_count": regions,
            "standard_signal_count": standards,
            "training_signal_count": training,
            "research_signal_count": research,
            "procurement_signal_count": procurement,
            "deployment_signal_count": deployment,
            "judicial_outcome_support_count": outcomes,
            "conflict_count": conflicts,
            "verified_numeric_signal_row_count": len(cap_signals),
        },
        "CES": ces,
        "CES_components": {k: round(v, 3) for k, v in ces_components.items()},
        "PSS": pss,
        "PSS_components": {k: round(v, 3) for k, v in pss_components.items()},
        "TMS": tms,
        "TMS_components": {k: round(v, 3) for k, v in tms_components.items()},
        "TMS_longitudinal_status": "LONGITUDINAL" if longitudinal else "INSUFFICIENT_LONGITUDINAL_DATA_PROXY_ONLY",
        "TMS_recent90_count": recent90,
        "TMS_prior90_count": prior90,
        "TMS_growth_rate": longitudinal_growth,
        "DI": di,
        "DI_components": {k: round(v, 3) for k, v in di_components.items()},
        "confidence": confidence,
        "limitations": list(row.get("data_limitations") or []) + [
            "scores_measure_strength_of_public_verified_evidence_not_secret_or_absolute_capability",
            "absence_of_public_evidence_is_not_evidence_of_absence",
        ],
    }


def _doctrine_shift(signal_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    doctrine = [r for r in signal_rows if str(r.get("signal_type") or "") == "doctrine_and_commentary"]
    formal = [r for r in signal_rows if str(r.get("signal_type") or "") == "law_and_norm"]
    dates = {str(r.get("event_date") or "") for r in doctrine + formal if r.get("event_date")}
    if not doctrine:
        return {"DSI": None, "status": "INSUFFICIENT_DOCTRINE_EVENTS", "event_count": 0}
    institutions = {str(r.get("institution_type") or "") for r in doctrine if r.get("institution_type")}
    conflicts = sum(1 for r in doctrine if r.get("support_or_conflict") == "CONFLICTS")
    score = (
        30 * _sat(len(doctrine), 6)
        + 25 * _sat(len(institutions), 4)
        + 25 * _sat(len(formal), 4)
        + 20 * _sat(len(dates), 6)
        - min(30, conflicts * 10)
    )
    return {
        "DSI": _clip(score),
        "status": "PUBLIC_EVIDENCE_INDEX",
        "event_count": len(doctrine),
        "formal_rule_signal_count": len(formal),
        "institution_type_count": len(institutions),
        "distinct_date_count": len(dates),
        "conflict_count": conflicts,
        "limitations": ["does_not_measure_unpublished_doctrine_or_internal_policy"],
    }


def _enforcement_shift(signal_rows: list[Mapping[str, Any]], outcome_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_signals = [r for r in signal_rows if str(r.get("signal_type") or "") in {"judicial_outcome", "statistics_and_report"}]
    numeric_total = 0
    changed_dimensions = 0
    dimensions = (
        "case_volume",
        "prosecution_count",
        "non_prosecution_count",
        "evidence_acceptance_count",
        "evidence_exclusion_count",
        "appeal_correction_count",
        "supervision_count",
    )
    by_dim: dict[str, list[int]] = defaultdict(list)
    for row in outcome_rows:
        for dim in dimensions:
            raw = row.get(dim)
            if raw is None:
                continue
            value = _nonnegative_int(raw, f"outcome_rows.{dim}")
            by_dim[dim].append(value)
            numeric_total += value
    for values in by_dim.values():
        if len(values) >= 2 and values[-1] != values[0]:
            changed_dimensions += 1
    if not outcome_signals and not outcome_rows:
        return {"ESI": None, "status": "INSUFFICIENT_ENFORCEMENT_EVENTS", "signal_event_count": 0, "outcome_row_count": 0}
    institutions = {str(r.get("institution_type") or "") for r in outcome_signals if r.get("institution_type")}
    regions = {str(r.get("region") or "") for r in outcome_signals if r.get("region")}
    score = (
        30 * _sat(len(outcome_signals), 8)
        + 20 * _sat(len(outcome_rows), 8)
        + 20 * _sat(changed_dimensions, 4)
        + 15 * _sat(len(institutions), 4)
        + 15 * _sat(len(regions), 5)
    )
    return {
        "ESI": _clip(score),
        "status": "PUBLIC_EVIDENCE_INDEX",
        "signal_event_count": len(outcome_signals),
        "outcome_row_count": len(outcome_rows),
        "changed_numeric_dimensions": changed_dimensions,
        "numeric_total_for_audit_only": numeric_total,
        "institution_type_count": len(institutions),
        "region_count": len(regions),
        "limitations": ["requires_repeated_comparable_outcome_rows_for_strong_enforcement_shift_claims"],
    }


def justice_trend_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    as_of = _date(inputs.get("as_of_date"), "inputs.as_of_date", allow_none=False)
    assert as_of is not None
    windows_raw = _seq(inputs.get("time_windows_days", [30, 90, 180, 365]), "inputs.time_windows_days", 20)
    windows: list[int] = []
    for i, raw in enumerate(windows_raw):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 7 or raw > 3650:
            raise JusticeTrendError(f"inputs.time_windows_days[{i}] must be integer 7..3650")
        if raw not in windows:
            windows.append(raw)
    signal_rows = [
        _mapping(row, f"inputs.signal_rows[{i}]")
        for i, row in enumerate(_seq(inputs.get("signal_rows", []), "inputs.signal_rows"))
    ]
    capability_rows = [
        _mapping(row, f"inputs.capability_rows[{i}]")
        for i, row in enumerate(_seq(inputs.get("capability_rows", []), "inputs.capability_rows", 5000))
    ]
    outcome_rows = [
        _mapping(row, f"inputs.outcome_rows[{i}]")
        for i, row in enumerate(_seq(inputs.get("outcome_rows", []), "inputs.outcome_rows"))
    ]
    for row in signal_rows:
        if row.get("fact_status") not in {None, "PRIMARY_VERIFIED_SIGNAL_EVENT"}:
            raise JusticeTrendError("signal_rows may contain only PRIMARY_VERIFIED_SIGNAL_EVENT rows")
        if row.get("source_primary") is not True:
            raise JusticeTrendError("signal_rows.source_primary must be true")

    capability_results = [_capability_score(row, as_of, signal_rows) for row in capability_rows]
    ranked_tms = sorted(
        capability_results,
        key=lambda row: (float(row["TMS"]), float(row["CES"]), str(row["capability_id"])),
        reverse=True,
    )
    signal_type_counts = Counter(str(r.get("signal_type") or "UNKNOWN") for r in signal_rows)
    institution_type_counts = Counter(str(r.get("institution_type") or "UNKNOWN") for r in signal_rows)
    region_counts = Counter(str(r.get("region") or "UNKNOWN") for r in signal_rows)
    dates = sorted({_date(r.get("event_date"), "signal_rows.event_date") for r in signal_rows})
    dates = [d for d in dates if d is not None]
    longitudinal_status = (
        "LONGITUDINAL"
        if len(dates) >= 4 and (max(dates) - min(dates)).days >= 60
        else "INSUFFICIENT_LONGITUDINAL_DATA"
    )

    result = {
        "schema_version": "justice-trend-analysis-result-v1",
        "as_of_date": as_of.isoformat(),
        "methodology": {
            "type": "deterministic_public_evidence_indices",
            "scores_are_not_secret_capability_estimates": True,
            "scores_are_not_nationwide_deployment_percentages": True,
            "discovery_only_events_allowed": False,
            "network_used": False,
            "model_calls": 0,
        },
        "input_counts": {
            "signal_rows": len(signal_rows),
            "capability_rows": len(capability_rows),
            "outcome_rows": len(outcome_rows),
            "distinct_signal_dates": len(dates),
        },
        "longitudinal_status": longitudinal_status,
        "windowed_verified_signal_counts": _window_counts(signal_rows, as_of, windows),
        "change_point_screen": _simple_change_points(signal_rows, as_of, windows),
        "signal_type_counts": dict(sorted(signal_type_counts.items())),
        "institution_type_counts": dict(sorted(institution_type_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "capabilities": capability_results,
        "technology_momentum_ranking": [
            {
                "rank": i,
                "capability_id": row["capability_id"],
                "TMS": row["TMS"],
                "TMS_longitudinal_status": row["TMS_longitudinal_status"],
                "CES": row["CES"],
                "PSS": row["PSS"],
                "DI": row["DI"],
                "confidence": row["confidence"],
            }
            for i, row in enumerate(ranked_tms, 1)
        ],
        "doctrine_shift": _doctrine_shift(signal_rows),
        "enforcement_shift": _enforcement_shift(signal_rows, outcome_rows),
        "limitations": [
            "indices_measure_public_verified_evidence_strength_and_change_only",
            "absence_of_public_evidence_does_not_establish_absence_of_capability",
            "small_or_nonlongitudinal_samples_must_not_be_described_as_stable_trends",
            "procurement_training_research_signals_do_not_independently_prove_deployment",
            "no_secret_internal_or_covert_operational_capabilities_are_inferred",
        ],
    }
    return result
