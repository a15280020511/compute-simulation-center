#!/usr/bin/env python3
"""Resolve and execute governed domain library entries without network or dynamic code."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


class DomainLibraryError(ValueError):
    """Raised when governed domain material is missing, unknown, or invalid."""


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DomainLibraryError(f"invalid domain library document: {name}")
    return value


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _index(rows: Any, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise DomainLibraryError(f"{label} rows must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DomainLibraryError(f"{label} entry must be an object")
        identifier = str(row.get(key) or "")
        if not identifier or identifier in result:
            raise DomainLibraryError(f"{label} contains empty or duplicate ID")
        result[identifier] = row
    return result


def _resolve(ids: Sequence[str], index: Mapping[str, Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    normalized = [str(item) for item in ids]
    if len(set(normalized)) != len(normalized):
        raise DomainLibraryError(f"duplicate {label} IDs")
    unknown = sorted(set(normalized) - set(index))
    if unknown:
        raise DomainLibraryError(f"unknown {label} IDs: {', '.join(unknown)}")
    return [dict(index[identifier]) for identifier in normalized]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DomainLibraryError(f"{label} must be a finite number")
    return float(value)


def _numbers(value: Any, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DomainLibraryError(f"{label} must be a numeric array")
    result = [_number(item, label) for item in value]
    if not result:
        raise DomainLibraryError(f"{label} must not be empty")
    return result


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    top = _number(numerator, "numerator")
    bottom = _number(denominator, "denominator")
    if bottom == 0:
        raise DomainLibraryError("denominator must not be zero")
    return top / bottom


def _compute_factor_implementation(implementation_id: str, inputs: Mapping[str, Any]) -> float:
    if implementation_id == "safe_ratio":
        values = list(inputs.values())
        if len(values) != 2:
            raise DomainLibraryError("safe_ratio requires exactly two inputs")
        return _safe_ratio(values[0], values[1])
    if implementation_id == "log_positive":
        value = _number(next(iter(inputs.values())), "value")
        if value <= 0:
            raise DomainLibraryError("log_positive requires value > 0")
        return math.log(value)
    if implementation_id in {"growth_rate", "period_return"}:
        values = list(inputs.values())
        if len(values) != 2:
            raise DomainLibraryError(f"{implementation_id} requires exactly two inputs")
        return _safe_ratio(_number(values[1], "current") - _number(values[0], "previous"), values[0])
    if implementation_id == "sample_std":
        values = _numbers(next(iter(inputs.values())), "values")
        if len(values) < 2:
            raise DomainLibraryError("sample_std requires at least two observations")
        return statistics.stdev(values)
    if implementation_id == "hhi":
        shares = _numbers(next(iter(inputs.values())), "shares")
        if any(value < 0 for value in shares):
            raise DomainLibraryError("shares must be non-negative")
        total = sum(shares)
        if total <= 0:
            raise DomainLibraryError("shares must have positive sum")
        normalized = [value / total for value in shares]
        return sum(value * value for value in normalized)
    if implementation_id == "zscore_latest":
        history = _numbers(inputs.get("history"), "history")
        current = _number(inputs.get("current_value"), "current_value")
        if len(history) < 2:
            raise DomainLibraryError("zscore_latest requires at least two historical observations")
        standard_deviation = statistics.stdev(history)
        if standard_deviation == 0:
            raise DomainLibraryError("historical standard deviation must be positive")
        return (current - statistics.mean(history)) / standard_deviation
    raise DomainLibraryError(f"unsupported factor implementation: {implementation_id}")


def compute_registered_factor(factor_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    index = _index(_load("domain-factor-registry.json").get("factors"), "factor_id", "factor library")
    factor = _resolve([factor_id], index, "factor")[0]
    required = [str(item) for item in factor.get("required_inputs", [])]
    if set(inputs) != set(required):
        raise DomainLibraryError(f"factor {factor_id} requires inputs: {', '.join(required)}")
    ordered_inputs = {name: inputs[name] for name in required}
    value = _compute_factor_implementation(str(factor["implementation_id"]), ordered_inputs)
    result = {
        "schema_version": "compute-registered-factor-result-v1",
        "factor_id": factor_id,
        "domain": factor["domain"],
        "implementation_id": factor["implementation_id"],
        "value": value,
        "runtime_network_used": False,
        "arbitrary_code_used": False,
    }
    result["result_sha256"] = _sha(result)
    return result


def _compute_baseline_implementation(implementation_id: str, inputs: Mapping[str, Any]) -> Any:
    if implementation_id == "last_value":
        return _numbers(inputs.get("history"), "history")[-1]
    if implementation_id == "mean":
        return statistics.mean(_numbers(inputs.get("history"), "history"))
    if implementation_id == "seasonal_naive":
        history = _numbers(inputs.get("history"), "history")
        season_length = int(_number(inputs.get("season_length"), "season_length"))
        if season_length <= 0 or season_length > len(history):
            raise DomainLibraryError("season_length must be within history length")
        return history[-season_length]
    if implementation_id == "historical_rate":
        outcomes = _numbers(inputs.get("outcomes"), "outcomes")
        if any(value not in {0.0, 1.0} for value in outcomes):
            raise DomainLibraryError("historical_rate outcomes must be binary")
        return statistics.mean(outcomes)
    if implementation_id == "equal_weight":
        count = int(_number(inputs.get("item_count"), "item_count"))
        if count <= 0 or count > 10000:
            raise DomainLibraryError("item_count must be between 1 and 10000")
        return [1.0 / count] * count
    if implementation_id == "no_change":
        return _number(inputs.get("current_value"), "current_value")
    if implementation_id == "fixed_threshold":
        return int(_number(inputs.get("score"), "score") >= _number(inputs.get("threshold"), "threshold"))
    if implementation_id == "passthrough":
        if len(inputs) != 1:
            raise DomainLibraryError("passthrough requires exactly one input")
        value = next(iter(inputs.values()))
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            return value
        raise DomainLibraryError("unsupported passthrough value")
    raise DomainLibraryError(f"unsupported baseline implementation: {implementation_id}")


def compute_registered_baseline(baseline_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    index = _index(_load("baseline-registry.json").get("baselines"), "baseline_id", "baseline library")
    baseline = _resolve([baseline_id], index, "baseline")[0]
    required = [str(item) for item in baseline.get("required_inputs", [])]
    if set(inputs) != set(required):
        raise DomainLibraryError(f"baseline {baseline_id} requires inputs: {', '.join(required)}")
    ordered_inputs = {name: inputs[name] for name in required}
    value = _compute_baseline_implementation(str(baseline["implementation_id"]), ordered_inputs)
    result = {
        "schema_version": "compute-registered-baseline-result-v1",
        "baseline_id": baseline_id,
        "implementation_id": baseline["implementation_id"],
        "value": value,
        "runtime_network_used": False,
        "arbitrary_code_used": False,
    }
    result["result_sha256"] = _sha(result)
    return result


def validate_material_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    contract = _load("external-domain-material-contract.json")
    required = [str(item) for item in contract["required_envelope_fields"]]
    missing = [field for field in required if field not in envelope]
    if missing:
        raise DomainLibraryError(f"missing material envelope fields: {', '.join(missing)}")
    if envelope.get("source_center") != "intelligence-center":
        raise DomainLibraryError("domain material must originate from intelligence-center")
    if envelope.get("contains_personal_data") is not False:
        raise DomainLibraryError("personal data is not accepted")
    if envelope.get("material_type") not in contract["accepted_material_types"]:
        raise DomainLibraryError("unknown material type")
    files = envelope.get("files")
    if not isinstance(files, list) or not files:
        raise DomainLibraryError("material files must be a non-empty array")
    for row in files:
        if not isinstance(row, Mapping):
            raise DomainLibraryError("material file entry must be an object")
        if set(contract["file_fields"]) - set(row):
            raise DomainLibraryError("material file fields are incomplete")
        digest = str(row.get("sha256") or "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise DomainLibraryError("material file SHA256 is invalid")
        path = str(row.get("path") or "")
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise DomainLibraryError("material file path is unsafe")
    result = {
        "schema_version": "compute-domain-material-validation-v1",
        "status": "PASS",
        "transfer_id": envelope["transfer_id"],
        "material_type": envelope["material_type"],
        "file_count": len(files),
        "manifest_sha256": envelope["manifest_sha256"],
        "runtime_network_used": False,
    }
    result["validation_sha256"] = _sha(result)
    return result


def resolve_domain_library_selection(quality: Mapping[str, Any]) -> dict[str, Any]:
    factor_index = _index(_load("domain-factor-registry.json").get("factors"), "factor_id", "factor library")
    baseline_index = _index(_load("baseline-registry.json").get("baselines"), "baseline_id", "baseline library")
    rule_index = _index(_load("domain-rule-snapshot-registry.json").get("snapshots"), "rule_snapshot_id", "domain rule snapshot library")
    crosswalk_index = _index(_load("ontology-crosswalk-registry.json").get("crosswalks"), "crosswalk_id", "crosswalk library")
    event_index = _index(_load("regime-event-registry.json").get("events"), "event_id", "regime event library")
    feedback_index = _index(_load("outcome-feedback-registry.json").get("records"), "feedback_id", "outcome feedback library")

    factors = _resolve(quality.get("factor_ids", []), factor_index, "factor")
    baselines = _resolve(quality.get("baseline_ids", []), baseline_index, "baseline")
    rules = _resolve(quality.get("domain_rule_snapshot_ids", []), rule_index, "domain rule snapshot")
    crosswalks = _resolve(quality.get("crosswalk_ids", []), crosswalk_index, "crosswalk")
    events = _resolve(quality.get("regime_event_ids", []), event_index, "regime event")
    feedback = _resolve(quality.get("outcome_feedback_ids", []), feedback_index, "outcome feedback")

    report = {
        "schema_version": "compute-domain-library-selection-v1",
        "factors": factors,
        "baselines": baselines,
        "domain_rule_snapshots": rules,
        "crosswalks": crosswalks,
        "regime_events": events,
        "outcome_feedback": feedback,
        "runtime_network_used": False,
        "database_server_used": False,
    }
    report["selection_sha256"] = _sha(report)
    return report
