#!/usr/bin/env python3
"""Resolve selected institutional and domain library entries.

Unknown IDs fail closed for formal/high-stakes work. Exploratory tickets are
soft-normalized: known IDs are retained and unknown IDs become explicit
warnings, preventing a harmless annotation error from aborting the numeric run.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from domain_library_runtime import resolve_domain_library_selection

HERE = Path(__file__).resolve().parent


class LibrarySelectionError(ValueError):
    pass


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LibrarySelectionError(f"invalid library document: {name}")
    return value


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _index(rows: Any, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise LibrarySelectionError(f"{label} rows must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise LibrarySelectionError(f"{label} entry must be an object")
        identifier = str(row.get(key) or "")
        if not identifier or identifier in result:
            raise LibrarySelectionError(f"{label} contains empty or duplicate ID")
        result[identifier] = row
    return result


def _resolve(
    ids: list[str],
    index: Mapping[str, Mapping[str, Any]],
    label: str,
    *,
    strict: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(set(ids)) != len(ids):
        raise LibrarySelectionError(f"duplicate {label} IDs")
    unknown = sorted(set(ids) - set(index))
    if unknown and strict:
        raise LibrarySelectionError(f"unknown {label} IDs: {', '.join(unknown)}")
    resolved = [dict(index[identifier]) for identifier in ids if identifier in index]
    warnings = [f"UNKNOWN_{label.upper()}_ID:{identifier}" for identifier in unknown]
    return resolved, warnings


def resolve_library_selection(ticket: Mapping[str, Any]) -> dict[str, Any]:
    quality = ticket.get("quality_profile") if isinstance(ticket.get("quality_profile"), Mapping) else {}
    strategy_index = _index(_load("strategy-registry.json").get("strategies"), "id", "strategy library")
    method_index = _index(_load("method-registry.json").get("installed_method_packs"), "id", "method library")
    sample_index = _index(_load("sample-registry.json").get("samples"), "sample_id", "sample library")
    rule_index = _index(_load("rule-registry.json").get("rules"), "id", "rule library")

    strategy_id = str(quality.get("strategy_id") or "")
    method_ids = [str(item) for item in quality.get("method_ids", [])]
    sample_ids = [str(item) for item in quality.get("sample_ids", [])]
    rule_ids = [str(item) for item in quality.get("rule_ids", [])]
    benchmark_ids = [str(item) for item in quality.get("benchmark_ids", [])]
    decision_class = str(quality.get("decision_class") or "exploratory")
    strict = decision_class in {"formal", "high_stakes"}

    warnings: list[str] = []
    strategy, rows = _resolve([strategy_id], strategy_index, "strategy", strict=strict) if strategy_id else ([], [])
    warnings.extend(rows)
    methods, rows = _resolve(method_ids, method_index, "method", strict=strict)
    warnings.extend(rows)
    samples, rows = _resolve(sample_ids, sample_index, "sample", strict=strict)
    warnings.extend(rows)
    rules, rows = _resolve(rule_ids, rule_index, "rule", strict=strict)
    warnings.extend(rows)
    domain_libraries = resolve_domain_library_selection(quality)

    if strict and not strategy:
        warnings.append("NO_EXPLICIT_DECISION_STRATEGY")
    if sample_ids and not samples:
        warnings.append("NO_RESOLVED_SAMPLE")
    if strict and quality.get("factor_ids") and not quality.get("baseline_ids"):
        warnings.append("FACTOR_WITHOUT_EXPLICIT_BASELINE")

    report: dict[str, Any] = {
        "schema_version": "compute-library-selection-v2",
        "status": "WARN" if warnings else "PASS",
        "decision_class": decision_class,
        "selection_owner": "gpts-usage-center",
        "strategy": strategy,
        "methods": methods,
        "samples": samples,
        "rules": rules,
        "domain_libraries": domain_libraries,
        "benchmark_ids": benchmark_ids,
        "warnings": warnings,
        "runtime_network_used": False,
        "database_server_used": False,
    }
    report["selection_sha256"] = _sha(report)
    return report
