#!/usr/bin/env python3
"""Validate the complete governed domain library set with deterministic offline fixtures."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from domain_library_runtime import compute_registered_baseline, compute_registered_factor

HERE = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "domain-factor-registry.json",
    "baseline-registry.json",
    "metric-threshold-registry.json",
    "domain-rule-snapshot-registry.json",
    "ontology-crosswalk-registry.json",
    "regime-event-registry.json",
    "outcome-feedback-registry.json",
    "external-domain-material-contract.json",
    "institutional-library-registry.json",
]


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {name}")
    return value


def validate() -> dict[str, Any]:
    documents = {name: _load(name) for name in REQUIRED_FILES}
    factors = documents["domain-factor-registry.json"]["factors"]
    baselines = documents["baseline-registry.json"]["baselines"]
    libraries = documents["institutional-library-registry.json"]["libraries"]
    if len(factors) != 20:
        raise ValueError("factor registry must contain 20 controlled-preview factors")
    if len(baselines) < 10:
        raise ValueError("baseline registry is incomplete")
    if len(libraries) < 24:
        raise ValueError("institutional library registry is incomplete")

    for name, field in {
        "domain-rule-snapshot-registry.json": "snapshots",
        "ontology-crosswalk-registry.json": "crosswalks",
        "regime-event-registry.json": "events",
        "outcome-feedback-registry.json": "records",
    }.items():
        if documents[name][field] != [] or documents[name]["status"] != "structure-complete-data-pending":
            raise ValueError(f"{name} must remain explicitly data-pending until real evidence arrives")

    factor_receipt = compute_registered_factor(
        "commercial-conversion-rate", {"transactions": 25, "footfall": 100}
    )
    baseline_receipt = compute_registered_baseline("historical-mean", {"history": [1, 2, 3]})
    if abs(float(factor_receipt["value"]) - 0.25) > 1e-12:
        raise ValueError("factor numerical truth failed")
    if baseline_receipt["value"] != 2:
        raise ValueError("baseline numerical truth failed")

    for source_name in ("domain_library_runtime.py", "library_runtime.py"):
        source = (HERE / source_name).read_text(encoding="utf-8")
        for forbidden in (
            "import requests",
            "import socket",
            "urllib.request",
            "subprocess.",
            "huggingface_hub",
            "HF_TOKEN",
            "OPENROUTER_API_KEY",
            "eval(",
            "exec(",
        ):
            if forbidden in source:
                raise ValueError(f"forbidden runtime capability in {source_name}: {forbidden}")
        tree = ast.parse(source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if {"eval", "exec", "compile"} & calls:
            raise ValueError(f"dynamic code execution in {source_name}")

    return {
        "schema_version": "compute-domain-library-completion-receipt-v1",
        "status": "PASS",
        "registered_library_count": len(libraries),
        "generic_factor_count": len(factors),
        "generic_baseline_count": len(baselines),
        "external_material_registries": 4,
        "external_material_status": "structure-complete-data-pending",
        "compute_runtime_network_used": False,
        "database_credentials_used": False,
        "direct_center_connection": False,
        "model_calls": 0,
        "ticket_supplied_code": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = validate()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
