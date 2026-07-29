#!/usr/bin/env python3
"""Validate institutional compute catalogs and cross-references."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+==[^=\s]+$")


class GovernanceCatalogError(ValueError):
    pass


def _load(relative: str) -> Any:
    path = HERE / relative
    if not path.is_file():
        raise GovernanceCatalogError(f"missing governance file: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(relative: str) -> dict[str, Any]:
    value = _load(relative)
    Draft202012Validator.check_schema(value)
    return value


def _validate(schema_relative: str, value: Any, label: str) -> None:
    validator = Draft202012Validator(_schema(schema_relative))
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        where = ".".join(str(item) for item in first.absolute_path) or "$"
        raise GovernanceCatalogError(f"{label} {where}: {first.message}")


def _unique(rows: list[Mapping[str, Any]], key: str, label: str) -> set[str]:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise GovernanceCatalogError(f"{label} contains empty or duplicate {key}")
    return set(values)


def _pinned(filename: str) -> bool:
    path = HERE / filename
    if not path.is_file():
        return False
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return bool(rows) and all(PIN_RE.fullmatch(row) for row in rows)


def validate_catalogs() -> dict[str, Any]:
    capabilities = _load("compute-capabilities.json")
    ticket_schema = _schema("compute-ticket.schema.json")
    tool_registry = _load("tool-registry.json")
    model_registry = _load("model-registry.json")
    method_registry = _load("method-registry.json")
    rule_registry = _load("rule-registry.json")
    storage = _load("storage-architecture.json")
    benchmark_registry = _load("benchmark-registry.json")
    institutional_registry = _load("institutional-library-registry.json")
    assumption_library = _load("assumption-library.json")
    distribution_registry = _load("distribution-registry.json")
    scenario_registry = _load("scenario-registry.json")
    experiment_registry = _load("experiment-design-registry.json")
    credibility_registry = _load("credibility-factor-registry.json")
    strategy_registry = _load("strategy-registry.json")
    sample_registry = _load("sample-registry.json")

    schema_files = (
        "model-performance-ledger.schema.json", "assumption-register.schema.json",
        "assumption-library.schema.json", "calibration-profile.schema.json",
        "constraint-profile.schema.json", "validation-profile.schema.json",
        "mechanism-register.schema.json", "experiment-profile.schema.json",
        "credibility-profile.schema.json",
    )
    for filename in schema_files:
        _schema(filename)
    _validate("assumption-library.schema.json", assumption_library, "assumption_library")
    _validate("assumption-register.schema.json", assumption_library.get("assumptions") or [], "assumption_library.assumptions")

    capability_rows = capabilities.get("operations")
    groups = tool_registry.get("groups")
    models = model_registry.get("models")
    methods = method_registry.get("installed_method_packs")
    rules = rule_registry.get("rules")
    categories = benchmark_registry.get("categories")
    libraries = institutional_registry.get("libraries")
    distributions = distribution_registry.get("distributions")
    scenarios = scenario_registry.get("scenario_types")
    designs = experiment_registry.get("designs")
    factors = credibility_registry.get("factors")
    strategies = strategy_registry.get("strategies")
    samples = sample_registry.get("samples")
    collections = (capability_rows, groups, models, methods, rules, categories, libraries, distributions, scenarios, designs, factors, strategies, samples)
    if not all(isinstance(item, list) for item in collections):
        raise GovernanceCatalogError("governance catalogs contain invalid row collections")

    capability_operations = _unique(capability_rows, "id", "capability catalog")
    ticket_operations = set(ticket_schema["properties"]["operation"]["enum"])
    model_operations = {str(row.get("operation") or "") for row in models}
    registered_operations = {str(operation) for group in groups for operation in group.get("operations", [])}
    if capability_operations != ticket_operations or capability_operations != model_operations:
        raise GovernanceCatalogError("capability, ticket and model operation catalogs are inconsistent")
    if not registered_operations.issubset(capability_operations):
        raise GovernanceCatalogError("tool registry contains an operation outside the capability catalog")
    if int(capabilities.get("operation_count", -1)) != len(capability_operations):
        raise GovernanceCatalogError("operation_count does not match capability catalog")
    managed_modes = sum(len(group.get("modes") or {}) for group in groups)
    if int(capabilities.get("managed_mode_count", -1)) != managed_modes:
        raise GovernanceCatalogError("managed_mode_count does not match tool registry")

    method_ids = _unique(methods, "id", "installed method registry")
    for row in methods:
        if row.get("status") != "installed" or row.get("network_policy") != "deny":
            raise GovernanceCatalogError(f"installed method has invalid policy: {row.get('id')}")
        requirement = row.get("requirements")
        if requirement and not _pinned(str(requirement)):
            raise GovernanceCatalogError(f"method requirement is missing or not pinned: {requirement}")
        if not set(str(item) for item in row.get("operations", [])).issubset(capability_operations):
            raise GovernanceCatalogError(f"method references unknown operation: {row.get('id')}")
    for row in method_registry.get("conditional_backends", []) + method_registry.get("external_adapters", []):
        if row.get("status") != "not-installed":
            raise GovernanceCatalogError(f"conditional backend enabled without activation: {row.get('id')}")

    rule_ids = _unique(rules, "id", "rule registry")
    for row in rules:
        if not (HERE / str(row.get("implementation") or "")).is_file():
            raise GovernanceCatalogError(f"rule implementation is missing: {row.get('id')}")
        if row.get("severity") != "blocking":
            raise GovernanceCatalogError(f"institutional rule must fail closed: {row.get('id')}")

    if storage.get("live_external_database_access") is not False or storage.get("database_credentials_allowed") is not False or storage.get("arbitrary_sql_allowed") is not False:
        raise GovernanceCatalogError("compute storage architecture must remain offline and credential-free")
    stores = storage.get("stores")
    if not isinstance(stores, list):
        raise GovernanceCatalogError("storage architecture has no stores")
    store_ids = _unique(stores, "id", "storage architecture")
    required_stores = {"repository-registries", "frozen-input-snapshots", "artifact-evidence", "performance-feedback-ledger"}
    if required_stores - store_ids:
        raise GovernanceCatalogError("required institutional stores are missing")

    category_ids = _unique(categories, "id", "benchmark registry")
    if category_ids != {"golden", "parameter-recovery", "frozen-real", "adversarial", "shadow"}:
        raise GovernanceCatalogError("benchmark categories are incomplete")
    benchmark_case_count = 0
    for row in categories:
        manifest = _load(str(row["manifest"]))
        cases = manifest.get("cases") or manifest.get("datasets") or manifest.get("registered_shadow_programs") or []
        if not isinstance(cases, list):
            raise GovernanceCatalogError(f"invalid benchmark manifest: {row['manifest']}")
        benchmark_case_count += len(cases)

    library_ids = _unique(libraries, "id", "institutional library registry")
    expected_library_ids = {
        "model-library", "method-library", "strategy-library", "rule-library",
        "assumption-library", "distribution-prior-library", "parameter-calibration-library",
        "constraint-library", "scenario-library", "mechanism-coverage-library",
        "experiment-design-library", "data-evidence-library", "sample-library",
        "benchmark-validation-library", "credibility-library", "feedback-results-library",
    }
    if library_ids != expected_library_ids:
        raise GovernanceCatalogError("the sixteen institutional libraries are incomplete")
    for row in libraries:
        if not (HERE / str(row.get("authority") or "")).is_file():
            raise GovernanceCatalogError(f"institutional library authority is missing: {row.get('id')}")
    policy = institutional_registry.get("policy") or {}
    if policy.get("runtime_network_allowed") is not False or policy.get("ticket_supplied_code_allowed") is not False or policy.get("unverified_domain_truth_prepopulation_allowed") is not False or policy.get("database_server_required") is not False:
        raise GovernanceCatalogError("institutional library safety policy is invalid")

    if assumption_library.get("policy", {}).get("domain_assumptions_may_be_prepopulated") is not False:
        raise GovernanceCatalogError("unverified domain assumptions may not be prepopulated")
    assumption_rows = assumption_library.get("assumptions") or []
    assumption_ids = _unique(assumption_rows, "assumption_id", "assumption library") if assumption_rows else set()
    distribution_ids = _unique(distributions, "id", "distribution registry")
    scenario_ids = _unique(scenarios, "id", "scenario registry")
    experiment_ids = _unique(designs, "id", "experiment design registry")
    factor_ids = _unique(factors, "id", "credibility factor registry")
    strategy_ids = _unique(strategies, "id", "strategy registry")
    if len(distribution_ids) < 9 or len(scenario_ids) < 9 or len(experiment_ids) < 8 or len(strategy_ids) < 8:
        raise GovernanceCatalogError("uncertainty, scenario, experiment or strategy registries are incomplete")
    if len(factor_ids) != 12:
        raise GovernanceCatalogError("credibility factor registry must contain 12 factors")
    if credibility_registry.get("policy", {}).get("single_weighted_credibility_score_allowed") is not False:
        raise GovernanceCatalogError("a magic weighted credibility score is forbidden")
    if sample_registry.get("policy", {}).get("sample_hash_required") is not True or sample_registry.get("policy", {}).get("synthetic_samples_must_be_labeled") is not True:
        raise GovernanceCatalogError("sample library provenance policy is incomplete")

    profiles = {}
    for filename in ("exploratory.json", "formal.json", "high-stakes.json"):
        profile = _load(f"acceptance-profiles/{filename}")
        profiles[str(profile.get("id") or "")] = profile
    if set(profiles) != {"exploratory", "formal", "high_stakes"}:
        raise GovernanceCatalogError("acceptance profiles are incomplete")
    if profiles["high_stakes"].get("independent_cross_check_required") is not True or profiles["high_stakes"].get("explicit_user_approval_required") is not True:
        raise GovernanceCatalogError("high-stakes acceptance controls are incomplete")

    return {
        "schema_version": "compute-governance-catalog-validation-v3",
        "status": "PASS",
        "operation_count": len(capability_operations),
        "managed_mode_count": managed_modes,
        "installed_method_pack_count": len(method_ids),
        "rule_count": len(rule_ids),
        "strategy_count": len(strategy_ids),
        "sample_entry_count": len(samples),
        "store_count": len(store_ids),
        "benchmark_category_count": len(category_ids),
        "benchmark_case_count": benchmark_case_count,
        "acceptance_profile_count": len(profiles),
        "institutional_library_count": len(library_ids),
        "assumption_library_entry_count": len(assumption_ids),
        "distribution_count": len(distribution_ids),
        "scenario_type_count": len(scenario_ids),
        "experiment_design_count": len(experiment_ids),
        "credibility_factor_count": len(factor_ids),
        "live_external_database_access": False,
        "database_server_required": False,
        "arbitrary_rule_code_allowed": False,
        "conditional_backends_installed": False,
        "unverified_domain_truth_prepopulation_allowed": False,
        "single_weighted_credibility_score_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate_catalogs(), ensure_ascii=False, indent=2))
