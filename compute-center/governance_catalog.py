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
SCHEMA_FILES = (
    "model-performance-ledger.schema.json",
    "assumption-register.schema.json",
    "assumption-library.schema.json",
    "calibration-profile.schema.json",
    "constraint-profile.schema.json",
    "validation-profile.schema.json",
    "mechanism-register.schema.json",
    "experiment-profile.schema.json",
    "credibility-profile.schema.json",
)
CATALOG_FILES = {
    "capabilities": "compute-capabilities.json",
    "tool_registry": "tool-registry.json",
    "model_registry": "model-registry.json",
    "method_registry": "method-registry.json",
    "rule_registry": "rule-registry.json",
    "storage": "storage-architecture.json",
    "benchmark_registry": "benchmark-registry.json",
    "institutional_registry": "institutional-library-registry.json",
    "assumption_library": "assumption-library.json",
    "distribution_registry": "distribution-registry.json",
    "scenario_registry": "scenario-registry.json",
    "experiment_registry": "experiment-design-registry.json",
    "credibility_registry": "credibility-factor-registry.json",
    "strategy_registry": "strategy-registry.json",
    "sample_registry": "sample-registry.json",
}
EXPECTED_BENCHMARK_CATEGORIES = {
    "golden",
    "parameter-recovery",
    "frozen-real",
    "adversarial",
    "shadow",
}
EXPECTED_LIBRARY_IDS = {
    "model-library",
    "method-library",
    "strategy-library",
    "rule-library",
    "assumption-library",
    "distribution-prior-library",
    "parameter-calibration-library",
    "constraint-library",
    "scenario-library",
    "mechanism-coverage-library",
    "experiment-design-library",
    "data-evidence-library",
    "sample-library",
    "benchmark-validation-library",
    "credibility-library",
    "feedback-results-library",
}


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
    if not errors:
        return
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
    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return bool(rows) and all(PIN_RE.fullmatch(row) for row in rows)


def _load_catalogs() -> dict[str, Any]:
    return {name: _load(filename) for name, filename in CATALOG_FILES.items()}


def _validate_catalog_schemas(catalogs: Mapping[str, Any]) -> dict[str, Any]:
    for filename in SCHEMA_FILES:
        _schema(filename)
    assumption_library = catalogs["assumption_library"]
    _validate("assumption-library.schema.json", assumption_library, "assumption_library")
    _validate(
        "assumption-register.schema.json",
        assumption_library.get("assumptions") or [],
        "assumption_library.assumptions",
    )
    primary_ticket_schema = _schema("compute-ticket.schema.json")
    dedicated_ticket_schemas = [_schema("sagemath-ticket.schema.json")]
    return {
        "primary": primary_ticket_schema,
        "dedicated": dedicated_ticket_schemas,
    }


def _catalog_rows(catalogs: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    row_keys = {
        "capabilities": ("capabilities", "operations"),
        "groups": ("tool_registry", "groups"),
        "models": ("model_registry", "models"),
        "methods": ("method_registry", "installed_method_packs"),
        "rules": ("rule_registry", "rules"),
        "categories": ("benchmark_registry", "categories"),
        "libraries": ("institutional_registry", "libraries"),
        "distributions": ("distribution_registry", "distributions"),
        "scenarios": ("scenario_registry", "scenario_types"),
        "designs": ("experiment_registry", "designs"),
        "factors": ("credibility_registry", "factors"),
        "strategies": ("strategy_registry", "strategies"),
        "samples": ("sample_registry", "samples"),
    }
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for name, (catalog_name, key) in row_keys.items():
        value = catalogs[catalog_name].get(key)
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            raise GovernanceCatalogError(f"governance catalog {catalog_name}.{key} is not a row collection")
        rows[name] = value
    return rows


def _validate_operations(
    catalogs: Mapping[str, Any],
    rows: Mapping[str, list[Mapping[str, Any]]],
    ticket_schema: Mapping[str, Any],
) -> tuple[set[str], int]:
    capabilities = catalogs["capabilities"]
    capability_operations = _unique(rows["capabilities"], "id", "capability catalog")
    ticket_operations = set(ticket_schema["primary"]["properties"]["operation"]["enum"])
    for dedicated_schema in ticket_schema["dedicated"]:
        operation_schema = dedicated_schema["properties"]["operation"]
        if "const" in operation_schema:
            ticket_operations.add(str(operation_schema["const"]))
        else:
            ticket_operations.update(str(item) for item in operation_schema.get("enum", []))
    model_operations = {str(row.get("operation") or "") for row in rows["models"]}
    registered_operations = {
        str(operation)
        for group in rows["groups"]
        for operation in group.get("operations", [])
    }
    if capability_operations != ticket_operations or capability_operations != model_operations:
        raise GovernanceCatalogError("capability, ticket and model operation catalogs are inconsistent")
    if not registered_operations.issubset(capability_operations):
        raise GovernanceCatalogError("tool registry contains an operation outside the capability catalog")
    if int(capabilities.get("operation_count", -1)) != len(capability_operations):
        raise GovernanceCatalogError("operation_count does not match capability catalog")
    managed_modes = sum(len(group.get("modes") or {}) for group in rows["groups"])
    if int(capabilities.get("managed_mode_count", -1)) != managed_modes:
        raise GovernanceCatalogError("managed_mode_count does not match tool registry")
    return capability_operations, managed_modes


def _validate_methods(
    catalogs: Mapping[str, Any],
    methods: list[Mapping[str, Any]],
    capability_operations: set[str],
) -> set[str]:
    method_ids = _unique(methods, "id", "installed method registry")
    for row in methods:
        method_id = row.get("id")
        if row.get("status") != "installed" or row.get("network_policy") != "deny":
            raise GovernanceCatalogError(f"installed method has invalid policy: {method_id}")
        requirement = row.get("requirements")
        if requirement and not _pinned(str(requirement)):
            raise GovernanceCatalogError(f"method requirement is missing or not pinned: {requirement}")
        operations = {str(item) for item in row.get("operations", [])}
        if not operations.issubset(capability_operations):
            raise GovernanceCatalogError(f"method references unknown operation: {method_id}")
    method_registry = catalogs["method_registry"]
    inactive = method_registry.get("conditional_backends", []) + method_registry.get("external_adapters", [])
    if any(row.get("status") != "not-installed" for row in inactive):
        enabled = next(row.get("id") for row in inactive if row.get("status") != "not-installed")
        raise GovernanceCatalogError(f"conditional backend enabled without activation: {enabled}")
    return method_ids


def _validate_rules(rules: list[Mapping[str, Any]]) -> set[str]:
    rule_ids = _unique(rules, "id", "rule registry")
    for row in rules:
        rule_id = row.get("id")
        if not (HERE / str(row.get("implementation") or "")).is_file():
            raise GovernanceCatalogError(f"rule implementation is missing: {rule_id}")
        if row.get("severity") != "blocking":
            raise GovernanceCatalogError(f"institutional rule must fail closed: {rule_id}")
    return rule_ids


def _validate_storage(storage: Mapping[str, Any]) -> set[str]:
    forbidden = (
        storage.get("live_external_database_access") is not False
        or storage.get("database_credentials_allowed") is not False
        or storage.get("arbitrary_sql_allowed") is not False
    )
    if forbidden:
        raise GovernanceCatalogError("compute storage architecture must remain offline and credential-free")
    stores = storage.get("stores")
    if not isinstance(stores, list) or any(not isinstance(row, Mapping) for row in stores):
        raise GovernanceCatalogError("storage architecture has no stores")
    store_ids = _unique(stores, "id", "storage architecture")
    required = {
        "repository-registries",
        "frozen-input-snapshots",
        "artifact-evidence",
        "performance-feedback-ledger",
    }
    if required - store_ids:
        raise GovernanceCatalogError("required institutional stores are missing")
    return store_ids


def _validate_benchmarks(categories: list[Mapping[str, Any]]) -> tuple[set[str], int]:
    category_ids = _unique(categories, "id", "benchmark registry")
    if category_ids != EXPECTED_BENCHMARK_CATEGORIES:
        raise GovernanceCatalogError("benchmark categories are incomplete")
    case_count = 0
    case_keys = ("cases", "datasets", "registered_shadow_programs")
    for row in categories:
        manifest = _load(str(row["manifest"]))
        cases = next((manifest.get(key) for key in case_keys if key in manifest), [])
        if not isinstance(cases, list):
            raise GovernanceCatalogError(f"invalid benchmark manifest: {row['manifest']}")
        case_count += len(cases)
    return category_ids, case_count


def _validate_institutional_libraries(
    catalogs: Mapping[str, Any],
    libraries: list[Mapping[str, Any]],
) -> set[str]:
    library_ids = _unique(libraries, "id", "institutional library registry")
    if library_ids != EXPECTED_LIBRARY_IDS:
        raise GovernanceCatalogError("the sixteen institutional libraries are incomplete")
    for row in libraries:
        if not (HERE / str(row.get("authority") or "")).is_file():
            raise GovernanceCatalogError(f"institutional library authority is missing: {row.get('id')}")
    policy = catalogs["institutional_registry"].get("policy") or {}
    invalid = (
        policy.get("runtime_network_allowed") is not False
        or policy.get("ticket_supplied_code_allowed") is not False
        or policy.get("unverified_domain_truth_prepopulation_allowed") is not False
        or policy.get("database_server_required") is not False
    )
    if invalid:
        raise GovernanceCatalogError("institutional library safety policy is invalid")
    return library_ids


def _validate_assumptions(catalogs: Mapping[str, Any]) -> set[str]:
    assumption_library = catalogs["assumption_library"]
    if assumption_library.get("policy", {}).get("domain_assumptions_may_be_prepopulated") is not False:
        raise GovernanceCatalogError("unverified domain assumptions may not be prepopulated")
    rows = assumption_library.get("assumptions") or []
    return _unique(rows, "assumption_id", "assumption library") if rows else set()


def _validate_uncertainty_registries(
    catalogs: Mapping[str, Any],
    rows: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, set[str]]:
    identifiers = {
        "distribution_ids": _unique(rows["distributions"], "id", "distribution registry"),
        "scenario_ids": _unique(rows["scenarios"], "id", "scenario registry"),
        "experiment_ids": _unique(rows["designs"], "id", "experiment design registry"),
        "factor_ids": _unique(rows["factors"], "id", "credibility factor registry"),
        "strategy_ids": _unique(rows["strategies"], "id", "strategy registry"),
    }
    if (
        len(identifiers["distribution_ids"]) < 9
        or len(identifiers["scenario_ids"]) < 9
        or len(identifiers["experiment_ids"]) < 8
        or len(identifiers["strategy_ids"]) < 8
    ):
        raise GovernanceCatalogError("uncertainty, scenario, experiment or strategy registries are incomplete")
    if len(identifiers["factor_ids"]) != 12:
        raise GovernanceCatalogError("credibility factor registry must contain 12 factors")
    if catalogs["credibility_registry"].get("policy", {}).get("single_weighted_credibility_score_allowed") is not False:
        raise GovernanceCatalogError("a magic weighted credibility score is forbidden")
    sample_policy = catalogs["sample_registry"].get("policy", {})
    if sample_policy.get("sample_hash_required") is not True or sample_policy.get("synthetic_samples_must_be_labeled") is not True:
        raise GovernanceCatalogError("sample library provenance policy is incomplete")
    return identifiers


def _validate_acceptance_profiles() -> dict[str, Mapping[str, Any]]:
    profiles: dict[str, Mapping[str, Any]] = {}
    for filename in ("exploratory.json", "formal.json", "high-stakes.json"):
        profile = _load(f"acceptance-profiles/{filename}")
        profiles[str(profile.get("id") or "")] = profile
    if set(profiles) != {"exploratory", "formal", "high_stakes"}:
        raise GovernanceCatalogError("acceptance profiles are incomplete")
    high_stakes = profiles["high_stakes"]
    if high_stakes.get("independent_cross_check_required") is not True or high_stakes.get("explicit_user_approval_required") is not True:
        raise GovernanceCatalogError("high-stakes acceptance controls are incomplete")
    return profiles


def validate_catalogs() -> dict[str, Any]:
    catalogs = _load_catalogs()
    ticket_schema = _validate_catalog_schemas(catalogs)
    rows = _catalog_rows(catalogs)
    capability_operations, managed_modes = _validate_operations(catalogs, rows, ticket_schema)
    method_ids = _validate_methods(catalogs, rows["methods"], capability_operations)
    rule_ids = _validate_rules(rows["rules"])
    store_ids = _validate_storage(catalogs["storage"])
    category_ids, benchmark_case_count = _validate_benchmarks(rows["categories"])
    library_ids = _validate_institutional_libraries(catalogs, rows["libraries"])
    assumption_ids = _validate_assumptions(catalogs)
    uncertainty = _validate_uncertainty_registries(catalogs, rows)
    profiles = _validate_acceptance_profiles()

    return {
        "schema_version": "compute-governance-catalog-validation-v3",
        "status": "PASS",
        "operation_count": len(capability_operations),
        "managed_mode_count": managed_modes,
        "installed_method_pack_count": len(method_ids),
        "rule_count": len(rule_ids),
        "strategy_count": len(uncertainty["strategy_ids"]),
        "sample_entry_count": len(rows["samples"]),
        "store_count": len(store_ids),
        "benchmark_category_count": len(category_ids),
        "benchmark_case_count": benchmark_case_count,
        "acceptance_profile_count": len(profiles),
        "institutional_library_count": len(library_ids),
        "assumption_library_entry_count": len(assumption_ids),
        "distribution_count": len(uncertainty["distribution_ids"]),
        "scenario_type_count": len(uncertainty["scenario_ids"]),
        "experiment_design_count": len(uncertainty["experiment_ids"]),
        "credibility_factor_count": len(uncertainty["factor_ids"]),
        "live_external_database_access": False,
        "database_server_required": False,
        "arbitrary_rule_code_allowed": False,
        "conditional_backends_installed": False,
        "unverified_domain_truth_prepopulation_allowed": False,
        "single_weighted_credibility_score_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate_catalogs(), ensure_ascii=False, indent=2))
