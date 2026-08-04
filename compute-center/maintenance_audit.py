#!/usr/bin/env python3
"""Read-only audit for compute-center upgrades, health, retention, and cleanup."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

COMPUTE_ARTIFACT_RETENTION_DAYS = 30
COMPUTE_VALIDATION_RETENTION_DAYS = 14
CACHE_RETENTION_DAYS = 14


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def classify_update(old: str, new: str) -> str:
    pattern = re.compile(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?$")
    old_match = pattern.fullmatch(old.strip())
    new_match = pattern.fullmatch(new.strip())
    if not old_match or not new_match:
        return "unknown"
    old_parts = tuple(int(item or 0) for item in old_match.groups())
    new_parts = tuple(int(item or 0) for item in new_match.groups())
    if new_parts[0] != old_parts[0]:
        return "major"
    if new_parts[1] != old_parts[1]:
        return "minor"
    if new_parts[2] != old_parts[2]:
        return "patch"
    return "same"


def artifact_retention_days(name: str) -> int | None:
    if name.startswith("compute-ticket-"):
        return COMPUTE_ARTIFACT_RETENTION_DAYS
    if name.startswith("compute-center-validation-"):
        return COMPUTE_VALIDATION_RETENTION_DAYS
    return None


def plan_artifact_cleanup(rows: Iterable[Mapping[str, Any]], now: datetime) -> list[dict[str, Any]]:
    planned = []
    for row in rows:
        name = str(row.get("name") or "")
        retention = artifact_retention_days(name)
        created_at = str(row.get("created_at") or "")
        age_days = (now - _parse_time(created_at)).days if created_at else None
        reason = "expired" if bool(row.get("expired")) else None
        if reason is None and retention is not None and age_days is not None and age_days > retention:
            reason = f"older-than-{retention}-days"
        if reason:
            planned.append({"id": row.get("id"), "name": name, "age_days": age_days, "reason": reason})
    return planned


def plan_cache_cleanup(rows: Iterable[Mapping[str, Any]], now: datetime) -> list[dict[str, Any]]:
    cutoff = now - timedelta(days=CACHE_RETENTION_DAYS)
    planned = []
    for row in rows:
        timestamp = str(row.get("last_accessed_at") or row.get("created_at") or "")
        if timestamp and _parse_time(timestamp) < cutoff:
            planned.append({"id": row.get("id"), "key": row.get("key"), "last_accessed_at": timestamp, "reason": f"not-accessed-for-{CACHE_RETENTION_DAYS}-days"})
    return planned


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _requirement_rows(path: Path) -> list[str]:
    return [line.strip() for line in _read(path).splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _exactly_pinned(rows: list[str], *, allow_empty: bool = False) -> bool:
    return (allow_empty or bool(rows)) and all(
        row.startswith("-r ") or re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", row)
        for row in rows
    )


def _sagemath_runtime_is_pinned(runtime: Mapping[str, Any]) -> bool:
    image = str(runtime.get("image") or "")
    return bool(
        runtime.get("schema_version") == "sagemath-runtime-v1"
        and re.fullmatch(r"sagemath/sagemath@sha256:[0-9a-f]{64}", image)
        and runtime.get("network_policy") == "none"
        and runtime.get("read_only_root") is True
        and runtime.get("cap_drop_all") is True
        and runtime.get("no_new_privileges") is True
        and isinstance(runtime.get("timeout_seconds"), int)
        and 1 <= int(runtime["timeout_seconds"]) <= 300
    )


def _cache_dependency_paths(workflow: str) -> set[str]:
    paths: set[str] = set()
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)cache-dependency-path:\s*(.*?)\s*$", line)
        if not match:
            continue
        indent = len(match.group(1))
        value = match.group(2).strip()
        if value and value not in {"|", ">"}:
            paths.add(value.strip("'\""))
            continue
        for child in lines[index + 1 :]:
            if not child.strip():
                continue
            child_indent = len(child) - len(child.lstrip())
            if child_indent <= indent:
                break
            paths.add(child.strip().strip("'\""))
    return paths


def _required_configuration_paths(root: Path, compute: Path) -> list[Path]:
    return [
        compute / "requirements.txt",
        compute / "requirements-mesa.txt",
        compute / "requirements-finance.txt",
        compute / "requirements-sagemath.txt",
        compute / "sagemath-runtime.json",
        compute / "tool-registry.json",
        root / ".github" / "dependabot.yml",
        root / ".github" / "workflows" / "dependabot-auto-merge.yml",
        root / ".github" / "workflows" / "compute-validate.yml",
        root / ".github" / "workflows" / "compute-ticket.yml",
    ]


def _registry_valid(registry: Mapping[str, Any]) -> tuple[bool, list[Any]]:
    groups = registry.get("groups")
    if not isinstance(groups, list):
        return False, []
    typed_groups = [row for row in groups if isinstance(row, Mapping)]
    group_ids = [str(row.get("id") or "") for row in typed_groups]
    valid = (
        len(typed_groups) == len(groups)
        and len(groups) >= 4
        and len(group_ids) == len(set(group_ids))
        and all(group_ids)
        and registry.get("schema_version") == "compute-tool-registry-v1"
        and registry.get("arbitrary_modules_allowed") is False
        and registry.get("arbitrary_requirements_allowed") is False
        and registry.get("default_network_policy") == "deny"
        and all(
            row.get("network_policy") == "deny"
            and isinstance(row.get("rollback"), Mapping)
            and bool(row["rollback"].get("stable_module"))
            for row in typed_groups
        )
    )
    return bool(valid), groups


def _configuration_checks(
    *,
    requirement_sets: Mapping[str, list[str]],
    core_names: set[str],
    registry_valid: bool,
    requirement_paths: list[Path],
    sagemath_runtime_valid: bool,
    dependabot: str,
    auto_merge: str,
    validate: str,
    ticket: str,
) -> list[dict[str, Any]]:
    cache_paths = _cache_dependency_paths(ticket)
    expected_cache_paths = {f"compute-center/{path.name}" for path in requirement_paths}
    return [
        {
            "name": "all-compute-requirements-exactly-pinned",
            "pass": bool(requirement_sets)
            and all(
                _exactly_pinned(
                    rows,
                    allow_empty=(name == "requirements-sagemath.txt" and sagemath_runtime_valid),
                )
                for name, rows in requirement_sets.items()
            ),
        },
        {
            "name": "sagemath-runtime-exact-digest-and-isolated",
            "pass": sagemath_runtime_valid,
        },
        {
            "name": "required-compute-packages",
            "pass": {"jsonschema", "numpy", "scipy", "simpy"}.issubset(core_names),
        },
        {"name": "tool-registry-valid", "pass": registry_valid},
        {
            "name": "all-requirements-in-cache-key",
            "pass": expected_cache_paths <= cache_paths,
            "observed": sorted(cache_paths),
            "expected": sorted(expected_cache_paths),
        },
        {
            "name": "registry-driven-installation",
            "pass": "tool_registry.py requirements" in ticket
            and 'for requirement in "${requirement_files[@]}"' in ticket,
        },
        {
            "name": "dependabot-isolated-directory",
            "pass": 'directory: "/compute-center"' in dependabot,
        },
        {
            "name": "dependabot-minor-patch-group",
            "pass": "compute-center-minor-patch" in dependabot
            and "update-types: [minor, patch]" in dependabot,
        },
        {
            "name": "dependabot-auto-merge-gated-by-compute-ci",
            "pass": "Validate Compute Center" in auto_merge
            and "conclusion == 'success'" in auto_merge,
        },
        {
            "name": "major-updates-not-auto-merged",
            "pass": "Major or unclassified dependency update left for manual review"
            in auto_merge,
        },
        {"name": "weekly-compute-health", "pass": 'cron: "7 5 * * 0"' in validate},
        {
            "name": "scheduled-health-incident",
            "pass": "Compute center health failed" in validate
            and "health recovered" in validate,
        },
        {
            "name": "compute-entry-isolated",
            "pass": "startsWith(github.event.issue.title, '[compute]')" in ticket,
        },
        {"name": "zero-model-secret", "pass": "OPENROUTER_API_KEY" not in ticket},
    ]


def validate_configuration(root: Path) -> dict[str, Any]:
    compute = root / "compute-center"
    requirement_paths = sorted(compute.glob("requirements*.txt"))
    missing = [
        str(path.relative_to(root))
        for path in _required_configuration_paths(root, compute)
        if not path.is_file()
    ]
    if missing:
        return {"status": "FAIL", "missing_files": missing, "checks": []}

    requirement_sets = {
        path.name: _requirement_rows(path) for path in requirement_paths
    }
    core_names = {
        row.split("==", 1)[0].lower()
        for row in requirement_sets["requirements.txt"]
        if "==" in row
    }
    registry = json.loads(_read(compute / "tool-registry.json"))
    if not isinstance(registry, Mapping):
        registry = {}
    runtime = json.loads(_read(compute / "sagemath-runtime.json"))
    if not isinstance(runtime, Mapping):
        runtime = {}
    sagemath_runtime_valid = _sagemath_runtime_is_pinned(runtime)
    registry_status, groups = _registry_valid(registry)
    checks = _configuration_checks(
        requirement_sets=requirement_sets,
        core_names=core_names,
        registry_valid=registry_status,
        requirement_paths=requirement_paths,
        sagemath_runtime_valid=sagemath_runtime_valid,
        dependabot=_read(root / ".github" / "dependabot.yml"),
        auto_merge=_read(root / ".github" / "workflows" / "dependabot-auto-merge.yml"),
        validate=_read(root / ".github" / "workflows" / "compute-validate.yml"),
        ticket=_read(root / ".github" / "workflows" / "compute-ticket.yml"),
    )
    return {
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "requirement_files": requirement_sets,
        "sagemath_runtime_valid": sagemath_runtime_valid,
        "tool_registry_group_count": len(groups),
        "checks": checks,
        "update_classifier_examples": {
            "patch": classify_update("1.2.3", "1.2.4"),
            "minor": classify_update("1.2.3", "1.3.0"),
            "major": classify_update("1.2.3", "2.0.0"),
        },
    }


def _load_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit compute-center upgrade and cleanup controls without deleting data.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifacts-json")
    parser.add_argument("--caches-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    config = validate_configuration(Path(args.root))
    report = {"version": 2, "created_at": now.isoformat(), "configuration": config, "artifact_cleanup_plan": plan_artifact_cleanup(_load_rows(args.artifacts_json), now), "cache_cleanup_plan": plan_cache_cleanup(_load_rows(args.caches_json), now), "read_only": True}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": config["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if config["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
