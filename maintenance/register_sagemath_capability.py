#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTER = ROOT / "compute-center"

modes = {
    "simplify": {"maturity": "controlled-preview", "network_policy": "deny", "deterministic": True, "limits": {"max_expression_characters": 2000, "max_variables": 20}},
    "solve": {"maturity": "controlled-preview", "network_policy": "deny", "deterministic": True, "limits": {"max_expression_characters": 2000, "max_variables": 20}},
    "differentiate": {"maturity": "controlled-preview", "network_policy": "deny", "deterministic": True, "limits": {"max_expression_characters": 2000, "max_order": 10}},
    "integrate": {"maturity": "controlled-preview", "network_policy": "deny", "deterministic": True, "limits": {"max_expression_characters": 2000}},
    "matrix_analysis": {"maturity": "controlled-preview", "network_policy": "deny", "deterministic": True, "limits": {"max_rows": 20, "max_columns": 20}},
    "number_theory": {"maturity": "controlled-preview", "network_policy": "deny", "deterministic": True, "limits": {"max_values": 20, "max_absolute_integer": 10**18}},
}

registry_path = CENTER / "tool-registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
registry["groups"] = [group for group in registry["groups"] if group.get("id") != "sagemath-symbolic"]
registry["groups"].append({
    "id": "sagemath-symbolic",
    "module": "sagemath_operations",
    "operations": ["symbolic_mathematics"],
    "input_validation": "mode_allowlist",
    "default_requirements": ["requirements-sagemath.txt"],
    "mode_requirements": {},
    "network_policy": "deny",
    "deterministic": True,
    "maturity": "controlled-preview",
    "resource_limits": {"max_seconds": 90, "max_memory_mb": 3072},
    "rollback": {"stable_module": "sagemath_operations", "strategy": "git-revert"},
    "modes": modes,
})
registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

capabilities_path = CENTER / "compute-capabilities.json"
capabilities = json.loads(capabilities_path.read_text(encoding="utf-8"))
already_registered = any(row.get("id") == "symbolic_mathematics" for row in capabilities["operations"])
capabilities["runtime_packages"]["SageMath"] = "10.9 official image pinned by exact repository digest; Docker network=none, read-only root and no-new-privileges"
capabilities["operations"] = [row for row in capabilities["operations"] if row.get("id") != "symbolic_mathematics"]
capabilities["operations"].append({
    "id": "symbolic_mathematics",
    "engine": "SageMath 10.9 exact-digest offline container",
    "availability": "controlled-preview",
    "use_when": "exact symbolic simplification, equation solving, calculus, matrix algebra or number theory are required",
    "typical_output": "exact symbolic expressions, solutions, derivatives, integrals, matrix invariants or number-theory results",
})
capabilities["operation_count"] = len(capabilities["operations"])
if not already_registered:
    capabilities["managed_mode_count"] = int(capabilities["managed_mode_count"]) + 6
    capabilities["effective_managed_mode_count"] = int(capabilities["effective_managed_mode_count"]) + 6
scope_line = "exact symbolic algebra, equation solving, calculus, matrix invariants and number theory in a pinned offline SageMath runtime"
if scope_line not in capabilities["toolkit_assessment"]["scope"]:
    capabilities["toolkit_assessment"]["scope"].append(scope_line)
capabilities_path.write_text(json.dumps(capabilities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

matrix_path = CENTER / "systems-computation-matrix.json"
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
matrix["routes"]["symbolic_mathematics"] = {
    "problem_class": "exact symbolic mathematics",
    "system_level": "mechanism",
    "feedback_structure": "closed-form symbolic transformation with exact-runtime verification",
    "required_gates": ["input_quality", "assumption_register", "constraint_feasibility", "external_validation"],
}
matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps({
    "status": "PASS",
    "operation": "symbolic_mathematics",
    "modes": sorted(modes),
    "operation_count": capabilities["operation_count"],
    "managed_mode_count": capabilities["managed_mode_count"],
    "effective_managed_mode_count": capabilities["effective_managed_mode_count"],
}, ensure_ascii=False))
