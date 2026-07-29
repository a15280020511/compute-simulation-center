#!/usr/bin/env python3
"""Risk-aware experiment-design assurance for deterministic and stochastic operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "experiment-design-registry.json"
SCHEMA_PATH = HERE / "experiment-profile.schema.json"
STOCHASTIC = {"monte_carlo","discrete_event_simulation","agent_evolution","markov_simulation","agent_based_simulation","information_diffusion_analysis"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assess_experiment(ticket: Mapping[str, Any]) -> dict[str, Any]:
    registry = _load(REGISTRY_PATH)
    designs = {str(row["id"]): row for row in registry["designs"]}
    profile = ticket.get("experiment_profile") if isinstance(ticket.get("experiment_profile"), Mapping) else None
    operation = str(ticket.get("operation") or "")
    decision_class = str(((ticket.get("quality_profile") or {}).get("decision_class") if isinstance(ticket.get("quality_profile"), Mapping) else None) or "formal")
    issues: list[dict[str, Any]] = []
    if profile is None:
        if operation in STOCHASTIC:
            issues.append({"code":"EXPERIMENT_PROFILE_MISSING","status":"WARN","blocking":decision_class == "high_stakes","message":"Formal stochastic simulation should declare seeds, replications, aggregation and stopping rules."})
        status = "BLOCKED" if any(row["blocking"] for row in issues) else "WARN" if issues else "NOT_REQUIRED"
        return {"schema_version":"compute-experiment-assurance-v1","status":status,"operation":operation,"design_id":None,"issues":issues}
    schema = _load(SCHEMA_PATH); Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(profile), key=lambda item:list(item.absolute_path))
    if errors:
        first = errors[0]; where = ".".join(str(item) for item in first.absolute_path) or "$"
        raise ValueError(f"experiment_profile {where}: {first.message}")
    design_id = str(profile["design_id"])
    design = designs.get(design_id)
    if design is None:
        issues.append({"code":"UNKNOWN_EXPERIMENT_DESIGN","status":"FAIL","blocking":True,"message":"Experiment design is not registered."})
    elif operation not in design.get("applies_to", []):
        issues.append({"code":"EXPERIMENT_DESIGN_MISMATCH","status":"FAIL","blocking":True,"message":"Experiment design is not approved for this operation."})
    if operation in STOCHASTIC:
        if profile.get("base_seed") is None and not isinstance((ticket.get("inputs") or {}).get("seed"), int):
            issues.append({"code":"SEED_MISSING","status":"FAIL","blocking":decision_class != "exploratory","message":"Stochastic formal use requires a fixed seed."})
        replications = int(profile.get("replications", 1))
        minimum = 10 if decision_class == "high_stakes" else 3 if decision_class == "formal" else 1
        if replications < minimum:
            issues.append({"code":"INSUFFICIENT_REPLICATIONS","status":"WARN","blocking":decision_class == "high_stakes","message":f"Declared replications {replications} are below the {minimum} run profile minimum."})
        if not profile.get("stopping_rule"):
            issues.append({"code":"STOPPING_RULE_MISSING","status":"WARN","blocking":decision_class == "high_stakes","message":"Stochastic simulation requires an explicit stopping or precision rule."})
    blocking = [row for row in issues if row.get("blocking") and row.get("status") in {"FAIL","WARN"}]
    status = "BLOCKED" if blocking else "WARN" if issues else "PASS"
    return {"schema_version":"compute-experiment-assurance-v1","status":status,"operation":operation,"design_id":design_id,"registered_controls":list(design.get("controls", [])) if design else [],"profile":dict(profile),"issues":issues}
