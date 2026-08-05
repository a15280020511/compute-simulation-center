#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTER = ROOT / "compute-center"
OPERATION = "large_scale_data_intelligence"

schema_path = CENTER / "compute-ticket.schema.json"
schema = json.loads(schema_path.read_text(encoding="utf-8"))
enum = schema["properties"]["operation"]["enum"]
if OPERATION not in enum:
    enum.append(OPERATION)
    enum.sort()
schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

registry_path = CENTER / "model-registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
models = registry["models"]
if not any(row.get("operation") == OPERATION for row in models):
    models.append({
        "model_id": "large-scale-data-intelligence-registered-v1",
        "operation": OPERATION,
        "calibration_supported": False,
        "allowed_backends": [],
        "maturity": "controlled-preview",
        "engineering_maturity": "controlled-preview",
        "evidence_maturity": "experimental",
        "risk_tier": "medium",
        "intended_use": [
            "Bounded offline collision, comparison, numeric profiling, timeline and sparse graph analysis."
        ],
        "prohibited_use": [
            "Unbounded Cartesian joins",
            "Personal identity targeting",
            "External network access",
            "Autonomous enforcement or decision execution"
        ],
        "theoretical_basis": "Blocking, sort-and-sweep, one-pass Welford aggregation and sparse graph algorithms.",
        "known_failure_conditions": [
            "Candidate-pair budget exhausted",
            "Input exceeds registered dimensions",
            "Blocking fields are incomplete or too coarse"
        ],
        "revalidation_trigger": [
            "algorithm change",
            "limit change",
            "performance degradation",
            "scope change"
        ]
    })
    models.sort(key=lambda row: str(row.get("operation") or ""))
registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

test_path = CENTER / "tests" / "test_capability_manager_v2.py"
text = test_path.read_text(encoding="utf-8")
text = text.replace('self.assertEqual(len(groups), 14)', 'self.assertEqual(len(groups), 15)')
text = text.replace('self.assertEqual(len(operations), 24)', 'self.assertEqual(len(operations), 25)')
text = text.replace('            "symbolic_mathematics",\n', '            "symbolic_mathematics", "large_scale_data_intelligence",\n')
test_path.write_text(text, encoding="utf-8")

print("large-scale catalog alignment applied")
