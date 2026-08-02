from __future__ import annotations

import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "compute-center" / "compute-ticket.schema.json"
schema = json.loads(path.read_text(encoding="utf-8"))
operations = schema["properties"]["operation"]["enum"]
if "sector_model_analysis" in operations:
    raise SystemExit("sector_model_analysis already exists in ticket schema")
operations.append("sector_model_analysis")
operations.sort()
path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
