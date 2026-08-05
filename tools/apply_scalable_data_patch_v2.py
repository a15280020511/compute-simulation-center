#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tools" / "apply_scalable_data_patch.py"), run_name="__main__")
path = ROOT / "compute-center" / "systems-computation-matrix.json"
value = json.loads(path.read_text(encoding="utf-8"))
value["routes"]["large_scale_data_intelligence"]["system_level"] = "observation"
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("systems matrix level corrected")
