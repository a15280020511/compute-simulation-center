#!/usr/bin/env python3
"""One-time deterministic patch for the compute-ticket dependency cache inventory."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / ".github" / "workflows" / "compute-ticket.yml"
ADDITIONS = [
    "compute-center/requirements-final-mlxtend.txt",
    "compute-center/requirements-final-pymdp.txt",
    "compute-center/requirements-final-pyod.txt",
    "compute-center/requirements-final-quantlib.txt",
]

text = PATH.read_text(encoding="utf-8")
missing = [item for item in ADDITIONS if item not in text]
if missing:
    anchor = "            compute-center/requirements-intelligence-problog.txt\n"
    if anchor not in text:
        raise SystemExit("compute-ticket cache anchor not found")
    insertion = "".join(f"            {item}\n" for item in missing)
    text = text.replace(anchor, anchor + insertion, 1)
    PATH.write_text(text, encoding="utf-8")

for item in ADDITIONS:
    if text.count(item) != 1:
        raise SystemExit(f"dependency cache entry must occur exactly once: {item}")

print("FINAL_DEPENDENCY_CACHE_PATCH_PASS")
