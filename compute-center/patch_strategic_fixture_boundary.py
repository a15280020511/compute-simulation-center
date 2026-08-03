#!/usr/bin/env python3
"""One-time deterministic patch separating the 30-mode and 12-mode fixture suites."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "validate_strategic_policy_pack.py"

text = PATH.read_text(encoding="utf-8")
import_anchor = "from strategic_policy_intelligence_operations import HANDLERS\n"
import_line = (
    "from validate_behavior_finance_intelligence_pack import "
    "FIXTURES as BEHAVIOR_FINANCE_FIXTURES\n"
)
if import_line not in text:
    if import_anchor not in text:
        raise SystemExit("strategic validator import anchor not found")
    text = text.replace(import_anchor, import_anchor + import_line, 1)

text = text.replace(
    'parser.add_argument("--mode", choices=sorted(HANDLERS), required=True)',
    'parser.add_argument("--mode", choices=sorted(FIXTURES), required=True)',
    1,
)

old = '''    if set(FIXTURES) != set(HANDLERS):
        raise AssertionError(
            f"fixture mismatch missing={sorted(set(HANDLERS)-set(FIXTURES))} "
            f"extra={sorted(set(FIXTURES)-set(HANDLERS))}"
        )
'''
new = '''    strategic_modes = set(FIXTURES)
    behavior_modes = set(BEHAVIOR_FINANCE_FIXTURES)
    runtime_modes = set(HANDLERS)
    overlap = strategic_modes & behavior_modes
    if overlap:
        raise AssertionError(f"fixture suites overlap: {sorted(overlap)}")
    if strategic_modes | behavior_modes != runtime_modes:
        raise AssertionError(
            f"combined fixture mismatch missing={sorted(runtime_modes-(strategic_modes | behavior_modes))} "
            f"extra={sorted((strategic_modes | behavior_modes)-runtime_modes)}"
        )
    if len(strategic_modes) != 30 or len(behavior_modes) != 12 or len(runtime_modes) != 42:
        raise AssertionError(
            f"unexpected mode counts strategic={len(strategic_modes)} "
            f"behavior={len(behavior_modes)} runtime={len(runtime_modes)}"
        )
'''
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("strategic validator fixture assertion block not found")

PATH.write_text(text, encoding="utf-8")
print("STRATEGIC_FIXTURE_BOUNDARY_PATCH_PASS")
