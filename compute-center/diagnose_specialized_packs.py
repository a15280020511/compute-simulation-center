#!/usr/bin/env python3
"""Emit compact machine-readable diagnostics for one specialized capability pack."""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path
from validate_specialized_packs import PACKS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=sorted(PACKS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        results = PACKS[args.pack](args.seed)
        proof = {
            "status": "PASS",
            "pack": args.pack,
            "completed_modes": sorted(results),
            "mode_count": len(results),
        }
        code = 0
    except Exception as exc:
        lines = traceback.format_exc().splitlines()
        proof = {
            "status": "FAIL",
            "pack": args.pack,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": lines[-18:],
        }
        code = 1
    text = json.dumps(proof, ensure_ascii=False, indent=2)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
