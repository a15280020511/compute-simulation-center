#!/usr/bin/env python3
"""Fixed stdin/stdout worker for the isolated causal-policy runtime."""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from importlib.metadata import version

EXPECTED = {
    "dowhy": "0.14",
    "numpy": "2.4.6",
    "scipy": "1.15.3",
    "networkx": "3.6.1",
    "jsonschema": "4.26.0",
}


def _verify_versions() -> None:
    observed = {name: version(name) for name in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError(f"causal isolated runtime version mismatch: {observed!r}")


def main() -> int:
    _verify_versions()
    payload = json.load(sys.stdin)
    if not isinstance(payload, Mapping):
        raise ValueError("causal worker input must be a JSON object")
    from causal_policy_operations import causal_policy_evaluation

    result = causal_policy_evaluation(payload)
    if not isinstance(result, Mapping):
        raise RuntimeError("causal worker returned a non-object result")
    output = dict(result)
    engine = dict(output.get("engine") or {})
    engine["runtime_isolation"] = "fixed-venv"
    engine["scipy_version"] = EXPECTED["scipy"]
    output["engine"] = engine
    json.dump(output, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
