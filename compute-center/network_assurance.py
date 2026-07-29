#!/usr/bin/env python3
"""Fail closed unless the numerical runtime has no external route."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def verify() -> dict[str, object]:
    checks = []
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53)):
        blocked = False
        error = None
        try:
            with socket.create_connection((host, port), timeout=1.0):
                blocked = False
        except OSError as exc:
            blocked = True
            error = f"{type(exc).__name__}: {exc}"
        checks.append({"host": host, "port": port, "external_connection_blocked": blocked, "error": error})
    status = "PASS" if all(row["external_connection_blocked"] for row in checks) else "FAIL"
    return {
        "schema_version": "compute-network-assurance-v1",
        "status": status,
        "network_namespace_required": True,
        "external_route_available": status != "PASS",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
