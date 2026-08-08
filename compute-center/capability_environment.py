#!/usr/bin/env python3
"""Prepare fixed per-capability Python environments for dependency-incompatible packs.

This is intentionally not a general environment manager. Only repository-controlled
operations in ISOLATED_ENVIRONMENTS are eligible. Tickets cannot supply interpreter
paths, requirement files, package names, or commands.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE / ".runtime-envs"

ISOLATED_ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "causal_policy_evaluation": {
        "name": "causal-policy",
        "requirements": ["requirements-causal.txt"],
        "reason": "DoWhy 0.14 requires SciPy <=1.15.3 while the core compute runtime pins SciPy 1.18.0",
    }
}


class CapabilityEnvironmentError(RuntimeError):
    """Raised when a repository-controlled isolated environment cannot be prepared safely."""


def metadata_for_ticket(ticket: Mapping[str, Any]) -> dict[str, Any] | None:
    operation = str(ticket.get("operation") or "")
    raw = ISOLATED_ENVIRONMENTS.get(operation)
    if raw is None:
        return None
    name = str(raw["name"])
    requirements = [str(item) for item in raw["requirements"]]
    if not name or not requirements:
        raise CapabilityEnvironmentError(f"invalid isolated environment policy for {operation}")
    paths: list[str] = []
    for filename in requirements:
        path = HERE / filename
        if not path.is_file():
            raise CapabilityEnvironmentError(f"isolated requirement file is missing: {filename}")
        paths.append(str(path))
    env_dir = RUNTIME_ROOT / name
    interpreter = env_dir / "bin" / "python"
    return {
        "operation": operation,
        "name": name,
        "requirements": paths,
        "environment_dir": str(env_dir),
        "interpreter": str(interpreter),
        "reason": str(raw["reason"]),
        "network_policy": "inherit-deny-at-execution",
        "ticket_supplied_requirements_allowed": False,
        "ticket_supplied_commands_allowed": False,
    }


def core_requirement_files(ticket: Mapping[str, Any], requirement_files: list[str]) -> list[str]:
    metadata = metadata_for_ticket(ticket)
    if metadata is None:
        return requirement_files
    isolated = {Path(item).resolve() for item in metadata["requirements"]}
    result = [item for item in requirement_files if Path(item).resolve() not in isolated]
    return result


def _run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise CapabilityEnvironmentError(
            f"isolated environment command failed with exit code {completed.returncode}: {command[0]}"
        )


def prepare(ticket: Mapping[str, Any], *, reset: bool = True) -> dict[str, Any]:
    metadata = metadata_for_ticket(ticket)
    if metadata is None:
        return {
            "status": "NOT_REQUIRED",
            "operation": str(ticket.get("operation") or ""),
            "isolated": False,
        }
    env_dir = Path(metadata["environment_dir"])
    if reset and env_dir.exists():
        shutil.rmtree(env_dir)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if env_dir.exists():
        raise CapabilityEnvironmentError(f"isolated environment path already exists: {env_dir}")
    _run_checked([sys.executable, "-m", "venv", str(env_dir)])
    interpreter = Path(metadata["interpreter"])
    if not interpreter.is_file():
        raise CapabilityEnvironmentError("isolated interpreter was not created")
    pip_command = [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    for requirement in metadata["requirements"]:
        pip_command.extend(["-r", str(requirement)])
    _run_checked(pip_command)
    _run_checked([str(interpreter), "-m", "pip", "check"])
    version_check = subprocess.run(
        [
            str(interpreter),
            "-c",
            (
                "import json; import dowhy, numpy, scipy, networkx, jsonschema; "
                "print(json.dumps({'dowhy': dowhy.__version__, 'numpy': numpy.__version__, "
                "'scipy': scipy.__version__, 'networkx': networkx.__version__, "
                "'jsonschema': jsonschema.__version__}, sort_keys=True))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if version_check.returncode != 0:
        raise CapabilityEnvironmentError(
            "isolated causal environment version verification failed: " + version_check.stderr.strip()
        )
    versions = json.loads(version_check.stdout.strip())
    expected = {
        "dowhy": "0.14",
        "numpy": "2.4.6",
        "scipy": "1.15.3",
        "networkx": "3.6.1",
        "jsonschema": "4.26.0",
    }
    if versions != expected:
        raise CapabilityEnvironmentError(f"isolated environment version mismatch: {versions!r}")
    return {
        "status": "PASS",
        "isolated": True,
        **metadata,
        "versions": versions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--ticket", required=True)
    prepare_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    ticket = json.loads(Path(args.ticket).read_text(encoding="utf-8"))
    if not isinstance(ticket, Mapping):
        raise SystemExit("ticket must be a JSON object")
    result = prepare(ticket)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
