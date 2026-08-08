#!/usr/bin/env python3
"""Core-runtime gateway to the fixed isolated causal-policy worker."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compute_runner import ComputeError

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE / ".runtime-envs" / "causal-policy"
INTERPRETER = ENV_DIR / "bin" / "python"
WORKER = HERE / "causal_policy_worker.py"
MAX_WORKER_SECONDS = 180


def _validate_runtime_paths() -> None:
    if not WORKER.is_file():
        raise ComputeError("isolated causal worker is missing")
    if not INTERPRETER.is_file():
        raise ComputeError(
            "isolated causal runtime is not prepared; run capability_environment.py prepare before execution"
        )
    resolved_root = ENV_DIR.resolve()
    resolved_interpreter = INTERPRETER.resolve()
    if resolved_root not in resolved_interpreter.parents:
        raise ComputeError("isolated causal interpreter escaped the repository-controlled runtime root")


def causal_policy_evaluation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ComputeError("causal inputs must be an object")
    _validate_runtime_paths()
    try:
        payload = json.dumps(dict(inputs), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ComputeError(f"causal inputs are not finite JSON: {exc}") from exc
    try:
        completed = subprocess.run(
            [str(INTERPRETER), str(WORKER)],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            timeout=MAX_WORKER_SECONDS,
            cwd=str(HERE),
        )
    except subprocess.TimeoutExpired as exc:
        raise ComputeError(f"isolated causal worker exceeded {MAX_WORKER_SECONDS} seconds") from exc
    if completed.returncode != 0:
        error = completed.stderr.strip()
        if len(error) > 2000:
            error = error[-2000:]
        raise ComputeError(
            f"isolated causal worker failed with exit code {completed.returncode}: {error or 'no stderr'}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ComputeError("isolated causal worker returned invalid JSON") from exc
    if not isinstance(result, Mapping):
        raise ComputeError("isolated causal worker returned a non-object result")
    engine = result.get("engine")
    if not isinstance(engine, Mapping):
        raise ComputeError("isolated causal worker omitted engine receipt")
    if engine.get("runtime_isolation") != "fixed-venv" or engine.get("network_used") is not False:
        raise ComputeError("isolated causal worker receipt violated runtime policy")
    return dict(result)


OPERATIONS = {"causal_policy_evaluation": causal_policy_evaluation}
