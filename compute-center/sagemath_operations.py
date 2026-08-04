#!/usr/bin/env python3
"""Bounded SageMath operations executed in an exact-digest offline container."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compute_runner import ComputeError

HERE = Path(__file__).resolve().parent
RUNTIME = json.loads((HERE / "sagemath-runtime.json").read_text(encoding="utf-8"))
MODES = {"simplify", "solve", "differentiate", "integrate", "matrix_analysis", "number_theory"}
ALLOWED_NAMES = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "sqrt", "abs", "pi", "e"}
EXPRESSION_RE = re.compile(r"^[A-Za-z0-9+*/^()., \t-]{1,2000}$")
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
VARIABLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,15}$")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str, max_items: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not 1 <= len(value) <= max_items:
        raise ComputeError(f"{name} must contain 1 to {max_items} items")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ComputeError(f"{name} must be finite")
    return float(value)


def _variables(raw: Any) -> list[str]:
    rows = [] if raw in (None, []) else list(_sequence(raw, "inputs.variables", 20))
    result: list[str] = []
    for index, value in enumerate(rows):
        name = str(value)
        if not VARIABLE_RE.fullmatch(name) or name in ALLOWED_NAMES or name in result:
            raise ComputeError(f"invalid or duplicate variable at inputs.variables[{index}]")
        result.append(name)
    return result


def _expression(value: Any, variables: list[str]) -> str:
    expression = str(value or "").strip()
    if not EXPRESSION_RE.fullmatch(expression) or "**" in expression:
        raise ComputeError("expression contains forbidden characters or syntax")
    unknown = sorted(set(NAME_RE.findall(expression)) - set(variables) - ALLOWED_NAMES)
    if unknown:
        raise ComputeError(f"expression contains non-allowlisted names: {unknown}")
    return expression


def validate_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(inputs.get("mode") or "")
    if mode not in MODES:
        raise ComputeError(f"inputs.mode must be one of {sorted(MODES)}")
    payload: dict[str, Any] = {"mode": mode}

    if mode in {"simplify", "solve", "differentiate", "integrate"}:
        variables = _variables(inputs.get("variables"))
        payload["variables"] = variables
        payload["expression"] = _expression(inputs.get("expression"), variables)
        variable = str(inputs.get("variable") or (variables[0] if len(variables) == 1 else ""))
        if mode != "simplify" and variable not in variables:
            raise ComputeError("inputs.variable must be one of inputs.variables")
        if variable:
            payload["variable"] = variable
        if mode == "differentiate":
            payload["order"] = _integer(inputs.get("order", 1), "inputs.order", 1, 10)
        if mode == "integrate":
            has_lower = "lower" in inputs
            has_upper = "upper" in inputs
            if has_lower != has_upper:
                raise ComputeError("lower and upper must be supplied together")
            if has_lower:
                payload["lower"] = _finite(inputs["lower"], "inputs.lower")
                payload["upper"] = _finite(inputs["upper"], "inputs.upper")

    elif mode == "matrix_analysis":
        rows = _sequence(inputs.get("matrix"), "inputs.matrix", 20)
        matrix: list[list[float]] = []
        width: int | None = None
        for index, row in enumerate(rows):
            parsed = [_finite(item, f"inputs.matrix[{index}]") for item in _sequence(row, f"inputs.matrix[{index}]", 20)]
            width = width or len(parsed)
            if len(parsed) != width:
                raise ComputeError("matrix rows must have equal length")
            matrix.append(parsed)
        payload["matrix"] = matrix

    elif mode == "number_theory":
        action = str(inputs.get("action") or "")
        if action not in {"factor", "is_prime", "gcd", "lcm", "euler_phi"}:
            raise ComputeError("unsupported number_theory action")
        values = [_integer(item, f"inputs.values[{index}]", -(10**18), 10**18) for index, item in enumerate(_sequence(inputs.get("values"), "inputs.values", 20))]
        if action in {"factor", "is_prime", "euler_phi"} and len(values) != 1:
            raise ComputeError(f"{action} requires exactly one value")
        payload.update({"action": action, "values": values})

    return payload


SAGE_RUNNER = r'''from sage.all import *
import json

payload = json.load(open('/work/payload.json', encoding='utf-8'))
mode = payload['mode']
locals_map = {
    'sin': sin, 'cos': cos, 'tan': tan,
    'asin': asin, 'acos': acos, 'atan': atan,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'exp': exp, 'log': log, 'sqrt': sqrt, 'abs': abs,
    'pi': pi, 'e': e,
}
for name in payload.get('variables', []):
    locals_map[name] = var(name)

def expression():
    return sage_eval(payload['expression'].replace('^', '**'), locals=locals_map)

if mode == 'simplify':
    result = {'expression': str(expression().full_simplify())}
elif mode == 'solve':
    result = {'solutions': [str(item) for item in solve(expression() == 0, locals_map[payload['variable']])]}
elif mode == 'differentiate':
    result = {'derivative': str(diff(expression(), locals_map[payload['variable']], payload['order'])), 'order': payload['order']}
elif mode == 'integrate':
    variable = locals_map[payload['variable']]
    value = integral(expression(), variable, payload['lower'], payload['upper']) if 'lower' in payload else integral(expression(), variable)
    result = {'integral': str(value), 'definite': 'lower' in payload}
elif mode == 'matrix_analysis':
    matrix_value = matrix(SR, payload['matrix'])
    square = matrix_value.nrows() == matrix_value.ncols()
    result = {
        'rows': matrix_value.nrows(),
        'columns': matrix_value.ncols(),
        'rank': matrix_value.rank(),
        'determinant': str(matrix_value.det()) if square else None,
        'characteristic_polynomial': str(matrix_value.charpoly()) if square else None,
        'eigenvalues': [str(item) for item in matrix_value.eigenvalues()] if square else [],
    }
elif mode == 'number_theory':
    values = [Integer(item) for item in payload['values']]
    action = payload['action']
    if action == 'factor':
        value = str(factor(values[0]))
    elif action == 'is_prime':
        value = bool(values[0].is_prime())
    elif action == 'gcd':
        value = int(gcd(values))
    elif action == 'lcm':
        value = int(lcm(values))
    else:
        value = int(euler_phi(values[0]))
    result = {'action': action, 'value': value}
else:
    raise ValueError('unsupported mode')

print(json.dumps({'engine': 'SageMath', 'mode': mode, 'result': result}, ensure_ascii=False))
'''


def _run_sage(payload: Mapping[str, Any]) -> dict[str, Any]:
    if os.environ.get("SAGEMATH_FIXTURE_MODE") == "1":
        return {"engine": "SageMath-fixture", "mode": payload["mode"], "result": {"validated": True}}

    image = str(RUNTIME.get("image") or "")
    if not re.fullmatch(r"sagemath/sagemath@sha256:[0-9a-f]{64}", image):
        raise ComputeError("repository-pinned SageMath image is invalid")

    with tempfile.TemporaryDirectory(prefix="compute-sagemath-") as temp_dir:
        root = Path(temp_dir)
        (root / "payload.json").write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
        (root / "runner.py").write_text(SAGE_RUNNER, encoding="utf-8")
        for path in root.iterdir():
            path.chmod(0o644)
        command = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(RUNTIME["pids_limit"]),
            "--memory", f"{RUNTIME['memory_mb']}m", "--cpus", str(RUNTIME["cpus"]),
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            "-v", f"{root}:/work:ro", "--entrypoint", "sage", image,
            "-python", "/work/runner.py",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=int(RUNTIME["timeout_seconds"]), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ComputeError(f"SageMath runtime failed: {exc}") from exc
        if completed.returncode != 0:
            raise ComputeError(f"SageMath container returned {completed.returncode}: {completed.stderr[-1500:]}")
        try:
            output = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            raise ComputeError("SageMath returned invalid JSON") from exc
        output["runtime"] = {
            "image": image,
            "network_policy": "none",
            "arbitrary_code_allowed": False,
        }
        return output


def symbolic_mathematics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return _run_sage(validate_inputs(_mapping(inputs, "inputs")))


OPERATIONS = {"symbolic_mathematics": symbolic_mathematics}
