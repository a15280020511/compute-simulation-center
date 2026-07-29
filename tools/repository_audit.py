#!/usr/bin/env python3
"""Deterministic repository-wide static audit.

The scanner reads every repository file that is safe to decode, parses Python and
JSON, inspects GitHub Actions and requirements, and writes machine-readable and
human-readable reports. It does not execute repository code or access the network.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "audit-artifacts",
    "compute-artifacts",
    "compute-validation-artifacts",
    "literature-artifacts",
}
TEXT_SUFFIXES = {
    ".py", ".json", ".yml", ".yaml", ".md", ".txt", ".toml", ".ini",
    ".cfg", ".sh", ".dockerfile", ".gitignore",
}
ENTRYPOINT_NAMES = {
    "compute_dispatch.py",
    "compute_ticket.py",
    "tool_registry.py",
    "maintenance_audit.py",
    "network_assurance.py",
    "literature_ticket.py",
    "literature_evidence.py",
}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
PINNED_ACTION_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}(?:\s*#.*)?$")
REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(?:==|~=|>=|<=|>|<|!=).*$")
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_., ]+))")


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    path: str
    line: int
    message: str


@dataclass
class FileRecord:
    path: str
    size_bytes: int
    sha256: str
    line_count: int | None
    kind: str


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in rel.parts):
            continue
        yield path


def _decode_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _kind(path: Path) -> str:
    name = path.name.lower()
    if path.suffix == ".py":
        return "python"
    if path.suffix == ".json":
        return "json"
    if path.suffix in {".yml", ".yaml"}:
        return "yaml"
    if path.suffix == ".md":
        return "markdown"
    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "dockerfile"
    if path.suffix == ".sh":
        return "shell"
    if "requirements" in name and path.suffix == ".txt":
        return "requirements"
    return "text" if path.suffix.lower() in TEXT_SUFFIXES or name.startswith(".") else "binary"


def _complexity(node: ast.AST) -> int:
    score = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.IfExp, ast.Match)):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += max(1, len(child.values) - 1)
        elif isinstance(child, ast.comprehension):
            score += 1
    return score


def _python_audit(rel: str, text: str) -> tuple[list[Finding], dict[str, Any], set[str], set[str]]:
    findings: list[Finding] = []
    metrics: dict[str, Any] = {"functions": 0, "classes": 0, "max_complexity": 0, "max_function_lines": 0}
    imports: set[str] = set()
    definitions: set[str] = set()
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        findings.append(Finding("critical", "PY-SYNTAX", rel, int(exc.lineno or 1), str(exc)))
        return findings, metrics, imports, definitions

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics["functions"] += 1
            definitions.add(node.name)
            complexity = _complexity(node)
            metrics["max_complexity"] = max(metrics["max_complexity"], complexity)
            end = int(getattr(node, "end_lineno", node.lineno))
            length = end - int(node.lineno) + 1
            metrics["max_function_lines"] = max(metrics["max_function_lines"], length)
            if complexity > 20:
                findings.append(Finding("medium", "PY-COMPLEXITY", rel, node.lineno, f"function {node.name!r} complexity={complexity}"))
            if length > 180:
                findings.append(Finding("medium", "PY-FUNCTION-SIZE", rel, node.lineno, f"function {node.name!r} spans {length} lines"))
        elif isinstance(node, ast.ClassDef):
            metrics["classes"] += 1
            definitions.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Pass):
            parent_context = "placeholder pass statement"
            findings.append(Finding("low", "PY-PASS", rel, node.lineno, parent_context))
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(Finding("high", "PY-BARE-EXCEPT", rel, node.lineno, "bare except hides termination and programming errors"))
            elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                findings.append(Finding("low", "PY-BROAD-EXCEPT", rel, node.lineno, f"broad handler for {node.type.id}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
                findings.append(Finding("critical", "PY-DYNAMIC-CODE", rel, node.lineno, f"use of {func.id}()"))
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                owner, name = func.value.id, func.attr
                if owner == "os" and name == "system":
                    findings.append(Finding("critical", "PY-OS-SYSTEM", rel, node.lineno, "os.system executes a shell"))
                if owner == "subprocess":
                    for keyword in node.keywords:
                        if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            findings.append(Finding("critical", "PY-SHELL-TRUE", rel, node.lineno, "subprocess call uses shell=True"))
                if owner in {"pickle", "dill"} and name in {"load", "loads"}:
                    findings.append(Finding("high", "PY-UNSAFE-DESERIALIZE", rel, node.lineno, f"{owner}.{name} can execute untrusted payloads"))
            if isinstance(func, ast.Name) and func.id == "open":
                for keyword in node.keywords:
                    if keyword.arg == "encoding" and isinstance(keyword.value, ast.Constant):
                        break
                else:
                    # Binary opens intentionally do not need encoding.
                    mode_is_binary = any(
                        isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "b" in arg.value
                        for arg in node.args[1:2]
                    )
                    if not mode_is_binary:
                        findings.append(Finding("low", "PY-OPEN-ENCODING", rel, node.lineno, "text open() has no explicit encoding"))

    if metrics["max_complexity"] > 35:
        findings.append(Finding("high", "PY-MODULE-COMPLEXITY", rel, 1, f"module contains function complexity={metrics['max_complexity']}"))
    return findings, metrics, imports, definitions


def _line_audit(rel: str, text: str, kind: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for index, line in enumerate(lines, 1):
        if line.rstrip(" \t") != line:
            findings.append(Finding("low", "TXT-TRAILING-WHITESPACE", rel, index, "trailing whitespace"))
        if "\t" in line and kind in {"python", "yaml", "json"}:
            findings.append(Finding("low", "TXT-TAB", rel, index, "tab character in structured source"))
        if len(line) > 180 and not re.search(r"https?://|sha256|BEGIN|END", line):
            findings.append(Finding("low", "TXT-LONG-LINE", rel, index, f"line length={len(line)}"))
        if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", line, re.IGNORECASE):
            findings.append(Finding("medium", "TXT-DEBT-MARKER", rel, index, line.strip()[:240]))
        if "a15280020511/test" in line:
            severity = "low" if any(part in rel.lower() for part in ("migration", "recovery", "readme", "governance-compatibility")) else "high"
            findings.append(Finding(severity, "ARCH-OLD-REPO-REF", rel, index, "reference to the legacy test repository"))
        if "test-api-center" in line:
            findings.append(Finding("high", "ARCH-OLD-IMAGE", rel, index, "reference to legacy API image"))
        if "temporary_governance_repository" in line:
            findings.append(Finding("medium", "ARCH-TEMP-GOVERNANCE", rel, index, "temporary governance field remains after physical split"))
        if "repository_dispatch" in line:
            findings.append(Finding("critical", "ARCH-CROSS-REPO-DISPATCH", rel, index, "repository_dispatch violates center isolation"))
        if "pull_request_target" in line and kind == "yaml":
            findings.append(Finding("high", "GHA-PR-TARGET", rel, index, "pull_request_target expands trust boundary"))
        if re.search(r"\bpermissions:\s*write-all\b", line):
            findings.append(Finding("critical", "GHA-WRITE-ALL", rel, index, "workflow grants write-all"))
        if kind == "yaml" and re.match(r"\s*uses:\s*", line):
            action = line.split("uses:", 1)[1].strip()
            if action.startswith("./") or action.startswith("docker://"):
                continue
            if not PINNED_ACTION_RE.match(action):
                findings.append(Finding("high", "GHA-UNPINNED-ACTION", rel, index, f"action is not pinned to a 40-character commit SHA: {action}"))
        if re.search(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^$<{][^'\"]{8,}['\"]", line):
            findings.append(Finding("critical", "SEC-POSSIBLE-SECRET", rel, index, "possible hard-coded credential"))
    return findings


def _requirements_audit(records: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    package_specs: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for rel, text in records.items():
        for index, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-r "):
                continue
            match = REQ_RE.match(line)
            if not match:
                findings.append(Finding("medium", "REQ-UNPINNED", rel, index, f"dependency is not constrained: {line}"))
                continue
            package_specs[match.group(1).lower().replace("_", "-")].append((rel, index, line))
    for package, specs in package_specs.items():
        unique = {spec for _, _, spec in specs}
        if len(unique) > 1:
            first_rel, first_line, _ = specs[0]
            findings.append(Finding("medium", "REQ-CONFLICTING-SPECS", first_rel, first_line, f"{package} has multiple constraints: {sorted(unique)}"))
    return findings


def _workflow_audit(root: Path, yaml_records: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    names: dict[str, list[str]] = defaultdict(list)
    for rel, text in yaml_records.items():
        if not rel.startswith(".github/workflows/"):
            continue
        name_match = re.search(r"(?m)^name:\s*(.+?)\s*$", text)
        if name_match:
            names[name_match.group(1).strip(" '\"")].append(rel)
        if "workflow_dispatch:" not in text and "pull_request:" not in text and "push:" not in text and "issues:" not in text and "schedule:" not in text:
            findings.append(Finding("high", "GHA-NO-TRIGGER", rel, 1, "workflow has no recognized trigger"))
        if "environment: compute-numeric-offline" in text and "unshare --net" not in text and "network_assurance.py" not in text:
            findings.append(Finding("high", "ARCH-OFFLINE-NOT-ENFORCED", rel, 1, "numeric environment is declared without OS network isolation evidence"))
        if "environment: compute-literature-evidence" in text and "compute_dispatch" in text:
            findings.append(Finding("critical", "ARCH-LITERATURE-DISPATCH", rel, 1, "literature workflow references numeric dispatcher"))
    for name, paths in names.items():
        if len(paths) > 1:
            findings.append(Finding("medium", "GHA-DUPLICATE-NAME", paths[0], 1, f"workflow name {name!r} is reused by {paths}"))
    return findings


def _json_audit(json_records: dict[str, str]) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    schema_ids: dict[str, list[str]] = defaultdict(list)
    parsed = 0
    for rel, text in json_records.items():
        try:
            value = json.loads(text)
            parsed += 1
        except json.JSONDecodeError as exc:
            findings.append(Finding("critical", "JSON-PARSE", rel, exc.lineno, exc.msg))
            continue
        if isinstance(value, dict):
            schema_id = value.get("$id") or value.get("schema_version")
            if isinstance(schema_id, str) and schema_id:
                schema_ids[schema_id].append(rel)
    for schema_id, paths in schema_ids.items():
        if len(paths) > 1 and schema_id.startswith("http"):
            findings.append(Finding("high", "JSON-DUPLICATE-ID", paths[0], 1, f"schema id {schema_id!r} reused by {paths}"))
    return findings, {"parsed_json_files": parsed, "schema_id_count": len(schema_ids)}


def audit(root: Path) -> dict[str, Any]:
    findings: list[Finding] = []
    files: list[FileRecord] = []
    text_records: dict[str, str] = {}
    python_metrics: dict[str, dict[str, Any]] = {}
    python_imports: dict[str, set[str]] = {}
    python_defs: dict[str, set[str]] = {}
    hashes: dict[str, list[str]] = defaultdict(list)

    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes[digest].append(rel)
        kind = _kind(path)
        text = _decode_text(path)
        line_count = None if text is None else len(text.splitlines())
        files.append(FileRecord(rel, len(data), digest, line_count, kind))
        if len(data) > 2_000_000:
            findings.append(Finding("medium", "FILE-LARGE", rel, 1, f"repository file size={len(data)} bytes"))
        if text is None:
            continue
        text_records[rel] = text
        findings.extend(_line_audit(rel, text, kind))
        if kind == "python":
            py_findings, metrics, imports, defs = _python_audit(rel, text)
            findings.extend(py_findings)
            python_metrics[rel] = metrics
            python_imports[rel] = imports
            python_defs[rel] = defs

    for digest, paths in hashes.items():
        nontrivial = [path for path in paths if not path.endswith(("__init__.py", ".gitkeep"))]
        if len(nontrivial) > 1:
            findings.append(Finding("low", "FILE-DUPLICATE", nontrivial[0], 1, f"identical content shared by {nontrivial}; sha256={digest}"))

    requirements = {rel: text for rel, text in text_records.items() if "requirements" in Path(rel).name and rel.endswith(".txt")}
    yaml_records = {rel: text for rel, text in text_records.items() if rel.endswith((".yml", ".yaml"))}
    json_records = {rel: text for rel, text in text_records.items() if rel.endswith(".json")}
    findings.extend(_requirements_audit(requirements))
    findings.extend(_workflow_audit(root, yaml_records))
    json_findings, json_metrics = _json_audit(json_records)
    findings.extend(json_findings)

    module_names = {Path(rel).stem for rel in python_metrics}
    imported_names = set().union(*python_imports.values()) if python_imports else set()
    orphan_candidates = []
    for rel in sorted(python_metrics):
        path = Path(rel)
        stem = path.stem
        if "tests" in path.parts or path.name in ENTRYPOINT_NAMES or path.name == "__init__.py" or path.parts[0] == "tools":
            continue
        if stem not in imported_names and stem in module_names:
            orphan_candidates.append(rel)
            findings.append(Finding("info", "PY-ORPHAN-CANDIDATE", rel, 1, "module is not imported by another Python file; verify workflow/CLI use before removal"))

    findings.sort(key=lambda item: (SEVERITY_ORDER[item.severity], item.path, item.line, item.rule))
    counts = Counter(item.severity for item in findings)
    return {
        "schema_version": "repository-audit-v1",
        "root": str(root),
        "file_count": len(files),
        "text_file_count": len(text_records),
        "python_file_count": len(python_metrics),
        "total_lines": sum(record.line_count or 0 for record in files),
        "findings": [asdict(item) for item in findings],
        "finding_counts": {key: counts.get(key, 0) for key in SEVERITY_ORDER},
        "files": [asdict(record) for record in files],
        "python_metrics": python_metrics,
        "orphan_candidates": orphan_candidates,
        "json_metrics": json_metrics,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full Repository Audit",
        "",
        f"- Files: `{report['file_count']}`",
        f"- Text files: `{report['text_file_count']}`",
        f"- Python files: `{report['python_file_count']}`",
        f"- Lines inspected: `{report['total_lines']}`",
        f"- Critical: `{report['finding_counts']['critical']}`",
        f"- High: `{report['finding_counts']['high']}`",
        f"- Medium: `{report['finding_counts']['medium']}`",
        f"- Low: `{report['finding_counts']['low']}`",
        f"- Info: `{report['finding_counts']['info']}`",
        "",
        "## Findings",
        "",
        "| Severity | Rule | File | Line | Finding |",
        "|---|---|---|---:|---|",
    ]
    for finding in report["findings"]:
        message = str(finding["message"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {finding['severity']} | `{finding['rule']}` | `{finding['path']}` | {finding['line']} | {message} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="audit-artifacts")
    parser.add_argument("--fail-on", choices=("none", "critical", "high"), default="none")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = audit(root)
    (output / "repository-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "repository-audit.md").write_text(_markdown(report), encoding="utf-8")
    summary = {
        "status": "PASS" if report["finding_counts"]["critical"] == 0 and report["finding_counts"]["high"] == 0 else "FINDINGS",
        "finding_counts": report["finding_counts"],
        "file_count": report["file_count"],
        "total_lines": report["total_lines"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.fail_on == "critical" and report["finding_counts"]["critical"]:
        return 1
    if args.fail_on == "high" and (report["finding_counts"]["critical"] or report["finding_counts"]["high"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
