#!/usr/bin/env python3
"""Repository-wide GitHub Actions diagnostic sweeper.

Uses only the Python standard library. It queries recent workflow runs, collects
job and step metadata, downloads full logs for non-successful runs, redacts
likely credentials, classifies failures, and writes a SHA-256 manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "workflow-diagnostics-v2"
API_VERSION = "2022-11-28"
MAX_PAGES = 10
MAX_LOG_ZIP_BYTES = 100 * 1024 * 1024
MAX_REDACTED_FILE_BYTES = 8 * 1024 * 1024
MAX_KEY_LINES_PER_RUN = 300
FAILURE_CONCLUSIONS = {
    "failure", "cancelled", "timed_out", "action_required", "stale", "startup_failure"
}
SECRET_NAME_RE = re.compile(
    r"(?i)(authorization|cookie|set-cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"bearer|client[_-]?secret|private[_-]?key|password|passwd|sendkey|sckey|secret|token)"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|cookie|set-cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|private[_-]?key|password|passwd|sendkey|sckey|secret|token)"
    r"\b\s*[:=]\s*([^\s,;]+)"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key|secret|sendkey|sckey)=)[^&#\s]+"
)
GITHUB_TOKEN_RE = re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{12,}\b")
LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{48,}\b")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")
ERROR_PATTERNS: Sequence[tuple[str, float, Sequence[str]]] = (
    ("secret_or_auth", 0.98, ("secret is missing", "missing or unavailable", "bad credentials", "unauthorized", "forbidden", "permission denied", "resource not accessible by integration", "authentication failed", "http 401", "http 403")),
    ("rate_limit_or_quota", 0.96, ("rate limit", "secondary rate limit", "too many requests", "quota exceeded", "insufficient quota", "http 429", "billing limit")),
    ("timeout_or_cancellation", 0.94, ("timed out", "timeout", "operation was canceled", "operation cancelled", "cancelled", "canceled", "deadline exceeded")),
    ("network_dns_tls", 0.92, ("name or service not known", "temporary failure in name resolution", "connection reset", "connection refused", "connection timed out", "tls", "sslerror", "certificate verify failed", "network is unreachable", "could not resolve host", "remote end closed")),
    ("dependency_install", 0.90, ("no matching distribution found", "resolutionimpossible", "could not build wheels", "failed building wheel", "npm err!", "package not found", "module not found", "modulenotfounderror", "dependency conflict")),
    ("schema_or_input", 0.90, ("schema validation", "validationerror", "invalid json", "jsondecodeerror", "required property", "additional properties are not allowed", "invalid ticket", "malformed input")),
    ("artifact_or_attestation", 0.90, ("artifact", "attestation", "digest mismatch", "sha-256 mismatch", "manifest mismatch", "no files were found with the provided path", "failed to upload artifact", "failed to download artifact")),
    ("provider_or_model", 0.88, ("provider error", "model not found", "model unavailable", "context length", "maximum context", "content filter", "openrouter", "anthropic", "openai", "deepseek")),
    ("test_or_assertion", 0.86, ("assertionerror", "tests failed", "failed test", "pytest", "assert ", "test failure")),
    ("resource_exhaustion", 0.86, ("no space left on device", "out of memory", "memoryerror", "killed", "exit code 137", "disk quota exceeded")),
    ("syntax_or_runtime", 0.82, ("syntaxerror", "typeerror", "valueerror", "keyerror", "indexerror", "attributeerror", "runtimeerror", "traceback (most recent call last)")),
)
KEY_LINE_RE = re.compile(
    r"(?i)(::error|##\[error\]|error:|exception|traceback|failed|failure|fatal|timed out|"
    r"timeout|unauthorized|forbidden|rate limit|quota|assertion|warning:|##\[warning\])"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = BEARER_RE.sub("Bearer [REDACTED]", text)
    text = SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = URL_SECRET_RE.sub(lambda m: f"{m.group(1)}[REDACTED]", text)
    text = GITHUB_TOKEN_RE.sub("[REDACTED_GITHUB_TOKEN]", text)
    output: list[str] = []
    for line in text.splitlines():
        if SECRET_NAME_RE.search(line):
            line = LONG_SECRET_RE.sub("[REDACTED]", line)
        output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


@dataclass
class GitHubClient:
    repository: str
    token: str
    api_root: str = "https://api.github.com"

    def _request(self, path: str) -> bytes:
        request = urllib.request.Request(
            self.api_root + path,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "repository-workflow-diagnostics",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(MAX_LOG_ZIP_BYTES + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API GET {path} failed: HTTP {exc.code}: {redact(body)}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub API GET {path} network failure: {exc}") from exc
        if len(data) > MAX_LOG_ZIP_BYTES:
            raise RuntimeError(f"GitHub API response exceeded {MAX_LOG_ZIP_BYTES} bytes for {path}")
        return data

    def json(self, path: str) -> Any:
        raw = self._request(path)
        return json.loads(raw.decode("utf-8")) if raw else None

    def bytes(self, path: str) -> bytes:
        return self._request(path)

    def recent_runs(self, cutoff: dt.datetime, max_runs: int) -> list[Mapping[str, Any]]:
        selected: list[Mapping[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            payload = self.json(f"/repos/{self.repository}/actions/runs?per_page=100&page={page}&exclude_pull_requests=false")
            rows = payload.get("workflow_runs", []) if isinstance(payload, Mapping) else []
            if not isinstance(rows, list) or not rows:
                break
            oldest: dt.datetime | None = None
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                created_at = parse_utc(str(row.get("created_at") or row.get("run_started_at") or utc_now()))
                oldest = created_at if oldest is None or created_at < oldest else oldest
                if created_at >= cutoff:
                    selected.append(row)
                    if len(selected) >= max_runs:
                        return selected
            if oldest is not None and oldest < cutoff:
                break
        return selected

    def jobs(self, run_id: int) -> list[Mapping[str, Any]]:
        output: list[Mapping[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            payload = self.json(f"/repos/{self.repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100&page={page}")
            rows = payload.get("jobs", []) if isinstance(payload, Mapping) else []
            if not isinstance(rows, list):
                break
            output.extend(row for row in rows if isinstance(row, Mapping))
            if len(rows) < 100:
                break
        return output


def normalized_line(line: str) -> str:
    line = TIMESTAMP_PREFIX_RE.sub("", line)
    line = re.sub(r"\b\d{5,}\b", "<N>", line)
    line = re.sub(r"[0-9a-f]{12,64}", "<HEX>", line, flags=re.I)
    return re.sub(r"\s+", " ", line).strip().lower()[:500]


def retry_guidance(category: str) -> Mapping[str, Any]:
    if category in {"network_dns_tls", "rate_limit_or_quota", "timeout_or_cancellation", "provider_or_model"}:
        return {"retryable": True, "strategy": "bounded_retry_with_backoff", "max_attempts": 2}
    if category in {"secret_or_auth", "schema_or_input", "dependency_install", "artifact_or_attestation"}:
        return {"retryable": False, "strategy": "fix_configuration_or_input_before_retry", "max_attempts": 0}
    return {"retryable": False, "strategy": "manual_root_cause_review", "max_attempts": 0}


def classify_failure(key_lines: Sequence[str], run: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    text = "\n".join(key_lines).lower()
    candidates: list[dict[str, Any]] = []
    for category, confidence, needles in ERROR_PATTERNS:
        hits = [needle for needle in needles if needle in text]
        if hits:
            candidates.append({"category": category, "confidence": confidence, "signals": hits[:8]})
    conclusion = str(run.get("conclusion") or "unknown")
    if conclusion in {"cancelled", "timed_out", "startup_failure"}:
        candidates.insert(0, {"category": "timeout_or_cancellation", "confidence": 0.99, "signals": [conclusion]})
    if not candidates:
        candidates.append({"category": "unknown", "confidence": 0.25, "signals": ["no known signature matched"]})
    candidates.sort(key=lambda row: float(row["confidence"]), reverse=True)
    failed_steps: list[dict[str, Any]] = []
    for job in jobs:
        for step in job.get("steps", []) if isinstance(job.get("steps"), list) else []:
            if isinstance(step, Mapping) and str(step.get("conclusion") or "") in FAILURE_CONCLUSIONS:
                failed_steps.append({
                    "job_id": job.get("id"), "job_name": job.get("name"),
                    "step_number": step.get("number"), "step_name": step.get("name"),
                    "conclusion": step.get("conclusion"),
                })
    material = "\n".join([str(candidates[0]["category"])] + [normalized_line(line) for line in key_lines[:20]])
    return {
        "primary_category": candidates[0]["category"],
        "confidence": candidates[0]["confidence"],
        "candidates": candidates[:5],
        "failed_steps": failed_steps,
        "failure_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "retry_guidance": retry_guidance(str(candidates[0]["category"])),
    }


def safe_extract_logs(zip_bytes: bytes, target: Path) -> tuple[list[Path], list[str]]:
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "workflow-logs.zip"
    zip_path.write_bytes(zip_bytes)
    extracted: list[Path] = []
    notes: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            filename = Path(info.filename).name
            if not filename:
                continue
            raw = archive.read(info)[: MAX_REDACTED_FILE_BYTES * 4]
            encoded = redact(raw.decode("utf-8", errors="replace")).encode("utf-8")
            if len(encoded) > MAX_REDACTED_FILE_BYTES:
                encoded = encoded[:MAX_REDACTED_FILE_BYTES] + b"\n[TRUNCATED]\n"
                notes.append(f"redacted log {filename} truncated")
            out = target / f"{len(extracted):03d}-{filename}"
            out.write_bytes(encoded)
            extracted.append(out)
    zip_path.unlink(missing_ok=True)
    return extracted, notes


def key_lines_from_logs(paths: Sequence[Path]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not KEY_LINE_RE.search(line):
                continue
            normalized = normalized_line(line)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rows.append(line[:2000])
            if len(rows) >= MAX_KEY_LINES_PER_RUN:
                return rows
    return rows


def duration_seconds(started: Any, completed: Any) -> float | None:
    if not started or not completed:
        return None
    try:
        return round((parse_utc(str(completed)) - parse_utc(str(started))).total_seconds(), 3)
    except (ValueError, TypeError):
        return None


def compact_run(run: Mapping[str, Any]) -> dict[str, Any]:
    actor = run.get("actor") if isinstance(run.get("actor"), Mapping) else {}
    trigger = run.get("triggering_actor") if isinstance(run.get("triggering_actor"), Mapping) else {}
    return {
        "id": run.get("id"), "name": run.get("name"), "display_title": run.get("display_title"),
        "workflow_id": run.get("workflow_id"), "path": run.get("path"), "event": run.get("event"),
        "status": run.get("status"), "conclusion": run.get("conclusion"),
        "run_number": run.get("run_number"), "run_attempt": run.get("run_attempt"),
        "head_branch": run.get("head_branch"), "head_sha": run.get("head_sha"),
        "created_at": run.get("created_at"), "run_started_at": run.get("run_started_at"),
        "updated_at": run.get("updated_at"),
        "duration_seconds": duration_seconds(run.get("run_started_at"), run.get("updated_at")),
        "html_url": run.get("html_url"), "actor": actor.get("login"),
        "triggering_actor": trigger.get("login"),
    }


def compact_job(job: Mapping[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for step in job.get("steps", []) if isinstance(job.get("steps"), list) else []:
        if isinstance(step, Mapping):
            steps.append({
                "number": step.get("number"), "name": step.get("name"),
                "status": step.get("status"), "conclusion": step.get("conclusion"),
                "started_at": step.get("started_at"), "completed_at": step.get("completed_at"),
                "duration_seconds": duration_seconds(step.get("started_at"), step.get("completed_at")),
            })
    return {
        "id": job.get("id"), "name": job.get("name"), "status": job.get("status"),
        "conclusion": job.get("conclusion"), "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "duration_seconds": duration_seconds(job.get("started_at"), job.get("completed_at")),
        "runner_name": job.get("runner_name"), "runner_group_name": job.get("runner_group_name"),
        "html_url": job.get("html_url"), "steps": steps,
    }


def build_manifest(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    json_dump(root / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "created_at": utc_now(),
        "file_count": len(files), "files": files,
        "security": {"secret_values_included": False, "logs_redacted": True},
    })


def diagnose_run(client: GitHubClient, run: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    run_id = int(run.get("id") or 0)
    run_dir = root / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    compact = compact_run(run)
    jobs = [compact_job(job) for job in client.jobs(run_id)]
    json_dump(run_dir / "run.json", compact)
    jsonl_dump(run_dir / "jobs.jsonl", jobs)
    key_lines: list[str] = []
    notes: list[str] = []
    log_paths: list[Path] = []
    conclusion = str(run.get("conclusion") or "unknown")
    if conclusion in FAILURE_CONCLUSIONS:
        try:
            log_paths, notes = safe_extract_logs(
                client.bytes(f"/repos/{client.repository}/actions/runs/{run_id}/logs"),
                run_dir / "redacted-logs",
            )
            key_lines = key_lines_from_logs(log_paths)
        except Exception as exc:
            notes.append(f"log download failed: {redact(str(exc))}")
        jsonl_dump(run_dir / "key-lines.jsonl", ({"line": line} for line in key_lines))
        failure = dict(classify_failure(key_lines, run, jobs))
        failure.update({
            "schema_version": SCHEMA_VERSION, "run_id": run_id,
            "workflow": compact.get("name"), "conclusion": conclusion,
            "notes": notes, "key_line_count": len(key_lines),
            "redacted_log_file_count": len(log_paths),
        })
        json_dump(run_dir / "failure.json", failure)
    return {
        **compact, "job_count": len(jobs),
        "failed_job_count": sum(1 for job in jobs if job.get("conclusion") in FAILURE_CONCLUSIONS),
        "diagnostic_path": run_dir.relative_to(root).as_posix(),
        "logs_collected": bool(log_paths), "notes": notes,
    }


def markdown_summary(repository: str, center: str, cutoff: str, rows: Sequence[Mapping[str, Any]], root: Path) -> str:
    failures = [row for row in rows if str(row.get("conclusion")) in FAILURE_CONCLUSIONS]
    categories: dict[str, int] = {}
    for row in failures:
        path = root / str(row["diagnostic_path"]) / "failure.json"
        if path.exists():
            category = str(json.loads(path.read_text(encoding="utf-8")).get("primary_category") or "unknown")
            categories[category] = categories.get(category, 0) + 1
    lines = [
        "# Workflow diagnostic sweep", "", f"- Repository: `{repository}`", f"- Center: `{center}`",
        f"- Cutoff: `{cutoff}`", f"- Runs inspected: **{len(rows)}**",
        f"- Non-success runs: **{len(failures)}**", "",
    ]
    if categories:
        lines.extend(["## Failure categories", ""])
        lines.extend(f"- `{category}`: {count}" for category, count in sorted(categories.items(), key=lambda x: (-x[1], x[0])))
        lines.append("")
    if failures:
        lines.extend(["## Failed or interrupted runs", "", "| Run | Workflow | Conclusion | SHA |", "|---:|---|---|---|"])
        for row in failures[:50]:
            lines.append(f"| [{row.get('id')}]({row.get('html_url')}) | {str(row.get('name') or '').replace('|', '\\|')} | `{row.get('conclusion')}` | `{str(row.get('head_sha') or '')[:12]}` |")
        lines.append("")
    lines.extend([
        "## Reading order", "",
        "`summary.md` → `runs/<run_id>/failure.json` → `key-lines.jsonl` → `jobs.jsonl` → `redacted-logs/` → `manifest.json`",
        "", "Secrets are redacted and raw environment variables are not collected.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--center", default=os.getenv("DIAGNOSTIC_CENTER", "unknown"))
    parser.add_argument("--output-dir", default="diagnostic-bundle")
    parser.add_argument("--lookback-hours", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--exclude-workflow-path", action="append", default=[])
    args = parser.parse_args()
    if not args.repository or "/" not in args.repository or not args.token:
        print("::error::valid repository and GITHUB_TOKEN are required", file=sys.stderr)
        return 2
    if not (0.1 <= args.lookback_hours <= 168) or not (1 <= args.max_runs <= 1000):
        print("::error::invalid sweep bounds", file=sys.stderr)
        return 2
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cutoff_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.lookback_hours)
    cutoff = cutoff_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    client = GitHubClient(args.repository, args.token)
    started = time.monotonic()
    try:
        runs = client.recent_runs(cutoff_dt, args.max_runs)
        excluded = set(args.exclude_workflow_path)
        rows = [diagnose_run(client, run, root) for run in runs if str(run.get("path") or "") not in excluded]
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        json_dump(root / "diagnostic-index.json", {
            "schema_version": SCHEMA_VERSION, "created_at": utc_now(),
            "repository": args.repository, "center": args.center, "cutoff": cutoff,
            "lookback_hours": args.lookback_hours, "max_runs": args.max_runs,
            "run_count": len(rows), "runs": rows,
            "collector": {
                "duration_seconds": round(time.monotonic() - started, 3),
                "secret_values_included": False, "raw_environment_collected": False,
            },
        })
        summary = markdown_summary(args.repository, args.center, cutoff, rows, root)
        (root / "summary.md").write_text(summary, encoding="utf-8")
        build_manifest(root)
        if os.getenv("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
                handle.write(summary)
        return 0
    except Exception as exc:
        message = redact(str(exc))
        json_dump(root / "collector-error.json", {
            "schema_version": SCHEMA_VERSION, "created_at": utc_now(),
            "repository": args.repository, "center": args.center,
            "error_type": type(exc).__name__, "message": message,
            "security": {"secret_values_included": False},
        })
        build_manifest(root)
        print(f"::error::diagnostic collector failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
