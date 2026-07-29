#!/usr/bin/env python3
"""Apply the deterministic compute hardening bundle and remove bootstrap inputs."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path.cwd().resolve()
CHUNK_DIR = ROOT / "tools" / "hardening-chunks"
SELF = ROOT / "tools" / "apply_hardening_bundle.py"
EXPECTED_ENCODED_BYTES = 109448
EXPECTED_ENCODED_SHA256 = "73177ecf11eb79acdffeee3d4b4aeee1068745481f12a6835ad797b04d71b5c1"
EXPECTED_ARCHIVE_SHA256 = "dd0b1b33316bc498378c0bbf884183d27070d3af863503e5deeea79a0f51b009"


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise SystemExit(f"unsafe bundle path: {name!r}")
    return Path(*pure.parts)


def load_archive() -> bytes:
    chunks = sorted(CHUNK_DIR.glob("*.part"))
    if [path.name for path in chunks] != [f"{index:03d}.part" for index in range(7)]:
        raise SystemExit("hardening bundle must contain exactly chunks 000.part through 006.part")
    encoded = "".join(path.read_text(encoding="utf-8") for path in chunks)
    if len(encoded.encode("utf-8")) != EXPECTED_ENCODED_BYTES:
        raise SystemExit("hardening bundle encoded length mismatch")
    if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != EXPECTED_ENCODED_SHA256:
        raise SystemExit("hardening bundle encoded SHA-256 mismatch")
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("hardening archive SHA-256 mismatch")
    return payload


def apply_archive(payload: bytes) -> list[str]:
    deletions: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            safe_relative(member.name)
            if not member.isfile():
                raise SystemExit(f"bundle contains non-file entry: {member.name!r}")

        deletion_member = archive.getmember(".hardening-delete.json")
        deletion_file = archive.extractfile(deletion_member)
        if deletion_file is None:
            raise SystemExit("deletion manifest is unreadable")
        deletion_payload = json.loads(deletion_file.read().decode("utf-8"))
        deletions = [str(value) for value in deletion_payload.get("delete", [])]

        for member in members:
            if member.name == ".hardening-delete.json":
                continue
            relative = safe_relative(member.name)
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"bundle member is unreadable: {member.name!r}")
            target = (ROOT / relative).resolve()
            if ROOT not in target.parents:
                raise SystemExit(f"bundle path escapes repository: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            target.chmod(stat.S_IMODE(member.mode) or 0o644)
    return deletions


def remove_declared_files(deletions: list[str]) -> None:
    for name in deletions:
        relative = safe_relative(name)
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents:
            raise SystemExit(f"deletion path escapes repository: {name!r}")
        if target.is_dir():
            raise SystemExit(f"refusing to delete directory: {name!r}")
        target.unlink(missing_ok=True)


def validate_repository() -> None:
    subprocess.run(["python", "-m", "compileall", "-q", "compute-center", "tools"], check=True)
    for path in sorted(ROOT.rglob("*.json")):
        if ".git" in path.parts or "audit-artifacts" in path.parts:
            continue
        json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    payload = load_archive()
    deletions = apply_archive(payload)
    remove_declared_files(deletions)
    shutil.rmtree(CHUNK_DIR)
    (ROOT / "tools" / "hardening-bundle.b64").unlink(missing_ok=True)
    SELF.unlink(missing_ok=True)
    validate_repository()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
