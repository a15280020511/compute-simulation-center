#!/usr/bin/env python3
"""Apply the deterministic compute hardening bundle and remove this bootstrap."""
from __future__ import annotations

import base64
import io
import json
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path.cwd().resolve()
BUNDLE = ROOT / "tools" / "hardening-bundle.b64"
SELF = ROOT / "tools" / "apply_hardening_bundle.py"


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise SystemExit(f"unsafe bundle path: {name!r}")
    return Path(*pure.parts)


def main() -> int:
    encoded = "".join(BUNDLE.read_text(encoding="utf-8").split())
    payload = base64.b64decode(encoded, validate=True)
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

    for name in deletions:
        relative = safe_relative(name)
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents:
            raise SystemExit(f"deletion path escapes repository: {name!r}")
        if target.is_dir():
            raise SystemExit(f"refusing to delete directory: {name!r}")
        target.unlink(missing_ok=True)

    BUNDLE.unlink(missing_ok=True)
    SELF.unlink(missing_ok=True)

    subprocess.run(["python", "-m", "compileall", "-q", "compute-center", "tools"], check=True)
    for path in sorted(ROOT.rglob("*.json")):
        if ".git" in path.parts or "audit-artifacts" in path.parts:
            continue
        json.loads(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
