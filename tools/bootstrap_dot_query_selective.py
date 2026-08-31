#!/usr/bin/env python3
"""Materialize the reviewed DOT query-selective experiment capsule."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

BRANCH = "science/dot-query-selective-r11-r30-v1"
EXPECTED_PARENT = "8eaa70676a922c7e5e3a4c1fdc2234ea7ae1455f"
EXPECTED_POLICY_BLOB = "fa3d3944aa8f0b6e17e37212ea086687e92ee501"
EXPECTED_PAYLOAD_SHA256 = "71aadf3310939c27a5f3d3e924fc499666ab6e4eaae85e1d2981e4f6e68ca8ab"
CHUNK_COUNT = 4
CHUNK_TEMPLATE = "tools/dot_query_payload_part_{:02d}.txt"
BOOTSTRAP_PATH = Path("tools/bootstrap_dot_query_selective.py")
WORKFLOW_PATH = Path(".github/workflows/bootstrap-dot-query-selective-v1.yml")
POLICY_PATH = Path("tests/test_trusted_self_hosted_validation_policy.py")
POLICY_BASE_PATH = Path("tests/_trusted_self_hosted_validation_policy_base.py")
ALLOWED = {
    ".github/workflows/dot-rope-query-selective-heldout-v1.yml",
    "docs/dot-rope-query-selective-heldout-v1.md",
    "protocols/dot-rope-query-selective-heldout-v1.json",
    "scripts/science/run_dot_rope_query_selective_heldout.py",
    "tests/test_dot_rope_query_selective_heldout.py",
    "tests/test_dot_rope_query_selective_heldout_workflow.py",
    "tests/test_trusted_self_hosted_validation_policy.py",
}


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> int:
    repository = os.environ.get("GITHUB_REPOSITORY")
    ref_name = os.environ.get("GITHUB_REF_NAME")
    actor = os.environ.get("GITHUB_ACTOR")
    if repository != "IPS-Stuttgart/Prob4D" or ref_name != BRANCH or actor != "FlorianPfaff":
        raise SystemExit("bootstrap context is not the reviewed repository/branch/actor")
    if run("git", "rev-parse", "HEAD^") != EXPECTED_PARENT:
        raise SystemExit("bootstrap parent changed")
    if run("git", "hash-object", str(POLICY_PATH)) != EXPECTED_POLICY_BLOB:
        raise SystemExit("trusted self-hosted policy bytes changed")

    encoded = "".join(
        Path(CHUNK_TEMPLATE.format(index)).read_text(encoding="ascii").strip()
        for index in range(CHUNK_COUNT)
    )
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != EXPECTED_PAYLOAD_SHA256:
        raise SystemExit("embedded experiment payload changed")

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members if member.isfile()}
        if names != ALLOWED:
            raise SystemExit(f"experiment payload roster changed: {sorted(names)}")
        for member in members:
            path = PurePosixPath(member.name)
            if member.isdir():
                continue
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"unsafe payload member: {member.name}")
        POLICY_BASE_PATH.write_bytes(POLICY_PATH.read_bytes())
        for member in members:
            if member.isfile():
                target = Path(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise SystemExit(f"cannot read payload member: {member.name}")
                target.write_bytes(source.read())

    paths_to_remove = [BOOTSTRAP_PATH, WORKFLOW_PATH]
    paths_to_remove.extend(Path(CHUNK_TEMPLATE.format(index)) for index in range(CHUNK_COUNT))
    for path in paths_to_remove:
        path.unlink()

    subprocess.check_call([
        "python3", "-m", "compileall", "-q",
        "scripts/science/run_dot_rope_query_selective_heldout.py",
        "tests/test_dot_rope_query_selective_heldout.py",
        "tests/test_dot_rope_query_selective_heldout_workflow.py",
        "tests/test_trusted_self_hosted_validation_policy.py",
    ])
    subprocess.check_call(["git", "diff", "--check"])
    subprocess.check_call(["git", "config", "user.name", "github-actions[bot]"])
    subprocess.check_call([
        "git", "config", "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    ])
    subprocess.check_call(["git", "add", "-A"])
    subprocess.check_call([
        "git", "commit", "-m", "Freeze R11-R30 DOT query-selective experiment",
    ])
    subprocess.check_call(["git", "push", "origin", f"HEAD:{BRANCH}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
