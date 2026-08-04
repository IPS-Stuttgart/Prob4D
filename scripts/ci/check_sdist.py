#!/usr/bin/env python3
"""Audit and exercise a built Prob4D source distribution."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REQUIRED_PATHS = frozenset(
    {
        ".github/CODEOWNERS",
        ".github/workflows/tests.yml",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "README.md",
        "SECURITY.md",
        "docs/provider-v2.md",
        "docs/repository-identity.md",
        "requirements/ci/minimum.txt",
        "scripts/ci/check_sdist.py",
        "tests/fixtures/prob4d_joint_observation_v1.json",
        "tests/test_github_action_pins.py",
        "tests/test_joint_observation_contract_fixture.py",
        "tests/test_project_identity.py",
        "tests/test_release_metadata.py",
    }
)
REPRESENTATIVE_TESTS = (
    "tests/test_sim3.py",
    "tests/test_provider_manifest.py",
    "tests/test_joint_observation_contract_fixture.py",
    "tests/test_project_identity.py",
    "tests/test_release_metadata.py",
    "tests/test_github_action_pins.py",
)


def _validated_members(archive: Path) -> tuple[str, tuple[tarfile.TarInfo, ...]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = tuple(handle.getmembers())
    if not members:
        raise RuntimeError("source distribution is empty")

    roots: set[str] = set()
    relative_paths: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise RuntimeError(f"unsafe source-distribution path: {member.name}")
        roots.add(path.parts[0])
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise RuntimeError(f"source distribution contains a non-regular member: {member.name}")
        if len(path.parts) > 1:
            relative_paths.add(PurePosixPath(*path.parts[1:]).as_posix())

    if len(roots) != 1:
        raise RuntimeError("source distribution must have exactly one root directory")
    missing = sorted(REQUIRED_PATHS - relative_paths)
    if missing:
        raise RuntimeError(f"source distribution omitted required assets: {missing}")
    return roots.pop(), members


def _extract_regular_files(archive: Path, destination: Path) -> str:
    root_name, members = _validated_members(archive)
    with tarfile.open(archive, "r:gz") as handle:
        for member in members:
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"unable to read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
    return root_name


def audit_sdist(archive: Path) -> None:
    archive = archive.resolve()
    if not archive.is_file():
        raise RuntimeError(f"source distribution does not exist: {archive}")
    with tempfile.TemporaryDirectory(prefix="prob4d-sdist-") as temporary:
        destination = Path(temporary)
        root_name = _extract_regular_files(archive, destination)
        source_root = destination / root_name
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root / "src")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *REPRESENTATIVE_TESTS],
            cwd=source_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        print(result.stdout, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args(argv)
    audit_sdist(arguments.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
