#!/usr/bin/env python3
"""Audit and exercise a built Prob4D source distribution."""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
import venv
from pathlib import Path, PurePosixPath

REQUIRED_PATHS = frozenset(
    {
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "docs/distribution-boundaries.md",
        "docs/public-api.md",
        "docs/releases/0.4.0.md",
        "protocols/cycle-guard-normalization-v1.json",
        "pyproject.toml",
        "src/prob4d/__init__.py",
        "src/prob4d/_version.py",
        "src/prob4d/api/__init__.py",
        "src/prob4d/api/v1.py",
        "src/prob4d/contract_data/observation_belief_v1/manifest.json",
        "src/prob4d/contract_data/observation_belief_v1/schema.json",
        "src/prob4d/py.typed",
    }
)
FORBIDDEN_PREFIXES = (
    ".github/",
    "evidence/",
    "environments/",
    "requirements/",
    "scripts/",
    "tests/",
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

    forbidden = sorted(
        path
        for path in relative_paths
        if any(
            path == prefix.rstrip("/") or path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES
        )
    )
    if forbidden:
        raise RuntimeError(f"source distribution contains repository-only assets: {forbidden[:20]}")
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


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _venv_executable(environment: Path, name: str) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / f"{name}.exe"
    return environment / "bin" / name


def _run(command: list[str | Path], *, cwd: Path | None = None) -> None:
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    if result.stdout:
        print(result.stdout, end="")


def _smoke_installed_archive(archive: Path, destination: Path) -> None:
    environment = destination / "installed"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = _venv_python(environment)
    _run([python, "-m", "pip", "install", "--no-deps", archive])

    smoke = """
from importlib import resources
from importlib.metadata import version

import prob4d
import prob4d.api.v1 as api_v1
import prob4d.provider_v1 as provider_v1
import prob4d.provider_v2 as provider_v2

installed = version("prob4d")
assert prob4d.__version__ == installed
assert api_v1.__version__ == installed
assert api_v1.API_VERSION == 1
assert api_v1.PROVIDER_API_VERSION == provider_v1.PROVIDER_API_VERSION == 1
assert provider_v2.PROVIDER_API_VERSION == 2
assert callable(api_v1.export_calibrated_observation_belief)
assert callable(provider_v2.load_claim_bearing_observation_belief)
assert resources.files("prob4d").joinpath("py.typed").is_file()
assert resources.files("prob4d").joinpath(
    "contract_data/observation_belief_v1/manifest.json"
).is_file()
"""
    _run([python, "-c", smoke])

    cli = _venv_executable(environment, "prob4d")
    _run([cli, "--help"])
    _run([cli, "project", "identity", "--compact"])


def audit_sdist(archive: Path) -> None:
    archive = archive.resolve()
    if not archive.is_file():
        raise RuntimeError(f"source distribution does not exist: {archive}")
    with tempfile.TemporaryDirectory(prefix="prob4d-sdist-") as temporary:
        destination = Path(temporary)
        _extract_regular_files(archive, destination)
        _smoke_installed_archive(archive, destination)
    print("source distribution boundary and installed smoke checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args(argv)
    audit_sdist(arguments.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
