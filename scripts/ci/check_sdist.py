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
        "docs/public-api-manifest.md",
        "docs/public-api.md",
        "docs/releases/0.5.0.md",
        "protocols/cycle-guard-normalization-v1.json",
        "pyproject.toml",
        "src/prob4d/__init__.py",
        "src/prob4d/__init__.pyi",
        "src/prob4d/_provider_export_core.py",
        "src/prob4d/_version.py",
        "src/prob4d/api/__init__.py",
        "src/prob4d/api/v2.py",
        "src/prob4d/provider_v1.py",
        "src/prob4d/provider_v2.py",
        "src/prob4d/contract_data/observation_belief_v1/manifest.json",
        "src/prob4d/contract_data/observation_belief_v1/schema.json",
        "src/prob4d/public_api_manifest.py",
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
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in FORBIDDEN_PREFIXES
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
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    _run([python, "-m", "pip", "install", archive])
    _run([python, "-m", "pip", "check"])

    smoke = """
from importlib import resources, util
from importlib.metadata import version
import sys

import prob4d

assert prob4d.__all__ == ["__version__"]
assert "prob4d.sim3" not in sys.modules
assert "Sim3" not in prob4d.__dict__
assert not hasattr(prob4d, "_LAZY_EXPORTS")
assert util.find_spec("prob4d.api.v1") is None
assert util.find_spec("prob4d.legacy_cli") is None
assert util.find_spec("prob4d.causal_stream_cli") is None

import prob4d.provider_v1 as provider_v1_artifacts
assert provider_v1_artifacts.PROVIDER_API_VERSION == 1
assert not hasattr(provider_v1_artifacts, "export_observation_belief")
assert not hasattr(provider_v1_artifacts, "export_calibrated_observation_belief")

import prob4d.api.v2 as api_v2
from prob4d.public_api_manifest import build_public_api_manifest

installed = version("prob4d")
assert installed == "0.5.0"
assert prob4d.__version__ == installed
assert api_v2.API_VERSION == 2
assert api_v2.PROVIDER_API_VERSION == 2
assert callable(api_v2.load_claim_bearing_observation_belief)
assert "Sim3" not in prob4d.__dict__

manifest = build_public_api_manifest()
assert manifest["package"]["version"] == installed
assert manifest["schema_version"] == 2
assert manifest["surfaces"]["package_root"]["loading"] == (
    "minimal-version-root-v1"
)
assert manifest["surfaces"]["package_root"]["exports"] == ["__version__"]
assert set(manifest["surfaces"]) == {"package_root", "api_v2"}
assert manifest["surfaces"]["api_v2"]["lifecycle"] == "current"

package = resources.files("prob4d")
assert package.joinpath("__init__.pyi").is_file()
assert package.joinpath("py.typed").is_file()
assert package.joinpath(
    "contract_data/observation_belief_v1/manifest.json"
).is_file()
"""
    _run([python, "-c", smoke])
    _run([python, "-m", "prob4d.public_api_manifest", "print"])

    cli = _venv_executable(environment, "prob4d")
    _run([cli, "--help"])
    _run([cli, "project", "identity", "--compact"])
    removed = _venv_executable(environment, "prob4d-validate-observation")
    if removed.exists():
        raise RuntimeError("removed legacy executable is still installed")


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