"""Setuptools hooks for content-bound Prob4D distribution identities."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

ROOT = Path(__file__).resolve().parent
SOURCE_PACKAGE = ROOT / "src" / "prob4d"
IDENTITY_MODULE = SOURCE_PACKAGE / "_build_identity.py"
IDENTITY_FILE = SOURCE_PACKAGE / "_build_identity.json"


def _load_identity_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_prob4d_build_identity_support",
        IDENTITY_MODULE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Prob4D build-identity support")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


identity = _load_identity_module()


def _git_source_identity(root: Path) -> tuple[str, bool] | None:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip()
        if Path(top_level).resolve() != root.resolve():
            return None
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return revision, not bool(status.strip())


def _source_identity() -> tuple[str | None, bool, str]:
    git_identity = _git_source_identity(ROOT)
    if git_identity is not None:
        revision, clean = git_identity
        return revision, clean, "git_checkout"
    if IDENTITY_FILE.is_file():
        inherited = identity.load_build_identity(
            IDENTITY_FILE,
            package_root=SOURCE_PACKAGE,
        )
        return (
            inherited["source_revision"],
            inherited["source_tree_clean"],
            "inherited_source_archive",
        )
    return None, False, "unavailable"


def _write_staged_identity(
    package_root: Path,
    source_identity: tuple[str | None, bool, str],
) -> None:
    revision, clean, source = source_identity
    identity.write_build_identity(
        package_root / identity.BUILD_IDENTITY_FILENAME,
        package_root=package_root,
        source_revision=revision,
        source_tree_clean=clean,
        source_identity_source=source,
    )


class Prob4DBuildPy(_build_py):
    """Inject a manifest-bound source revision into ordinary wheel staging."""

    def run(self) -> None:
        source_identity = _source_identity()
        super().run()
        if getattr(self, "editable_mode", False):
            # PEP 660 maps imports directly to the source tree and may not create
            # build_lib/prob4d. Runtime attestation therefore remains the clean
            # source-checkout path rather than claiming an installed artifact.
            return
        package_root = Path(self.build_lib) / "prob4d"
        if not package_root.is_dir():
            raise RuntimeError("ordinary Prob4D build did not stage the package root")
        _write_staged_identity(package_root, source_identity)


class Prob4DSdist(_sdist):
    """Carry the original clean revision through source-distribution rebuilds."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        revision, clean, source = _source_identity()
        super().make_release_tree(base_dir, files)
        package_root = Path(base_dir) / "src" / "prob4d"
        identity.write_build_identity(
            package_root / identity.BUILD_IDENTITY_FILENAME,
            package_root=package_root,
            source_revision=revision,
            source_tree_clean=clean,
            source_identity_source=source,
        )


setup(cmdclass={"build_py": Prob4DBuildPy, "sdist": Prob4DSdist})
