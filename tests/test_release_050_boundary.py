"""Current Prob4D 0.5.0 cleanup-release boundary checks."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

import prob4d
import prob4d.provider_v1 as provider_v1_artifacts
from prob4d.provider_v2 import prob4d_provider_manifest as provider_v2_manifest
from prob4d.provider_v2_tree_sparse_manifest import (
    prob4d_tree_sparse_provider_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.5.0"


def _declared_version(path: Path, pattern: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, path
    return match.group(1)


def test_all_release_version_declarations_are_synchronized() -> None:
    assert (
        _declared_version(
            ROOT / "pyproject.toml",
            r'^version\s*=\s*["\']([^"\']+)',
        )
        == EXPECTED_VERSION
    )
    assert (
        _declared_version(
            ROOT / "CITATION.cff",
            r'^version:\s*["\']?([^"\'\s]+)',
        )
        == EXPECTED_VERSION
    )
    package_source = (ROOT / "src" / "prob4d" / "__init__.py").read_text(
        encoding="utf-8"
    )
    version_source = (ROOT / "src" / "prob4d" / "_version.py").read_text(
        encoding="utf-8"
    )
    assert "from ._version import __version__" in package_source
    assert 'UNKNOWN_VERSION = "0+unknown"' in version_source
    assert EXPECTED_VERSION not in version_source
    assert prob4d.__version__ == EXPECTED_VERSION


def test_cleanup_release_changelog_and_boundary_are_published() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## Unreleased\n\nNo changes yet." in changelog
    assert "## 0.5.0 — 2026-08-12" in changelog
    lower_changelog = changelog.lower()
    assert "standalone `prob4d-*`" in lower_changelog
    assert "package-root export inventory" in lower_changelog
    assert "`prob4d.api.v1`" in changelog
    assert "artifact compatibility bridge" in lower_changelog

    boundary = (ROOT / "docs" / "releases" / "0.5.0.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(boundary.split())
    assert "intentionally incompatible cleanup release" in normalized
    assert "installs only the grouped `prob4d` executable" in normalized
    assert "pin Prob4D 0.4.1" in normalized
    assert "changes no estimator equations" in normalized


def test_compatibility_guide_matches_the_05_surface() -> None:
    compatibility = (ROOT / "docs" / "compatibility.md").read_text(
        encoding="utf-8"
    )
    assert "## Prob4D 0.5 surfaces" in compatibility
    assert "minimal root" in compatibility
    assert "`prob4d.api.v1` | removed" in compatibility
    assert "`prob4d.api.v2`" in compatibility
    assert "artifact compatibility bridge" in compatibility
    assert "All historical `prob4d-*`" in compatibility
    assert "must not be rewritten" in compatibility


def test_current_and_historical_artifact_boundaries_remain_distinct() -> None:
    revision = "a" * 40
    historical = provider_v1_artifacts.prob4d_provider_manifest(
        provider_revision=revision
    )
    provider_v2 = provider_v2_manifest(provider_revision=revision)
    tree_sparse = prob4d_tree_sparse_provider_manifest(provider_revision=revision)

    assert historical["provider_api_version"] == 1
    assert historical["artifact_schema_versions"]["ObservationFactorBundle"] == 3
    assert historical["metadata"]["artifact_compatibility_only"] is True
    assert historical["metadata"]["execution_reproduction_release"] == "0.4.1"
    assert historical["limitations"]["provider_v1_execution_available"] is False
    assert not hasattr(provider_v1_artifacts, "export_observation_belief")
    assert not hasattr(
        provider_v1_artifacts,
        "export_calibrated_observation_belief",
    )

    assert provider_v2["provider_version"] == EXPECTED_VERSION
    assert provider_v2["provider_api_version"] == 2
    assert provider_v2["artifact_schema_versions"]["ObservationFactorBundle"] == 4

    versions = tree_sparse["artifact_schema_versions"]
    assert versions["TreeSparseObservationArtifactV1"] == 1
    assert versions["ClaimBearingTreeSparseObservationEnvelopeV1"] == 1
    assert "content_addressed_tree_sparse_observation_artifacts" in tree_sparse[
        "capabilities"
    ]


def test_legacy_runtime_surface_is_absent() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"] == {"prob4d": "prob4d.cli:main"}
    assert importlib.util.find_spec("prob4d.api.v1") is None
    assert importlib.util.find_spec("prob4d.legacy_cli") is None
    assert importlib.util.find_spec("prob4d.causal_stream_cli") is None
    assert not (ROOT / "tests" / "test_api_v1.py").exists()
    assert not (ROOT / "tests" / "test_provider_v1.py").exists()
    assert not (ROOT / "tests" / "test_legacy_cli.py").exists()
    assert not (ROOT / ".github" / "workflows" / "legacy-cli-migration.yml").exists()
    assert not (ROOT / "docs" / "legacy-cli-migration.md").exists()
    assert prob4d.__all__ == ["__version__"]
    assert not hasattr(prob4d, "Sim3")
    assert not hasattr(prob4d, "_LAZY_EXPORTS")


def test_active_05_release_checks_do_not_reinstall_removed_surfaces() -> None:
    checked = (
        ROOT / "scripts" / "ci" / "check_sdist.py",
        ROOT / ".github" / "workflows" / "tests.yml",
        ROOT / "pyproject.toml",
    )
    forbidden = (
        "prob4d.api.v1 as",
        "prob4d observation export-v1 --help",
        "prob4d.legacy_cli:",
        '"--api-version", "1"',
    )
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (path, token)
