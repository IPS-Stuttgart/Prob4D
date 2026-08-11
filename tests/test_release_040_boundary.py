from __future__ import annotations

import re
from pathlib import Path

from prob4d.provider_manifest import (
    PROB4D_PROVIDER_PACKAGE_VERSION,
    prob4d_provider_manifest,
)
from prob4d.provider_v2 import prob4d_provider_manifest as provider_v2_manifest
from prob4d.provider_v2_tree_sparse_manifest import (
    prob4d_tree_sparse_provider_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.4.0"


def _declared_version(path: Path, pattern: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, path
    return match.group(1)


def test_all_source_release_version_declarations_are_synchronized() -> None:
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
    assert PROB4D_PROVIDER_PACKAGE_VERSION == EXPECTED_VERSION


def test_release_changelog_and_claim_boundary_are_published() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## Unreleased\n\nNo changes yet." in changelog
    assert "## 0.4.0 — 2026-08-10" in changelog
    lower_changelog = changelog.lower()
    assert "claim-bearing tree-sparse observation artifacts" in lower_changelog
    assert "outcome-blind provider support feasibility" in lower_changelog
    assert "replay-complete held-out provider evidence" in lower_changelog
    assert "canonical grouped `prob4d` registry" in lower_changelog
    assert "compatibility wrappers" in lower_changelog
    assert "versioned `prob4d.api.v1`" in lower_changelog
    assert "installed source distribution" in lower_changelog

    boundary = (ROOT / "docs" / "releases" / "0.4.0.md").read_text(
        encoding="utf-8"
    )
    normalized_boundary = " ".join(boundary.split())
    assert (
        "does not publish to a package registry or create a Git tag"
        in normalized_boundary
    )
    assert "does not establish real-provider competence" in normalized_boundary
    assert "313/324" in normalized_boundary


def test_compatibility_guide_matches_the_040_surface() -> None:
    compatibility = (ROOT / "docs" / "compatibility.md").read_text(
        encoding="utf-8"
    )

    assert "## Prob4D 0.4.x surfaces" in compatibility
    assert "## Prob4D 0.3.x surfaces" not in compatibility
    for surface in (
        "`prob4d.api.v2`",
        "`GaugeTreeSquareRootPriorV1`",
        "`TreeSparseObservationArtifactV1`",
        "`ClaimBearingTreeSparseObservationEnvelopeV1`",
        "`prob4d.provider_v2_factors.v1`",
    ):
        assert surface in compatibility
    assert "At the time Prob4D 0.3.0 was prepared" not in compatibility
    assert "Do not maintain a table of mutable companion-repository" in compatibility


def test_v1_v2_and_tree_sparse_manifest_boundaries_remain_distinct() -> None:
    revision = "a" * 40
    provider_v1 = prob4d_provider_manifest(provider_revision=revision)
    provider_v2 = provider_v2_manifest(provider_revision=revision)
    tree_sparse = prob4d_tree_sparse_provider_manifest(provider_revision=revision)

    assert provider_v1["provider_version"] == EXPECTED_VERSION
    assert provider_v1["provider_api_version"] == 1
    assert provider_v1["artifact_schema_versions"]["ObservationFactorBundle"] == 3

    assert provider_v2["provider_version"] == EXPECTED_VERSION
    assert provider_v2["provider_api_version"] == 2
    assert provider_v2["artifact_schema_versions"]["ObservationFactorBundle"] == 4

    versions = tree_sparse["artifact_schema_versions"]
    assert versions["TreeSparseObservationArtifactV1"] == 1
    assert versions["ClaimBearingTreeSparseObservationEnvelopeV1"] == 1
    assert "content_addressed_tree_sparse_observation_artifacts" in tree_sparse[
        "capabilities"
    ]


def test_release_boundary_contains_no_self_mutating_workflow() -> None:
    assert not (ROOT / ".github" / "workflows" / "prepare-release-040.yml").exists()
    assert not (ROOT / "scripts" / "ci" / "prepare_release_040.py").exists()


def test_ci_and_release_tests_do_not_retain_the_previous_version() -> None:
    checked = (
        ROOT / ".github" / "workflows" / "tests.yml",
        ROOT / "tests" / "test_release_metadata.py",
        ROOT / "tests" / "test_provider_manifest.py",
    )
    for path in checked:
        assert "0.3.1" not in path.read_text(encoding="utf-8"), path
