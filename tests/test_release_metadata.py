from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import prob4d

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LICENSE = "MIT"
EXPECTED_REPOSITORY = "https://github.com/IPS-Stuttgart/Prob4D"


def _project() -> dict[str, object]:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]


def _expected_version() -> str:
    value = _project()["version"]
    assert isinstance(value, str)
    return value


def test_package_and_citation_versions_are_synchronized() -> None:
    expected_version = _expected_version()
    assert prob4d.__version__ == expected_version
    try:
        installed_version = version("prob4d")
    except PackageNotFoundError:
        installed_version = expected_version
    assert installed_version == expected_version

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*["\']?([^"\'\s]+)', citation, re.MULTILINE)
    assert match is not None
    assert match.group(1) == expected_version
    assert re.search(r'^license:\s*["\']?MIT["\']?$', citation, re.MULTILINE)

    package_source = (ROOT / "src" / "prob4d" / "__init__.py").read_text(
        encoding="utf-8"
    )
    resolver_source = (ROOT / "src" / "prob4d" / "_version.py").read_text(
        encoding="utf-8"
    )
    assert "from ._version import __version__" in package_source
    assert re.search(r'__version__\s*=\s*["\']\d', package_source) is None
    assert 'UNKNOWN_VERSION = "0+unknown"' in resolver_source


def test_project_urls_point_to_the_canonical_repository() -> None:
    urls = _project()["urls"]
    assert isinstance(urls, dict)
    assert urls["Repository"] == EXPECTED_REPOSITORY
    assert urls["Documentation"].startswith(EXPECTED_REPOSITORY)
    assert urls["Issues"].startswith(EXPECTED_REPOSITORY)
    assert urls["Changelog"].startswith(EXPECTED_REPOSITORY)
    assert urls["Citation"].startswith(EXPECTED_REPOSITORY)
    assert urls["Security"].startswith(EXPECTED_REPOSITORY)
    assert urls["License"] == f"{EXPECTED_REPOSITORY}/blob/main/LICENSE"


def test_license_and_typing_metadata_are_explicit() -> None:
    project = _project()
    assert project["license"] == EXPECTED_LICENSE
    assert project["license-files"] == ["LICENSE"]

    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)
    assert "Typing :: Typed" in classifiers

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Florian Pfaff" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text


def test_observation_export_documentation_uses_an_explicit_route() -> None:
    documentation = (ROOT / "docs" / "observation-belief-export.md").read_text(encoding="utf-8")
    assert "prob4d observation export-calibrated \\" in documentation
    assert "prob4d observation export \\" not in documentation
    assert "prob4d observation export-v1" in documentation


def test_release_governance_files_exist() -> None:
    for name in (
        ".github/dependabot.yml",
        ".github/workflows/ecosystem-release-capsule.yml",
        ".github/workflows/security-scanning.yml",
        ".github/workflows/trusted-self-hosted-validation.yml",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "SECURITY.md",
        "docs/distribution-boundaries.md",
        "docs/ecosystem-release-capsule.md",
        "docs/public-api.md",
        "docs/releases/0.4.0.md",
        "docs/trusted-self-hosted-validation.md",
        "scripts/ci/build_ecosystem_release_capsule.py",
        "scripts/ci/check_sdist.py",
    ):
        assert (ROOT / name).is_file(), name
