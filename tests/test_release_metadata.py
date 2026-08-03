from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10.
    import tomli as tomllib

import prob4d


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.3.1"
EXPECTED_REPOSITORY = "https://github.com/IPS-Stuttgart/Prob4D"


def _project() -> dict[str, object]:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]


def test_package_and_citation_versions_are_synchronized() -> None:
    project = _project()
    assert project["version"] == EXPECTED_VERSION
    assert prob4d.__version__ == EXPECTED_VERSION
    try:
        installed_version = version("prob4d")
    except PackageNotFoundError:
        installed_version = EXPECTED_VERSION
    assert installed_version == EXPECTED_VERSION

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*["\']?([^"\'\s]+)', citation, re.MULTILINE)
    assert match is not None
    assert match.group(1) == EXPECTED_VERSION


def test_project_urls_point_to_the_canonical_repository() -> None:
    urls = _project()["urls"]
    assert isinstance(urls, dict)
    assert urls["Repository"] == EXPECTED_REPOSITORY
    assert urls["Documentation"].startswith(EXPECTED_REPOSITORY)
    assert urls["Issues"].startswith(EXPECTED_REPOSITORY)
    assert urls["Changelog"].startswith(EXPECTED_REPOSITORY)
    assert urls["Citation"].startswith(EXPECTED_REPOSITORY)
    assert urls["Security"].startswith(EXPECTED_REPOSITORY)


def test_release_governance_files_exist() -> None:
    for name in (
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "MANIFEST.in",
        "SECURITY.md",
    ):
        assert (ROOT / name).is_file(), name
