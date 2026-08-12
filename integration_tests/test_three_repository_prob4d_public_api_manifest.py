"""Materialize the installed Prob4D public-API manifest for the wheel capsule."""

from __future__ import annotations

import os
from importlib import metadata as importlib_metadata
from pathlib import Path

from prob4d.public_api_manifest import (
    PACKAGE_ROOT_LOADING_MODE,
    PACKAGE_ROOT_SURFACE_VERSION,
    PUBLIC_API_MANIFEST_VERSION,
    build_public_api_manifest,
    load_public_api_manifest,
    validate_public_api_manifest,
    write_public_api_manifest,
)

_PREEXISTING_EVIDENCE = frozenset(
    {
        "accepted-selection.json",
        "exact-prob4d-observation.npz",
        "golden-path-bundle.json",
        "lineage-bound-physical-posterior.npz",
        "lineage-bound-twin-belief.npz",
        "profile-bound.npz",
        "rejected-selection.json",
        "run-manifest-v2.json",
    }
)


def test_materialize_installed_public_api_manifest() -> None:
    manifest = build_public_api_manifest()
    assert validate_public_api_manifest(manifest) == manifest
    assert manifest["schema_version"] == PUBLIC_API_MANIFEST_VERSION == 2
    assert manifest["package"]["version"] == importlib_metadata.version("prob4d")

    surfaces = manifest["surfaces"]
    assert set(surfaces) == {"package_root", "api_v2"}
    package_root = surfaces["package_root"]
    assert package_root["surface_version"] == PACKAGE_ROOT_SURFACE_VERSION == 2
    assert package_root["loading"] == PACKAGE_ROOT_LOADING_MODE == "minimal-version-root-v1"
    assert package_root["exports"] == ["__version__"]

    api_v2 = surfaces["api_v2"]
    assert api_v2["api_version"] == 2
    assert api_v2["provider_api_version"] == 2
    assert api_v2["provider_factor_api_version"] == 2
    assert api_v2["lifecycle"] == "current"

    output_value = os.environ.get("THREE_REPOSITORY_EVIDENCE_OUTPUT")
    if output_value is None:
        return
    output = Path(output_value).resolve()
    if output.is_symlink() or not output.is_dir():
        raise AssertionError("golden-path evidence output must be a regular directory")
    actual = frozenset(path.name for path in output.iterdir())
    assert actual == _PREEXISTING_EVIDENCE

    destination = output / "public-api-manifest.json"
    written = write_public_api_manifest(destination, manifest)
    assert written == manifest
    assert load_public_api_manifest(destination) == manifest
    assert build_public_api_manifest() == manifest
