from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "build_ecosystem_release_capsule.py"
SPEC = importlib.util.spec_from_file_location("build_ecosystem_release_capsule", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _content_addressed(descriptor: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(descriptor)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    payload["manifest_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _manifest_v2() -> dict[str, Any]:
    return _content_addressed(
        {
            "schema": "prob4d.public-api-manifest",
            "schema_version": 2,
            "package": {
                "name": "prob4d",
                "version": "0.5.0",
                "project_id": "github-repository-id:1295794737",
            },
            "surfaces": {
                "package_root": {
                    "module": "prob4d",
                    "surface_version": 2,
                    "loading": "minimal-version-root-v1",
                    "exports": ["__version__"],
                },
                "api_v2": {
                    "module": "prob4d.api.v2",
                    "api_version": 2,
                    "provider_api_version": 2,
                    "provider_factor_api_version": 2,
                    "lifecycle": "current",
                    "exports": ["API_VERSION"],
                },
            },
            "claim_boundary": MODULE._PUBLIC_API_CLAIM_BOUNDARY,
        }
    )


def test_capsule_accepts_public_api_manifest_v2() -> None:
    manifest = _manifest_v2()
    assert MODULE._validate_public_api_manifest(manifest) == manifest


def test_capsule_rejects_retired_public_api_surfaces() -> None:
    descriptor = _manifest_v2()
    descriptor.pop("manifest_id")
    descriptor["surfaces"]["api_v1"] = {
        "module": "prob4d.api.v1",
        "api_version": 1,
        "provider_api_version": 1,
        "exports": ["API_VERSION"],
    }
    manifest = _content_addressed(descriptor)

    with pytest.raises(ValueError, match="public API surfaces"):
        MODULE._validate_public_api_manifest(manifest)
