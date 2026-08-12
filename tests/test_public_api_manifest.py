from __future__ import annotations

import json
from copy import deepcopy

import pytest

import prob4d
from prob4d.api import v2 as api_v2
from prob4d.public_api_manifest import (
    PACKAGE_ROOT_LOADING_MODE,
    PACKAGE_ROOT_SURFACE_VERSION,
    PUBLIC_API_MANIFEST_CLAIM_BOUNDARY,
    PUBLIC_API_MANIFEST_SCHEMA,
    PUBLIC_API_MANIFEST_VERSION,
    build_public_api_manifest,
    load_public_api_manifest,
    main,
    validate_public_api_manifest,
    write_public_api_manifest,
)


def test_public_api_manifest_is_deterministic_and_complete() -> None:
    first = build_public_api_manifest()
    second = build_public_api_manifest()

    assert first == second
    assert first["schema"] == PUBLIC_API_MANIFEST_SCHEMA
    assert first["schema_version"] == PUBLIC_API_MANIFEST_VERSION == 2
    assert first["package"]["version"] == prob4d.__version__
    assert first["claim_boundary"] == PUBLIC_API_MANIFEST_CLAIM_BOUNDARY

    surfaces = first["surfaces"]
    assert set(surfaces) == {"package_root", "api_v2"}
    assert surfaces["package_root"]["surface_version"] == PACKAGE_ROOT_SURFACE_VERSION
    assert surfaces["package_root"]["loading"] == PACKAGE_ROOT_LOADING_MODE
    assert surfaces["package_root"]["exports"] == ["__version__"]
    assert surfaces["api_v2"]["api_version"] == api_v2.API_VERSION == 2
    assert surfaces["api_v2"]["lifecycle"] == "current"
    assert surfaces["api_v2"]["exports"] == sorted(api_v2.__all__)
    assert validate_public_api_manifest(first) == first


def test_public_api_manifest_rejects_tampering_and_retired_surfaces() -> None:
    manifest = build_public_api_manifest()
    tampered = deepcopy(manifest)
    tampered["package"]["version"] = "999.0.0"
    with pytest.raises(ValueError, match="manifest_id does not match"):
        validate_public_api_manifest(tampered)

    retired = deepcopy(manifest)
    retired["surfaces"]["api_v1"] = {
        "module": "prob4d.api.v1",
        "api_version": 1,
        "exports": [],
    }
    with pytest.raises(ValueError, match="noncanonical keys"):
        validate_public_api_manifest(retired)

    unsorted = deepcopy(manifest)
    unsorted["surfaces"]["api_v2"]["exports"] = list(
        reversed(unsorted["surfaces"]["api_v2"]["exports"])
    )
    with pytest.raises(ValueError, match="sorted canonically"):
        validate_public_api_manifest(unsorted)


def test_public_api_manifest_round_trip_is_no_clobber(tmp_path) -> None:
    output = tmp_path / "public-api.json"
    expected = build_public_api_manifest()

    assert write_public_api_manifest(output) == expected
    assert load_public_api_manifest(output) == expected
    assert write_public_api_manifest(output) == expected

    changed = deepcopy(expected)
    changed["package"]["version"] = "999.0.0"
    changed["manifest_id"] = "0" * 64
    output.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        write_public_api_manifest(output)


def test_public_api_manifest_cli_print_build_and_verify(tmp_path, capsys) -> None:
    assert main(["print"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == build_public_api_manifest()

    output = tmp_path / "public-api.json"
    assert main(["build", "--output", str(output)]) == 0
    build_output = capsys.readouterr().out.strip()
    assert build_output == build_public_api_manifest()["manifest_id"]

    assert main(["verify", str(output), "--require-current"]) == 0
    verify_output = capsys.readouterr().out.strip()
    assert verify_output == build_output


def test_public_api_manifest_loader_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_public_api_manifest(path)
