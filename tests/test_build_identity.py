from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d._build_identity import (
    BUILD_IDENTITY_FILENAME,
    build_identity_record,
    load_build_identity,
    package_manifest,
    write_build_identity,
)

ROOT = Path(__file__).resolve().parents[1]


def test_build_identity_binds_all_installed_package_bytes(tmp_path: Path) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    data = package / "contract_data"
    data.mkdir()
    (data / "schema.json").write_text('{"version": 1}\n', encoding="utf-8")

    identity_path = package / BUILD_IDENTITY_FILENAME
    record = write_build_identity(
        identity_path,
        package_root=package,
        source_revision="a" * 40,
        source_tree_clean=True,
        source_identity_source="git_checkout",
    )

    loaded = load_build_identity(identity_path)
    assert loaded == record
    assert loaded["package_file_count"] == 2

    (package / "module.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="package bytes"):
        load_build_identity(identity_path)


def test_manifest_ignores_identity_and_interpreter_bytecode(tmp_path: Path) -> None:
    package = tmp_path / "prob4d"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    (cache / "module.cpython-312.pyc").write_bytes(b"generated")

    before = package_manifest(package)
    (package / BUILD_IDENTITY_FILENAME).write_text("{}\n", encoding="utf-8")
    after = package_manifest(package)

    assert before == after
    assert after[1] == 1


def test_build_identity_rejects_duplicate_keys(tmp_path: Path) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    manifest, count = package_manifest(package)
    path = package / BUILD_IDENTITY_FILENAME
    revision = "a" * 40
    path.write_text(
        "{"
        '"schema":"prob4d.installed-build-identity",'
        '"schema":"prob4d.installed-build-identity",'
        '"schema_version":1,'
        '"repository":"IPS-Stuttgart/Prob4D",'
        f'"source_revision":"{revision}",'
        '"source_tree_clean":true,'
        '"source_identity_source":"git_checkout",'
        f'"package_manifest_sha256":"{manifest}",'
        f'"package_file_count":{count}'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_build_identity(path)


def test_unavailable_identity_cannot_claim_clean_source(tmp_path: Path) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be marked clean"):
        build_identity_record(
            package,
            source_revision=None,
            source_tree_clean=True,
            source_identity_source="unavailable",
        )


def test_identity_record_is_strict_about_unknown_fields(tmp_path: Path) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    path = package / BUILD_IDENTITY_FILENAME
    record = write_build_identity(
        path,
        package_root=package,
        source_revision="a" * 40,
        source_tree_clean=True,
        source_identity_source="git_checkout",
    )
    record["unexpected"] = True
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed"):
        load_build_identity(path)


def test_distribution_build_hooks_are_declared() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert '"_build_identity.json"' in pyproject
    assert "include setup.py" in manifest
    assert "Prob4DBuildPy" in setup_source
    assert "Prob4DSdist" in setup_source
    assert 'getattr(self, "editable_mode", False)' in setup_source
    assert "ordinary Prob4D build did not stage the package root" in setup_source
    assert "--untracked-files=all" in setup_source
    assert "--ignore-submodules=none" in setup_source
