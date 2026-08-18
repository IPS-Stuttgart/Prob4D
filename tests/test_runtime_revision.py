from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import prob4d.runtime_revision as runtime_revision
from prob4d._build_identity import BUILD_IDENTITY_FILENAME, write_build_identity


def test_matching_clean_checkout_is_independently_verified(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_revision,
        "_resolve_runtime_revision",
        lambda **kwargs: ("a" * 40, "source_checkout", True),
    )
    attestation = runtime_revision.assert_runtime_revision("a" * 40)

    assert attestation.matched is True
    assert attestation.independently_verified is True
    assert attestation.as_metadata()["clean_checkout"] is True


def test_matching_deployment_assertion_is_exploratory_only(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_revision,
        "_resolve_runtime_revision",
        lambda **kwargs: ("a" * 40, "deployment_environment", None),
    )
    inspected = runtime_revision.inspect_runtime_revision("a" * 40)

    assert inspected.matched is True
    assert inspected.independently_verified is False
    assert inspected.source == "deployment_environment"
    with pytest.raises(RuntimeError, match="independent VCS revision evidence"):
        runtime_revision.assert_runtime_revision("a" * 40)


def test_claim_bearing_revision_mismatch_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_revision,
        "_resolve_runtime_revision",
        lambda **kwargs: ("b" * 40, "installed_vcs_metadata", None),
    )
    with pytest.raises(RuntimeError, match="revision mismatch"):
        runtime_revision.assert_runtime_revision("a" * 40)


def test_claim_bearing_dirty_checkout_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_revision,
        "_resolve_runtime_revision",
        lambda **kwargs: ("a" * 40, "source_checkout", False),
    )
    with pytest.raises(RuntimeError, match="tracked or untracked modifications"):
        runtime_revision.assert_runtime_revision("a" * 40)


def test_unavailable_runtime_revision_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_revision,
        "_resolve_runtime_revision",
        lambda **kwargs: (None, "unavailable", None),
    )
    with pytest.raises(RuntimeError, match="cannot determine"):
        runtime_revision.assert_runtime_revision("a" * 40)


def test_revision_format_is_strict(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_revision,
        "_resolve_runtime_revision",
        lambda **kwargs: (None, "unavailable", None),
    )
    with pytest.raises(ValueError, match="exact lowercase"):
        runtime_revision.inspect_runtime_revision("A" * 40)


def test_content_verified_installed_build_identity_is_accepted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    identity_path = package / BUILD_IDENTITY_FILENAME
    write_build_identity(
        identity_path,
        package_root=package,
        source_revision="a" * 40,
        source_tree_clean=True,
        source_identity_source="git_checkout",
    )
    monkeypatch.setattr(runtime_revision, "_build_identity_path", lambda: identity_path)
    monkeypatch.setattr(
        runtime_revision,
        "_installed_direct_url_revision",
        lambda: None,
    )
    monkeypatch.setattr(runtime_revision, "_source_checkout_revision", lambda root: None)

    attestation = runtime_revision.assert_runtime_revision(
        "a" * 40,
        checkout_root=tmp_path / "missing-checkout",
    )

    assert attestation.source == "installed_vcs_metadata"
    assert attestation.clean_checkout is None
    assert attestation.independently_verified is True


def test_tampered_installed_build_identity_fails_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    module = package / "module.py"
    module.write_text("value = 1\n", encoding="utf-8")
    identity_path = package / BUILD_IDENTITY_FILENAME
    write_build_identity(
        identity_path,
        package_root=package,
        source_revision="a" * 40,
        source_tree_clean=True,
        source_identity_source="git_checkout",
    )
    module.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(runtime_revision, "_build_identity_path", lambda: identity_path)

    with pytest.raises(RuntimeError, match="build identity is invalid"):
        runtime_revision.inspect_runtime_revision("a" * 40)


def test_dirty_embedded_build_identity_fails_before_direct_url_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "prob4d"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    identity_path = package / BUILD_IDENTITY_FILENAME
    write_build_identity(
        identity_path,
        package_root=package,
        source_revision="a" * 40,
        source_tree_clean=False,
        source_identity_source="git_checkout",
    )
    monkeypatch.setattr(runtime_revision, "_build_identity_path", lambda: identity_path)
    monkeypatch.setattr(
        runtime_revision,
        "_installed_direct_url_revision",
        lambda: "b" * 40,
    )

    with pytest.raises(RuntimeError, match="does not attest a clean source"):
        runtime_revision._installed_vcs_revision()


def test_source_checkout_detects_untracked_files(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "initial"],
        check=True,
    )

    clean = runtime_revision._source_checkout_revision(repository)
    assert clean is not None and clean[1] is True

    (repository / "untracked.py").write_text("value = 1\n", encoding="utf-8")
    dirty = runtime_revision._source_checkout_revision(repository)
    assert dirty is not None and dirty[1] is False
