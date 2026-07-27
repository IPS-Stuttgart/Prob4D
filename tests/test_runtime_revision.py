from __future__ import annotations

import pytest

import prob4d.runtime_revision as runtime_revision


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
    with pytest.raises(RuntimeError, match="tracked modifications"):
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
