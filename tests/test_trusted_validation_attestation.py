from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from prob4d.trusted_validation_attestation import (
    TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY,
    TRUSTED_VALIDATION_ATTESTATION_DRAFT_SCHEMA,
    TrustedValidationBoundaryAttestationV1,
    build_trusted_validation_attestation_draft,
    load_trusted_validation_attestation,
    main,
    seal_trusted_validation_attestation_draft,
    write_trusted_validation_attestation,
)

SOURCE_REVISION = "1" * 40
WORKFLOW_SHA256 = "2" * 64
EVIDENCE_SHA256 = "3" * 64


def _verified_section(draft: dict[str, object], field: str, method: str) -> None:
    section = draft[field]
    assert isinstance(section, dict)
    section["verification_status"] = "verified"
    section["verified_by"] = "independent-verifier"
    section["verified_at"] = "2026-08-12T08:30:00Z"
    section["evidence_method"] = method
    section["evidence_locator"] = f"urn:prob4d:external-audit:{field}"
    section["evidence_sha256"] = EVIDENCE_SHA256
    assertions = section["assertions"]
    assert isinstance(assertions, dict)
    for key in assertions:
        assertions[key] = True


def _ready_draft() -> dict[str, object]:
    draft = build_trusted_validation_attestation_draft(
        source_revision=SOURCE_REVISION,
        workflow_sha256=WORKFLOW_SHA256,
    )
    _verified_section(draft, "github_environment", "github-api-query")
    _verified_section(draft, "runner_host", "independent-host-audit")
    _verified_section(
        draft,
        "dataset_namespace",
        "independent-dataset-namespace-audit",
    )
    _verified_section(draft, "exact_head_acceptance", "github-actions-api-query")
    return draft


def test_template_is_explicitly_unverified_and_not_ready() -> None:
    draft = build_trusted_validation_attestation_draft(
        source_revision=SOURCE_REVISION,
        workflow_sha256=WORKFLOW_SHA256,
    )

    assert draft["schema"] == TRUSTED_VALIDATION_ATTESTATION_DRAFT_SCHEMA
    attestation = seal_trusted_validation_attestation_draft(draft)

    assert not attestation.ready
    assert attestation.failure_reasons == (
        "dataset-namespace:unverified",
        "exact-head-acceptance:unverified",
        "github-environment:unverified",
        "runner-host:unverified",
    )


def test_complete_independent_evidence_seals_ready_attestation() -> None:
    attestation = seal_trusted_validation_attestation_draft(_ready_draft())

    assert attestation.ready
    assert attestation.failure_reasons == ()
    assert len(attestation.trusted_validation_boundary_attestation_id) == 64
    assert attestation.to_dict()["claim_boundary"] == (
        TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY
    )


def test_failed_negative_control_keeps_boundary_closed() -> None:
    draft = _ready_draft()
    section = draft["exact_head_acceptance"]
    assert isinstance(section, dict)
    section["verification_status"] = "failed"
    assertions = section["assertions"]
    assert isinstance(assertions, dict)
    assertions["stale_sha_rejected_before_self_hosted_checkout"] = False

    attestation = seal_trusted_validation_attestation_draft(draft)

    assert not attestation.ready
    assert attestation.failure_reasons == (
        "exact-head-acceptance:stale_sha_rejected_before_self_hosted_checkout",
    )


def test_verified_section_cannot_hide_failed_assertion() -> None:
    draft = _ready_draft()
    section = draft["runner_host"]
    assert isinstance(section, dict)
    assertions = section["assertions"]
    assert isinstance(assertions, dict)
    assertions["dedicated_non_administrator_account"] = False

    with pytest.raises(ValueError, match="every assertion to pass"):
        seal_trusted_validation_attestation_draft(draft)


def test_unverified_section_cannot_claim_external_evidence() -> None:
    draft = build_trusted_validation_attestation_draft(
        source_revision=SOURCE_REVISION,
        workflow_sha256=WORKFLOW_SHA256,
    )
    section = draft["runner_host"]
    assert isinstance(section, dict)
    section["evidence_locator"] = "urn:prob4d:external-audit:forged"

    with pytest.raises(ValueError, match="must not claim external evidence"):
        seal_trusted_validation_attestation_draft(draft)


def test_repository_relative_evidence_is_rejected() -> None:
    draft = _ready_draft()
    section = draft["runner_host"]
    assert isinstance(section, dict)
    section["evidence_locator"] = "docs/runner-audit.md"

    with pytest.raises(ValueError, match="external HTTPS URL"):
        seal_trusted_validation_attestation_draft(draft)


def test_repository_blob_url_is_rejected_as_external_evidence() -> None:
    draft = _ready_draft()
    section = draft["github_environment"]
    assert isinstance(section, dict)
    section["evidence_locator"] = (
        "https://github.com/IPS-Stuttgart/Prob4D/blob/main/docs/runner-audit.md"
    )

    with pytest.raises(ValueError, match="repository files cannot serve"):
        seal_trusted_validation_attestation_draft(draft)


def test_repository_file_is_not_an_evidence_method() -> None:
    draft = _ready_draft()
    section = draft["github_environment"]
    assert isinstance(section, dict)
    section["evidence_method"] = "repository-file"

    with pytest.raises(ValueError, match="evidence_method"):
        seal_trusted_validation_attestation_draft(draft)


def test_round_trip_recomputes_derived_fields_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    attestation = seal_trusted_validation_attestation_draft(_ready_draft())
    path = tmp_path / "attestation.json"
    write_trusted_validation_attestation(path, attestation)

    loaded = load_trusted_validation_attestation(path)
    assert loaded.trusted_validation_boundary_attestation_id == (
        attestation.trusted_validation_boundary_attestation_id
    )

    tampered = deepcopy(attestation.to_dict())
    tampered["ready"] = False
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="derived fields changed"):
        load_trusted_validation_attestation(path)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_trusted_validation_attestation(path)


def test_writer_is_no_clobber(tmp_path: Path) -> None:
    attestation = seal_trusted_validation_attestation_draft(_ready_draft())
    path = tmp_path / "attestation.json"
    write_trusted_validation_attestation(path, attestation)

    with pytest.raises(FileExistsError):
        write_trusted_validation_attestation(path, attestation)


def test_cli_template_seal_and_require_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    draft_path = tmp_path / "draft.json"
    artifact_path = tmp_path / "attestation.json"
    assert (
        main(
            [
                "template",
                "--source-revision",
                SOURCE_REVISION,
                "--workflow-sha256",
                WORKFLOW_SHA256,
                "--output",
                str(draft_path),
            ]
        )
        == 0
    )
    assert main(["seal", "--draft", str(draft_path), "--output", str(artifact_path)]) == 0
    assert main(["verify", "--artifact", str(artifact_path)]) == 0
    assert main(["verify", "--artifact", str(artifact_path), "--require-ready"]) == 2
    assert '"ready": false' in capsys.readouterr().out


def test_direct_constructor_requires_canonical_section_placement() -> None:
    attestation = seal_trusted_validation_attestation_draft(_ready_draft())

    with pytest.raises(ValueError, match="section name changed"):
        TrustedValidationBoundaryAttestationV1(
            repository=attestation.repository,
            environment_name=attestation.environment_name,
            workflow_path=attestation.workflow_path,
            source_revision=attestation.source_revision,
            workflow_sha256=attestation.workflow_sha256,
            github_environment=attestation.runner_host,
            runner_host=attestation.github_environment,
            dataset_namespace=attestation.dataset_namespace,
            exact_head_acceptance=attestation.exact_head_acceptance,
        )
