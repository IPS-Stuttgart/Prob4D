"""External attestation for the trusted self-hosted validation boundary.

The repository workflow can prove exact-head execution semantics, but repository files
cannot prove GitHub environment settings, runner-host hardening, dataset namespace
isolation, or that the registered positive and negative controls were actually run.
This module records those four externally verified boundaries without treating a
checked box or repository-local document as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, cast
from urllib.parse import urlparse

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)

TRUSTED_VALIDATION_ATTESTATION_DRAFT_SCHEMA: Final = (
    "prob4d.trusted-validation-boundary-attestation-draft"
)
TRUSTED_VALIDATION_ATTESTATION_SCHEMA: Final = (
    "prob4d.trusted-validation-boundary-attestation"
)
TRUSTED_VALIDATION_ATTESTATION_VERSION: Final = 1
TRUSTED_VALIDATION_ENVIRONMENT: Final = "trusted-self-hosted-validation"
TRUSTED_VALIDATION_WORKFLOW_PATH: Final = (
    ".github/workflows/trusted-self-hosted-validation.yml"
)
TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY: Final = (
    "A ready attestation records independent external evidence for the GitHub "
    "environment policy, runner-host hardening, dataset namespace isolation, and "
    "exact-head positive and negative controls at one evidence snapshot. It does not "
    "make repository assertions proof of external state, guarantee that the boundary "
    "remains unchanged, authorize protected target access, establish provider accuracy "
    "or calibration, establish BayesianPhysTwin or Causal4D benefit, approve deployment, "
    "or establish state of the art."
)

SectionName = Literal[
    "github-environment",
    "runner-host",
    "dataset-namespace",
    "exact-head-acceptance",
]
VerificationStatus = Literal["unverified", "verified", "failed"]

_SECTION_ORDER: Final[tuple[SectionName, ...]] = (
    "github-environment",
    "runner-host",
    "dataset-namespace",
    "exact-head-acceptance",
)
_SECTION_ASSERTIONS: Final[dict[SectionName, tuple[str, ...]]] = {
    "github-environment": (
        "independent_reviewer_required",
        "no_environment_secrets",
        "workflow_definition_restricted_to_main",
        "self_approval_prevented_or_documented_unavailable",
        "read_only_permissions_confirmed",
    ),
    "runner-host": (
        "dedicated_non_administrator_account",
        "no_ssh_cloud_registry_or_personal_credentials",
        "unrelated_homes_repositories_caches_and_services_restricted",
        "network_destinations_restricted_or_documented",
        "incident_rebuild_or_rotation_procedure_defined",
    ),
    "dataset-namespace": (
        "only_approved_datasets_exposed",
        "approved_datasets_read_only_where_possible",
        "unopened_target_cohorts_outside_ordinary_namespace",
        "dataset_namespace_inventory_independently_verified",
    ),
    "exact-head-acceptance": (
        "exact_head_positive_control_passed",
        "environment_approval_pause_observed",
        "stale_sha_rejected_before_self_hosted_checkout",
        "non_main_ref_rejected_before_self_hosted_checkout",
        "retained_evidence_bound_pr_head_base_runner_and_profile",
    ),
}
_ALLOWED_EVIDENCE_METHODS: Final[dict[SectionName, tuple[str, ...]]] = {
    "github-environment": (
        "github-api-query",
        "signed-administrative-attestation",
    ),
    "runner-host": (
        "independent-host-audit",
        "signed-administrative-attestation",
    ),
    "dataset-namespace": (
        "independent-dataset-namespace-audit",
        "signed-administrative-attestation",
    ),
    "exact-head-acceptance": (
        "github-actions-api-query",
        "signed-administrative-attestation",
    ),
}
_SECTION_FIELDS = frozenset(
    {
        "section_name",
        "verification_status",
        "verified_by",
        "verified_at",
        "evidence_method",
        "evidence_locator",
        "evidence_sha256",
        "assertions",
        "notes",
    }
)
_DRAFT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "repository",
        "environment_name",
        "workflow_path",
        "source_revision",
        "workflow_sha256",
        "github_environment",
        "runner_host",
        "dataset_namespace",
        "exact_head_acceptance",
        "metadata",
        "claim_boundary",
    }
)
_ATTESTATION_FIELDS = frozenset(
    set(_DRAFT_FIELDS)
    | {
        "ready",
        "failure_reasons",
        "trusted_validation_boundary_attestation_id",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_exact_string(value, name=name)


def _exact_source_revision(value: object) -> str:
    revision = require_exact_string(value, name="source_revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source_revision must be a full lowercase 40-character Git SHA")
    return revision


def _repository(value: object) -> str:
    repository = require_exact_string(value, name="repository")
    if repository.count("/") != 1 or repository.startswith("/") or repository.endswith("/"):
        raise ValueError("repository must use canonical owner/name form")
    return repository


def _verified_at(value: object) -> str | None:
    timestamp = _strict_optional_string(value, name="verified_at")
    if timestamp is None:
        return None
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", timestamp) is None:
        raise ValueError("verified_at must use UTC YYYY-MM-DDTHH:MM:SSZ form")
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("verified_at must be a valid UTC timestamp") from error
    return timestamp


def _external_evidence_locator(value: object) -> str:
    locator = require_exact_string(value, name="evidence_locator")
    parsed = urlparse(locator)
    if parsed.scheme == "https" and parsed.netloc:
        hostname = parsed.netloc.lower()
        path = parsed.path.lower()
        repository_file = (
            hostname == "raw.githubusercontent.com"
            or (hostname == "github.com" and ("/blob/" in path or "/tree/" in path))
            or (hostname == "api.github.com" and "/contents/" in path)
        )
        if repository_file:
            raise ValueError("repository files cannot serve as external evidence")
        return locator
    if locator.startswith("urn:prob4d:external-audit:") and len(locator) > len(
        "urn:prob4d:external-audit:"
    ):
        return locator
    raise ValueError(
        "evidence_locator must be an external HTTPS URL or prob4d external-audit URN"
    )


def _section_name(value: object) -> SectionName:
    name = require_exact_string(value, name="section_name")
    if name not in _SECTION_ORDER:
        raise ValueError(f"section_name must be one of {list(_SECTION_ORDER)}")
    return cast(SectionName, name)


def _verification_status(value: object) -> VerificationStatus:
    status = require_exact_string(value, name="verification_status")
    if status not in {"unverified", "verified", "failed"}:
        raise ValueError("verification_status is not supported")
    return cast(VerificationStatus, status)


def _notes(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError("notes must be a canonical tuple")
    notes = tuple(
        require_exact_string(item, name=f"notes[{index}]")
        for index, item in enumerate(value)
    )
    if notes != tuple(sorted(notes)) or len(notes) != len(set(notes)):
        raise ValueError("notes must be sorted and unique")
    return notes


def _notes_from_json(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("notes must be a JSON array")
    return _notes(tuple(value))


def _assertions(
    value: object,
    *,
    section_name: SectionName,
) -> Mapping[str, bool | None]:
    mapping = require_mapping(value, name=f"{section_name} assertions")
    expected = frozenset(_SECTION_ASSERTIONS[section_name])
    require_exact_fields(mapping, expected, name=f"{section_name} assertions")
    normalized: dict[str, bool | None] = {}
    for key in _SECTION_ASSERTIONS[section_name]:
        item = mapping[key]
        if item is not None and type(item) is not bool:
            raise ValueError(f"{section_name} assertion {key!r} must be Boolean or null")
        normalized[key] = item
    return frozen_finite_json_mapping(normalized, name=f"{section_name} assertions")


@dataclass(frozen=True, slots=True)
class TrustedValidationBoundarySectionV1:
    """One independently evidenced administrative or host boundary."""

    section_name: SectionName
    verification_status: VerificationStatus
    verified_by: str | None
    verified_at: str | None
    evidence_method: str | None
    evidence_locator: str | None
    evidence_sha256: str | None
    assertions: Mapping[str, bool | None]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _section_name(self.section_name)
        status = _verification_status(self.verification_status)
        verifier = _strict_optional_string(self.verified_by, name="verified_by")
        timestamp = _verified_at(self.verified_at)
        method = _strict_optional_string(self.evidence_method, name="evidence_method")
        locator = (
            None
            if self.evidence_locator is None
            else _external_evidence_locator(self.evidence_locator)
        )
        digest = (
            None
            if self.evidence_sha256 is None
            else require_sha256(self.evidence_sha256, name="evidence_sha256")
        )
        assertions = _assertions(self.assertions, section_name=name)
        notes = _notes(self.notes)

        evidence_values = (verifier, timestamp, method, locator, digest)
        assertion_values = tuple(assertions[key] for key in _SECTION_ASSERTIONS[name])
        if status == "unverified":
            if any(value is not None for value in evidence_values):
                raise ValueError("unverified sections must not claim external evidence")
            if any(value is not None for value in assertion_values):
                raise ValueError("unverified sections must leave every assertion null")
        else:
            if any(value is None for value in evidence_values):
                raise ValueError(
                    "completed sections require verifier, time, method, locator, and digest"
                )
            assert method is not None
            if method not in _ALLOWED_EVIDENCE_METHODS[name]:
                raise ValueError(
                    f"{name} evidence_method must be one of "
                    f"{list(_ALLOWED_EVIDENCE_METHODS[name])}"
                )
            if any(type(value) is not bool for value in assertion_values):
                raise ValueError("completed sections require every assertion to be Boolean")
            if status == "verified" and not all(cast(bool, value) for value in assertion_values):
                raise ValueError("verified sections require every assertion to pass")
            if status == "failed" and all(cast(bool, value) for value in assertion_values):
                raise ValueError("failed sections require at least one failed assertion")

        object.__setattr__(self, "section_name", name)
        object.__setattr__(self, "verification_status", status)
        object.__setattr__(self, "verified_by", verifier)
        object.__setattr__(self, "verified_at", timestamp)
        object.__setattr__(self, "evidence_method", method)
        object.__setattr__(self, "evidence_locator", locator)
        object.__setattr__(self, "evidence_sha256", digest)
        object.__setattr__(self, "assertions", assertions)
        object.__setattr__(self, "notes", notes)

    @classmethod
    def unverified(cls, section_name: SectionName) -> TrustedValidationBoundarySectionV1:
        return cls(
            section_name=section_name,
            verification_status="unverified",
            verified_by=None,
            verified_at=None,
            evidence_method=None,
            evidence_locator=None,
            evidence_sha256=None,
            assertions={key: None for key in _SECTION_ASSERTIONS[section_name]},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "section_name": self.section_name,
            "verification_status": self.verification_status,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
            "evidence_method": self.evidence_method,
            "evidence_locator": self.evidence_locator,
            "evidence_sha256": self.evidence_sha256,
            "assertions": plain_json(self.assertions),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: object) -> TrustedValidationBoundarySectionV1:
        mapping = require_mapping(value, name="trusted validation boundary section")
        require_exact_fields(
            mapping,
            _SECTION_FIELDS,
            name="trusted validation boundary section",
        )
        return cls(
            section_name=mapping["section_name"],
            verification_status=mapping["verification_status"],
            verified_by=mapping["verified_by"],
            verified_at=mapping["verified_at"],
            evidence_method=mapping["evidence_method"],
            evidence_locator=mapping["evidence_locator"],
            evidence_sha256=mapping["evidence_sha256"],
            assertions=mapping["assertions"],
            notes=_notes_from_json(mapping["notes"]),
        )


@dataclass(frozen=True, slots=True)
class TrustedValidationBoundaryAttestationV1:
    """Content-addressed readiness record for all four external boundaries."""

    repository: str
    environment_name: str
    workflow_path: str
    source_revision: str
    workflow_sha256: str
    github_environment: TrustedValidationBoundarySectionV1
    runner_host: TrustedValidationBoundarySectionV1
    dataset_namespace: TrustedValidationBoundarySectionV1
    exact_head_acceptance: TrustedValidationBoundarySectionV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ready: bool = field(init=False)
    failure_reasons: tuple[str, ...] = field(init=False)
    trusted_validation_boundary_attestation_id: str = field(init=False)

    def __post_init__(self) -> None:
        repository = _repository(self.repository)
        environment_name = require_exact_string(
            self.environment_name,
            name="environment_name",
        )
        if environment_name != TRUSTED_VALIDATION_ENVIRONMENT:
            raise ValueError("environment_name changed from the protected environment")
        workflow_path = require_exact_string(self.workflow_path, name="workflow_path")
        if workflow_path != TRUSTED_VALIDATION_WORKFLOW_PATH:
            raise ValueError("workflow_path changed from the trusted workflow")
        revision = _exact_source_revision(self.source_revision)
        workflow_digest = require_sha256(self.workflow_sha256, name="workflow_sha256")

        sections = {
            "github-environment": self.github_environment,
            "runner-host": self.runner_host,
            "dataset-namespace": self.dataset_namespace,
            "exact-head-acceptance": self.exact_head_acceptance,
        }
        for expected_name, section in sections.items():
            if not isinstance(section, TrustedValidationBoundarySectionV1):
                raise TypeError(f"{expected_name} section has the wrong type")
            if section.section_name != expected_name:
                raise ValueError(f"{expected_name} section name changed")

        reasons: list[str] = []
        for section_name in _SECTION_ORDER:
            section = sections[section_name]
            if section.verification_status == "unverified":
                reasons.append(f"{section_name}:unverified")
            elif section.verification_status == "failed":
                reasons.extend(
                    f"{section_name}:{key}"
                    for key in _SECTION_ASSERTIONS[section_name]
                    if section.assertions[key] is False
                )
        failure_reasons = tuple(sorted(reasons))
        ready = not failure_reasons
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="trusted validation attestation metadata",
        )

        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "environment_name", environment_name)
        object.__setattr__(self, "workflow_path", workflow_path)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "workflow_sha256", workflow_digest)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "ready", ready)
        object.__setattr__(self, "failure_reasons", failure_reasons)
        object.__setattr__(
            self,
            "trusted_validation_boundary_attestation_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": TRUSTED_VALIDATION_ATTESTATION_SCHEMA,
            "schema_version": TRUSTED_VALIDATION_ATTESTATION_VERSION,
            "repository": self.repository,
            "environment_name": self.environment_name,
            "workflow_path": self.workflow_path,
            "source_revision": self.source_revision,
            "workflow_sha256": self.workflow_sha256,
            "github_environment": self.github_environment.to_dict(),
            "runner_host": self.runner_host.to_dict(),
            "dataset_namespace": self.dataset_namespace.to_dict(),
            "exact_head_acceptance": self.exact_head_acceptance.to_dict(),
            "metadata": plain_json(self.metadata),
            "ready": self.ready,
            "failure_reasons": list(self.failure_reasons),
            "claim_boundary": TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["trusted_validation_boundary_attestation_id"] = (
            self.trusted_validation_boundary_attestation_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> TrustedValidationBoundaryAttestationV1:
        mapping = require_mapping(value, name="trusted validation boundary attestation")
        require_exact_fields(
            mapping,
            _ATTESTATION_FIELDS,
            name="trusted validation boundary attestation",
        )
        if mapping["schema"] != TRUSTED_VALIDATION_ATTESTATION_SCHEMA:
            raise ValueError("trusted validation attestation schema changed")
        if mapping["schema_version"] != TRUSTED_VALIDATION_ATTESTATION_VERSION:
            raise ValueError("trusted validation attestation version changed")
        if mapping["claim_boundary"] != TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY:
            raise ValueError("trusted validation attestation claim boundary changed")
        result = cls(
            repository=mapping["repository"],
            environment_name=mapping["environment_name"],
            workflow_path=mapping["workflow_path"],
            source_revision=mapping["source_revision"],
            workflow_sha256=mapping["workflow_sha256"],
            github_environment=TrustedValidationBoundarySectionV1.from_dict(
                mapping["github_environment"]
            ),
            runner_host=TrustedValidationBoundarySectionV1.from_dict(
                mapping["runner_host"]
            ),
            dataset_namespace=TrustedValidationBoundarySectionV1.from_dict(
                mapping["dataset_namespace"]
            ),
            exact_head_acceptance=TrustedValidationBoundarySectionV1.from_dict(
                mapping["exact_head_acceptance"]
            ),
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="trusted validation attestation metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("trusted validation attestation derived fields changed")
        return result


def build_trusted_validation_attestation_draft(
    *,
    source_revision: str,
    workflow_sha256: str,
    repository: str = "IPS-Stuttgart/Prob4D",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Return an explicitly unverified draft that cannot claim operational readiness."""

    return {
        "schema": TRUSTED_VALIDATION_ATTESTATION_DRAFT_SCHEMA,
        "schema_version": TRUSTED_VALIDATION_ATTESTATION_VERSION,
        "repository": _repository(repository),
        "environment_name": TRUSTED_VALIDATION_ENVIRONMENT,
        "workflow_path": TRUSTED_VALIDATION_WORKFLOW_PATH,
        "source_revision": _exact_source_revision(source_revision),
        "workflow_sha256": require_sha256(workflow_sha256, name="workflow_sha256"),
        "github_environment": TrustedValidationBoundarySectionV1.unverified(
            "github-environment"
        ).to_dict(),
        "runner_host": TrustedValidationBoundarySectionV1.unverified(
            "runner-host"
        ).to_dict(),
        "dataset_namespace": TrustedValidationBoundarySectionV1.unverified(
            "dataset-namespace"
        ).to_dict(),
        "exact_head_acceptance": TrustedValidationBoundarySectionV1.unverified(
            "exact-head-acceptance"
        ).to_dict(),
        "metadata": plain_json(
            frozen_finite_json_mapping(
                {} if metadata is None else metadata,
                name="trusted validation draft metadata",
            )
        ),
        "claim_boundary": TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY,
    }


def seal_trusted_validation_attestation_draft(
    value: object,
) -> TrustedValidationBoundaryAttestationV1:
    """Validate a completed draft and derive readiness, reasons, and content identity."""

    mapping = require_mapping(value, name="trusted validation attestation draft")
    require_exact_fields(mapping, _DRAFT_FIELDS, name="trusted validation attestation draft")
    if mapping["schema"] != TRUSTED_VALIDATION_ATTESTATION_DRAFT_SCHEMA:
        raise ValueError("trusted validation attestation draft schema changed")
    if mapping["schema_version"] != TRUSTED_VALIDATION_ATTESTATION_VERSION:
        raise ValueError("trusted validation attestation draft version changed")
    if mapping["claim_boundary"] != TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY:
        raise ValueError("trusted validation attestation draft claim boundary changed")
    return TrustedValidationBoundaryAttestationV1(
        repository=mapping["repository"],
        environment_name=mapping["environment_name"],
        workflow_path=mapping["workflow_path"],
        source_revision=mapping["source_revision"],
        workflow_sha256=mapping["workflow_sha256"],
        github_environment=TrustedValidationBoundarySectionV1.from_dict(
            mapping["github_environment"]
        ),
        runner_host=TrustedValidationBoundarySectionV1.from_dict(mapping["runner_host"]),
        dataset_namespace=TrustedValidationBoundarySectionV1.from_dict(
            mapping["dataset_namespace"]
        ),
        exact_head_acceptance=TrustedValidationBoundarySectionV1.from_dict(
            mapping["exact_head_acceptance"]
        ),
        metadata=require_finite_json_mapping(
            mapping["metadata"],
            name="trusted validation draft metadata",
        ),
    )


def _write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def write_trusted_validation_attestation_draft(
    path: str | Path,
    draft: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    seal_trusted_validation_attestation_draft(draft)
    _write_json(path, draft, overwrite=overwrite)


def load_trusted_validation_attestation_draft(path: str | Path) -> dict[str, Any]:
    draft = load_json_object(path, name="trusted validation attestation draft")
    seal_trusted_validation_attestation_draft(draft)
    return draft


def write_trusted_validation_attestation(
    path: str | Path,
    attestation: TrustedValidationBoundaryAttestationV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(attestation, TrustedValidationBoundaryAttestationV1):
        raise TypeError("attestation must be TrustedValidationBoundaryAttestationV1")
    _write_json(path, attestation.to_dict(), overwrite=overwrite)


def load_trusted_validation_attestation(
    path: str | Path,
) -> TrustedValidationBoundaryAttestationV1:
    return TrustedValidationBoundaryAttestationV1.from_dict(
        load_json_object(path, name="trusted validation boundary attestation")
    )


def _summary(value: Mapping[str, Any]) -> None:
    print(json.dumps(plain_json(value), sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template")
    template.add_argument("--source-revision", required=True)
    template.add_argument("--workflow-sha256", required=True)
    template.add_argument("--repository", default="IPS-Stuttgart/Prob4D")
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--overwrite", action="store_true")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "template":
        draft = build_trusted_validation_attestation_draft(
            source_revision=arguments.source_revision,
            workflow_sha256=arguments.workflow_sha256,
            repository=arguments.repository,
        )
        _write_json(arguments.output, draft, overwrite=arguments.overwrite)
        _summary({"draft": str(arguments.output), "ready": False})
        return 0
    if arguments.command == "seal":
        draft = load_json_object(
            arguments.draft,
            name="trusted validation attestation draft",
        )
        attestation = seal_trusted_validation_attestation_draft(draft)
        write_trusted_validation_attestation(
            arguments.output,
            attestation,
            overwrite=arguments.overwrite,
        )
        _summary(
            {
                "attestation_id": attestation.trusted_validation_boundary_attestation_id,
                "failure_reasons": list(attestation.failure_reasons),
                "ready": attestation.ready,
            }
        )
        return 0
    attestation = load_trusted_validation_attestation(arguments.artifact)
    _summary(
        {
            "attestation_id": attestation.trusted_validation_boundary_attestation_id,
            "failure_reasons": list(attestation.failure_reasons),
            "ready": attestation.ready,
        }
    )
    if arguments.require_ready and not attestation.ready:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TRUSTED_VALIDATION_ATTESTATION_CLAIM_BOUNDARY",
    "TRUSTED_VALIDATION_ATTESTATION_DRAFT_SCHEMA",
    "TRUSTED_VALIDATION_ATTESTATION_SCHEMA",
    "TRUSTED_VALIDATION_ATTESTATION_VERSION",
    "TRUSTED_VALIDATION_ENVIRONMENT",
    "TRUSTED_VALIDATION_WORKFLOW_PATH",
    "TrustedValidationBoundaryAttestationV1",
    "TrustedValidationBoundarySectionV1",
    "build_trusted_validation_attestation_draft",
    "load_trusted_validation_attestation",
    "load_trusted_validation_attestation_draft",
    "seal_trusted_validation_attestation_draft",
    "write_trusted_validation_attestation",
    "write_trusted_validation_attestation_draft",
]
