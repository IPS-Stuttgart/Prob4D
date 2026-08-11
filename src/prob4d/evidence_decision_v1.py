"""Independent validation of BayesianPhysTwin evidence decisions.

The validator is an independently implemented, generated binding for the
closed version-1 wire schema merged in BayesianPhysTwin. Prob4D can therefore
verify decisions that cite an exact Prob4D revision without adding a reverse
runtime dependency on BayesianPhysTwin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from .project_identity import is_prob4d_repository

EVIDENCE_DECISION_SCHEMA = "bayesian_phystwin.evidence_decision"
EVIDENCE_DECISION_SCHEMA_VERSION = 1
EVIDENCE_DECISION_SOURCE_REPOSITORY = "IPS-Stuttgart/BayesianPhysTwin"
EVIDENCE_DECISION_SOURCE_REVISION = (
    "4ee702f5130cfedbea7bce6be5e72483c92f63da"
)
EVIDENCE_DECISION_JSON_SCHEMA_SHA256 = (
    "d5615258c6cf666d0ed9684a87930989adf91817fe99b0387e83a31479dcd465"
)

DecisionStatus = Literal["pass", "fail", "degraded", "inconclusive"]
RunClassification = Literal[
    "controlled",
    "exploratory",
    "confirmatory",
    "diagnostic",
    "infrastructure",
]
RepositoryRole = Literal[
    "primary",
    "upstream",
    "observation",
    "downstream",
    "paper",
    "environment",
    "dependency",
]

_VALID_STATUSES = frozenset({"pass", "fail", "degraded", "inconclusive"})
_VALID_CLASSIFICATIONS = frozenset(
    {"controlled", "exploratory", "confirmatory", "diagnostic", "infrastructure"}
)
_VALID_ROLES = frozenset(
    {
        "primary",
        "upstream",
        "observation",
        "downstream",
        "paper",
        "environment",
        "dependency",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "decision_id",
        "schema_name",
        "schema_version",
        "created_utc",
        "claim_id",
        "protocol_id",
        "status",
        "run_classification",
        "claim_authorized",
        "evidence_level",
        "metric",
        "run_manifest_id",
        "evidence_fingerprint",
        "evidence_summary_sha256",
        "repositories",
        "limitations",
        "metadata",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "name",
        "comparison",
        "rule",
        "observed_value",
        "threshold_value",
        "unit",
    }
)
_REPOSITORY_FIELDS = frozenset({"repository", "revision", "dirty", "role"})
_GITHUB_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def evidence_decision_contract_identity() -> dict[str, object]:
    """Return the exact upstream schema identity used by this binding."""

    return {
        "schema_name": EVIDENCE_DECISION_SCHEMA,
        "schema_version": EVIDENCE_DECISION_SCHEMA_VERSION,
        "json_schema_sha256": EVIDENCE_DECISION_JSON_SCHEMA_SHA256,
        "source_repository": EVIDENCE_DECISION_SOURCE_REPOSITORY,
        "source_revision": EVIDENCE_DECISION_SOURCE_REVISION,
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _freeze_json(
    value: Any,
    *,
    name: str,
    path: str = "$",
    active: set[int] | None = None,
) -> Any:
    active = set() if active is None else active
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{name} contains a circular mapping at {path}")
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{name} requires string keys at {path}")
                result[key] = _freeze_json(
                    item,
                    name=name,
                    path=f"{path}.{key}",
                    active=active,
                )
            return MappingProxyType({key: result[key] for key in sorted(result)})
        finally:
            active.remove(identity)
    if type(value) in {list, tuple}:
        identity = id(value)
        if identity in active:
            raise ValueError(f"{name} contains a circular sequence at {path}")
        active.add(identity)
        try:
            return tuple(
                _freeze_json(
                    item,
                    name=name,
                    path=f"{path}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)
    raise ValueError(
        f"{name} contains a non-JSON value at {path}: {type(value).__name__}"
    )


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unknown:
        details.append(f"unknown {unknown}")
    raise ValueError(f"{name} does not match schema: {', '.join(details)}")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} requires string keys")
    return cast(Mapping[str, Any], value)


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or value != value.lower()
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    return value


def _require_repository(value: Any, *, name: str) -> str:
    text = _require_text(value, name=name)
    parts = text.split("/")
    if len(parts) != 2:
        raise ValueError(f"{name} must use canonical owner/name form")
    owner, repository = parts
    if _GITHUB_OWNER.fullmatch(owner) is None:
        raise ValueError(f"{name} contains an invalid GitHub owner")
    if (
        _GITHUB_REPOSITORY.fullmatch(repository) is None
        or repository in {".", ".."}
    ):
        raise ValueError(f"{name} contains an invalid GitHub repository")
    return text


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _require_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_created_utc(value: Any) -> str:
    text = _require_text(value, name="created_utc")
    if not (text.endswith("Z") or text.endswith("+00:00")):
        raise ValueError("created_utc must use the contract UTC suffix")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_utc must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("created_utc must be an ISO-8601 UTC timestamp")
    return text


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


@dataclass(frozen=True)
class DecisionMetricV1:
    """One scalar metric and frozen comparison rule."""

    name: str
    comparison: str
    rule: str
    observed_value: float
    threshold_value: float | None
    unit: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "comparison": self.comparison,
            "rule": self.rule,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class DecisionRepositoryStateV1:
    """Exact repository state cited by a decision."""

    repository: str
    revision: str
    dirty: bool
    role: RepositoryRole

    def as_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "dirty": self.dirty,
            "role": self.role,
        }


@dataclass(frozen=True)
class ValidatedEvidenceDecisionV1:
    """Validated, immutable consumer view of one evidence decision."""

    decision_id: str
    created_utc: str
    claim_id: str
    protocol_id: str
    status: DecisionStatus
    run_classification: RunClassification
    claim_authorized: bool
    evidence_level: int
    metric: DecisionMetricV1
    run_manifest_id: str
    evidence_fingerprint: str
    evidence_summary_sha256: str
    repositories: tuple[DecisionRepositoryStateV1, ...]
    limitations: tuple[str, ...]
    metadata: Mapping[str, Any]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": EVIDENCE_DECISION_SCHEMA,
            "schema_version": EVIDENCE_DECISION_SCHEMA_VERSION,
            "created_utc": self.created_utc,
            "claim_id": self.claim_id,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "run_classification": self.run_classification,
            "claim_authorized": self.claim_authorized,
            "evidence_level": self.evidence_level,
            "metric": self.metric.as_dict(),
            "run_manifest_id": self.run_manifest_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "evidence_summary_sha256": self.evidence_summary_sha256,
            "repositories": [
                repository.as_dict() for repository in self.repositories
            ],
            "limitations": list(self.limitations),
            "metadata": _plain_json(self.metadata),
        }

    @property
    def computed_decision_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {"decision_id": self.decision_id, **self.descriptor()}


def _parse_metric(value: Any) -> DecisionMetricV1:
    payload = _require_mapping(value, name="decision metric")
    _require_exact_fields(payload, expected=_METRIC_FIELDS, name="decision metric")
    threshold = payload["threshold_value"]
    return DecisionMetricV1(
        name=_require_text(payload["name"], name="metric name"),
        comparison=_require_text(
            payload["comparison"],
            name="metric comparison",
        ),
        rule=_require_text(payload["rule"], name="metric rule"),
        observed_value=_require_number(
            payload["observed_value"],
            name="observed_value",
        ),
        threshold_value=(
            None
            if threshold is None
            else _require_number(threshold, name="threshold_value")
        ),
        unit=_require_text(payload["unit"], name="metric unit"),
    )


def _parse_repository(value: Any) -> DecisionRepositoryStateV1:
    payload = _require_mapping(value, name="evidence-decision repository")
    _require_exact_fields(
        payload,
        expected=_REPOSITORY_FIELDS,
        name="evidence-decision repository",
    )
    role = payload["role"]
    if type(role) is not str or role not in _VALID_ROLES:
        raise ValueError("unsupported evidence-decision repository role")
    return DecisionRepositoryStateV1(
        repository=_require_repository(payload["repository"], name="repository"),
        revision=_require_revision(
            payload["revision"],
            name="repository revision",
        ),
        dirty=_require_bool(payload["dirty"], name="dirty"),
        role=cast(RepositoryRole, role),
    )


def validate_evidence_decision_v1(
    value: Mapping[str, Any],
) -> ValidatedEvidenceDecisionV1:
    """Validate one in-memory decision without importing BayesianPhysTwin."""

    payload = _require_mapping(value, name="evidence decision")
    _require_exact_fields(payload, expected=_DECISION_FIELDS, name="evidence decision")
    if payload["schema_name"] != EVIDENCE_DECISION_SCHEMA:
        raise ValueError("unsupported evidence-decision schema")
    if (
        _require_int(payload["schema_version"], name="schema_version")
        != EVIDENCE_DECISION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported evidence-decision schema version")

    status = payload["status"]
    if type(status) is not str or status not in _VALID_STATUSES:
        raise ValueError("unsupported evidence decision status")
    classification = payload["run_classification"]
    if type(classification) is not str or classification not in _VALID_CLASSIFICATIONS:
        raise ValueError("unsupported run classification")
    authorized = _require_bool(
        payload["claim_authorized"],
        name="claim_authorized",
    )
    if authorized and status != "pass":
        raise ValueError("only a passing decision can authorize a claim")
    if authorized and classification != "confirmatory":
        raise ValueError(
            "claim authorization requires a confirmatory run classification"
        )
    evidence_level = _require_int(payload["evidence_level"], name="evidence_level")
    if evidence_level not in {1, 2, 3}:
        raise ValueError("evidence_level must be one of 1, 2, or 3")

    repositories = tuple(
        _parse_repository(item)
        for item in _require_sequence(payload["repositories"], name="repositories")
    )
    primary = [
        repository
        for repository in repositories
        if repository.role == "primary"
    ]
    if len(primary) != 1:
        raise ValueError("evidence decision requires exactly one primary repository")
    names = [repository.repository for repository in repositories]
    if len(names) != len(set(names)):
        raise ValueError("evidence-decision repository names must be unique")
    if authorized and any(repository.dirty for repository in repositories):
        raise ValueError("an authorized claim cannot bind a dirty repository")
    normalized_repositories = (
        primary[0],
        *sorted(
            (
                repository
                for repository in repositories
                if repository.role != "primary"
            ),
            key=lambda repository: (repository.role, repository.repository),
        ),
    )

    raw_limitations = _require_sequence(payload["limitations"], name="limitations")
    limitations = tuple(
        _require_text(limitation, name="limitation")
        for limitation in raw_limitations
    )
    if len(limitations) != len(set(limitations)):
        raise ValueError("limitations must be unique")
    if status in {"degraded", "inconclusive"} and not limitations:
        raise ValueError(f"{status} decisions must record at least one limitation")

    decision = ValidatedEvidenceDecisionV1(
        decision_id=_require_sha256(payload["decision_id"], name="decision_id"),
        created_utc=_require_created_utc(payload["created_utc"]),
        claim_id=_require_text(payload["claim_id"], name="claim_id"),
        protocol_id=_require_text(payload["protocol_id"], name="protocol_id"),
        status=cast(DecisionStatus, status),
        run_classification=cast(RunClassification, classification),
        claim_authorized=authorized,
        evidence_level=evidence_level,
        metric=_parse_metric(payload["metric"]),
        run_manifest_id=_require_sha256(
            payload["run_manifest_id"],
            name="run_manifest_id",
        ),
        evidence_fingerprint=_require_sha256(
            payload["evidence_fingerprint"],
            name="evidence_fingerprint",
        ),
        evidence_summary_sha256=_require_sha256(
            payload["evidence_summary_sha256"],
            name="evidence_summary_sha256",
        ),
        repositories=normalized_repositories,
        limitations=limitations,
        metadata=_freeze_json(
            _require_mapping(payload["metadata"], name="metadata"),
            name="metadata",
        ),
    )
    if decision.computed_decision_id != decision.decision_id:
        raise ValueError("evidence decision digest does not match its descriptor")
    return decision


def load_evidence_decision_v1(path: str | Path) -> ValidatedEvidenceDecisionV1:
    """Load strict JSON and reject duplicate keys and non-finite constants."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("evidence decision is not valid JSON") from error
    return validate_evidence_decision_v1(
        _require_mapping(value, name="evidence decision")
    )


def require_authorized_evidence_decision_v1(
    decision: ValidatedEvidenceDecisionV1 | Mapping[str, Any],
    *,
    claim_id: str | None = None,
    protocol_id: str | None = None,
    minimum_evidence_level: int = 1,
) -> ValidatedEvidenceDecisionV1:
    """Require claim authorization and optional exact semantic bindings."""

    validated = (
        decision
        if isinstance(decision, ValidatedEvidenceDecisionV1)
        else validate_evidence_decision_v1(decision)
    )
    if not validated.claim_authorized:
        raise ValueError("evidence decision does not authorize its claim")
    if claim_id is not None and validated.claim_id != _require_text(
        claim_id,
        name="claim_id",
    ):
        raise ValueError("evidence decision claim_id does not match")
    if protocol_id is not None and validated.protocol_id != _require_text(
        protocol_id,
        name="protocol_id",
    ):
        raise ValueError("evidence decision protocol_id does not match")
    level = _require_int(minimum_evidence_level, name="minimum_evidence_level")
    if level not in {1, 2, 3}:
        raise ValueError("minimum_evidence_level must be one of 1, 2, or 3")
    if validated.evidence_level < level:
        raise ValueError("evidence decision does not meet the required evidence level")
    return validated


def require_repository_binding_v1(
    decision: ValidatedEvidenceDecisionV1 | Mapping[str, Any],
    *,
    repository_names: Sequence[str],
    expected_revision: str | None = None,
    allowed_roles: Sequence[RepositoryRole] | None = None,
    require_clean: bool | None = None,
) -> DecisionRepositoryStateV1:
    """Require one exact repository binding from a validated decision."""

    validated = (
        decision
        if isinstance(decision, ValidatedEvidenceDecisionV1)
        else validate_evidence_decision_v1(decision)
    )
    names = tuple(
        _require_repository(name, name="repository name")
        for name in repository_names
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("repository_names must be nonempty and unique")
    matches = [
        repository
        for repository in validated.repositories
        if repository.repository in names
    ]
    if len(matches) != 1:
        raise ValueError("evidence decision must bind exactly one matching repository")
    result = matches[0]
    if expected_revision is not None and result.revision != _require_revision(
        expected_revision,
        name="expected_revision",
    ):
        raise ValueError("evidence decision repository revision does not match")
    if allowed_roles is not None:
        roles = tuple(allowed_roles)
        if not roles or any(role not in _VALID_ROLES for role in roles):
            raise ValueError("allowed_roles contains an unsupported repository role")
        if result.role not in roles:
            raise ValueError("evidence decision repository role is not allowed")
    clean = validated.claim_authorized if require_clean is None else _require_bool(
        require_clean,
        name="require_clean",
    )
    if clean and result.dirty:
        raise ValueError("evidence decision repository binding is dirty")
    return result


def require_prob4d_evidence_binding_v1(
    decision: ValidatedEvidenceDecisionV1 | Mapping[str, Any],
    *,
    expected_revision: str | None = None,
    allowed_roles: Sequence[RepositoryRole] = (
        "observation",
        "upstream",
        "dependency",
    ),
    require_clean: bool | None = None,
) -> DecisionRepositoryStateV1:
    """Require the unique Prob4D state cited by a decision."""

    validated = (
        decision
        if isinstance(decision, ValidatedEvidenceDecisionV1)
        else validate_evidence_decision_v1(decision)
    )
    matches = [
        repository
        for repository in validated.repositories
        if is_prob4d_repository(repository.repository)
    ]
    if len(matches) != 1:
        raise ValueError("evidence decision must bind exactly one Prob4D repository")
    result = matches[0]
    return require_repository_binding_v1(
        validated,
        repository_names=(result.repository,),
        expected_revision=expected_revision,
        allowed_roles=allowed_roles,
        require_clean=require_clean,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate and print one evidence decision."""

    parser = argparse.ArgumentParser(
        description="Validate a BayesianPhysTwin evidence decision in Prob4D."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-prob4d-revision")
    parser.add_argument("--require-authorized", action="store_true")
    arguments = parser.parse_args(argv)

    decision = load_evidence_decision_v1(arguments.path)
    if arguments.require_authorized:
        require_authorized_evidence_decision_v1(decision)
    if arguments.expected_prob4d_revision is not None:
        require_prob4d_evidence_binding_v1(
            decision,
            expected_revision=arguments.expected_prob4d_revision,
        )
    print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DecisionMetricV1",
    "DecisionRepositoryStateV1",
    "DecisionStatus",
    "EVIDENCE_DECISION_JSON_SCHEMA_SHA256",
    "EVIDENCE_DECISION_SCHEMA",
    "EVIDENCE_DECISION_SCHEMA_VERSION",
    "EVIDENCE_DECISION_SOURCE_REPOSITORY",
    "EVIDENCE_DECISION_SOURCE_REVISION",
    "RepositoryRole",
    "RunClassification",
    "ValidatedEvidenceDecisionV1",
    "evidence_decision_contract_identity",
    "load_evidence_decision_v1",
    "main",
    "require_authorized_evidence_decision_v1",
    "require_prob4d_evidence_binding_v1",
    "require_repository_binding_v1",
    "validate_evidence_decision_v1",
]
