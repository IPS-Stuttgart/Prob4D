"""Content-addressed lock for independent CUT3R support auditing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from ._cut3r_source_competence_audit_common import (
    _LOCK_FIELDS,
    _LOCK_SPEC_FIELDS,
    CLAIM_BOUNDARY,
    LOCK_SCHEMA,
    PROPER_SCORE_REFERENCE_FIT_SCOPE,
    PROPER_SCORE_SEMANTICS,
    VERSION,
)
from ._cut3r_source_competence_v2_lock import (
    validate_cut3r_source_competence_v2_lock,
)
from .cut3r_comparison import validate_cut3r_comparison_lock
from .cut3r_source_competence import (
    _canonical_json,
    _exact_keys,
    _record_id,
    _sha256,
    _strict_boolean,
    _strict_mapping,
    _strict_string,
    validate_cut3r_source_competence_lock,
)


def _reference_sha256(reference_bytes: bytes) -> str:
    if type(reference_bytes) is not bytes or not reference_bytes:
        raise ValueError("proper-score reference must contain exact nonempty bytes")
    return hashlib.sha256(reference_bytes).hexdigest()


def _source_group_ids(source_lock: Mapping[str, Any]) -> list[str]:
    return [
        cast(str, group["group_id"])
        for group in cast(list[dict[str, Any]], source_lock["source_evaluation_groups"])
    ]


def build_cut3r_source_competence_audit_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    specification: Any,
    proper_score_reference_bytes: bytes,
) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_lock = validate_cut3r_source_competence_lock(
        comparison,
        source_competence_lock,
    )
    v2_lock = validate_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        common_support_lock,
    )
    spec = _strict_mapping(specification, name="support audit specification")
    _exact_keys(spec, _LOCK_SPEC_FIELDS, name="support audit specification")
    common_support_definition = _sha256(
        spec["common_support_definition_sha256"],
        name="common_support_definition_sha256",
    )
    if common_support_definition != v2_lock["common_support_definition_sha256"]:
        raise ValueError("audit specification uses a different support definition")
    semantics = _strict_string(
        spec["proper_score_semantics"],
        name="proper_score_semantics",
    )
    if semantics != PROPER_SCORE_SEMANTICS or semantics != v2_lock[
        "proper_score_semantics"
    ]:
        raise ValueError("audit requires the frozen arm-neutral proper-score semantics")
    fit_scope = _strict_string(
        spec["proper_score_reference_fit_scope"],
        name="proper_score_reference_fit_scope",
    )
    if fit_scope != PROPER_SCORE_REFERENCE_FIT_SCOPE:
        raise ValueError("proper-score reference must be source/target separated")
    complete = _strict_boolean(
        spec["require_complete_manifest_roster"],
        name="require_complete_manifest_roster",
    )
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": common_support_definition,
        "source_evaluation_group_ids": _source_group_ids(source_lock),
        "random_seeds": list(v2_lock["random_seeds"]),
        "contrast": dict(v2_lock["contrast"]),
        "proper_score_semantics": semantics,
        "proper_score_reference_artifact_id": _sha256(
            spec["proper_score_reference_artifact_id"],
            name="proper_score_reference_artifact_id",
        ),
        "proper_score_reference_sha256": _reference_sha256(
            proper_score_reference_bytes
        ),
        "proper_score_reference_fit_scope": fit_scope,
        "require_complete_manifest_roster": complete,
        "source_truth_required": True,
        "target_access": "forbidden",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["support_audit_lock_id"] = _record_id(payload)
    return validate_cut3r_source_competence_audit_lock(
        comparison,
        source_lock,
        v2_lock,
        payload,
    )


def validate_cut3r_source_competence_audit_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    value: Any,
) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_lock = validate_cut3r_source_competence_lock(
        comparison,
        source_competence_lock,
    )
    v2_lock = validate_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        common_support_lock,
    )
    payload = _strict_mapping(value, name="support audit lock")
    _exact_keys(payload, _LOCK_FIELDS, name="support audit lock")
    if payload["schema"] != LOCK_SCHEMA or payload["schema_version"] != VERSION:
        raise ValueError("unsupported support audit lock")
    expected = {
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": v2_lock[
            "common_support_definition_sha256"
        ],
    }
    for name, expected_value in expected.items():
        if _sha256(payload[name], name=name) != expected_value:
            raise ValueError(f"support audit lock uses a different {name}")
    if payload["source_evaluation_group_ids"] != _source_group_ids(source_lock):
        raise ValueError("support audit lock uses a different source roster")
    if payload["random_seeds"] != list(v2_lock["random_seeds"]):
        raise ValueError("support audit lock uses different random seeds")
    if payload["contrast"] != dict(v2_lock["contrast"]):
        raise ValueError("support audit lock uses a different contrast")
    if payload["proper_score_semantics"] != PROPER_SCORE_SEMANTICS:
        raise ValueError("support audit lock uses different proper-score semantics")
    _sha256(
        payload["proper_score_reference_artifact_id"],
        name="proper_score_reference_artifact_id",
    )
    _sha256(
        payload["proper_score_reference_sha256"],
        name="proper_score_reference_sha256",
    )
    if payload["proper_score_reference_fit_scope"] != PROPER_SCORE_REFERENCE_FIT_SCOPE:
        raise ValueError("support audit lock uses a different reference fit scope")
    _strict_boolean(
        payload["require_complete_manifest_roster"],
        name="require_complete_manifest_roster",
    )
    if payload["source_truth_required"] is not True:
        raise ValueError("support audit lock must require source truth")
    if payload["target_access"] != "forbidden":
        raise ValueError("support audit lock must forbid target access")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("support audit claim boundary changed")
    lock_id = _sha256(payload["support_audit_lock_id"], name="support_audit_lock_id")
    unsigned = dict(payload)
    unsigned.pop("support_audit_lock_id")
    if _record_id(unsigned) != lock_id:
        raise ValueError("support audit lock content identity changed")
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def verify_cut3r_source_competence_audit_reference(
    audit_lock: Mapping[str, Any],
    proper_score_reference_bytes: bytes,
) -> None:
    expected = _sha256(
        audit_lock["proper_score_reference_sha256"],
        name="proper_score_reference_sha256",
    )
    if _reference_sha256(proper_score_reference_bytes) != expected:
        raise ValueError("proper-score reference bytes do not match the frozen audit lock")


__all__ = [
    "build_cut3r_source_competence_audit_lock",
    "validate_cut3r_source_competence_audit_lock",
    "verify_cut3r_source_competence_audit_reference",
]
