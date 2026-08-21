"""Content-addressed common-support lock for CUT3R source competence v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ._cut3r_source_competence_v2_common import (
    _LOCK_FIELDS,
    _LOCK_SPEC_FIELDS,
    CLAIM_BOUNDARY,
    LOCK_SCHEMA,
    PROPER_SCORE_SEMANTICS,
    VERSION,
    WEIGHTING,
    _paired_policy,
    _source_group_ids,
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
from .source_provider_competence import SourceProviderCompetencePolicyV1


def build_cut3r_source_competence_v2_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    specification: Any,
) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_lock = validate_cut3r_source_competence_lock(
        comparison,
        source_competence_lock,
    )
    spec = _strict_mapping(specification, name="source competence v2 specification")
    _exact_keys(spec, _LOCK_SPEC_FIELDS, name="source competence v2 specification")
    policy = SourceProviderCompetencePolicyV1.from_dict(
        spec["source_competence_policy"]
    )
    if policy.to_dict() != source_lock["policy"]:
        raise ValueError("v2 specification does not match the frozen v1 source policy")
    semantics = _strict_string(
        spec["proper_score_semantics"],
        name="proper_score_semantics",
    )
    if semantics != PROPER_SCORE_SEMANTICS:
        raise ValueError("v2 proper score must be arm-neutral and fixed-scale")
    complete = _strict_boolean(
        spec["require_complete_source_roster"],
        name="require_complete_source_roster",
    )
    group_ids = _source_group_ids(source_lock)
    if complete and (
        policy.minimum_evaluable_groups != len(group_ids)
        or policy.maximum_technical_failures != 0
    ):
        raise ValueError(
            "complete source roster requires every frozen group and zero technical failures"
        )
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": _sha256(
            spec["common_support_definition_sha256"],
            name="common_support_definition_sha256",
        ),
        "source_evaluation_group_ids": group_ids,
        "random_seeds": list(source_lock["random_seeds"]),
        "contrast": dict(source_lock["contrast"]),
        "source_competence_policy": policy.to_dict(),
        "proper_score_semantics": semantics,
        "paired_policy": _paired_policy(spec["paired_policy"]),
        "require_complete_source_roster": complete,
        "weighting": dict(WEIGHTING),
        "source_truth_required": True,
        "target_access": "forbidden",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["common_support_lock_id"] = _record_id(payload)
    return validate_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        payload,
    )


def validate_cut3r_source_competence_v2_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    value: Any,
) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_lock = validate_cut3r_source_competence_lock(
        comparison,
        source_competence_lock,
    )
    payload = _strict_mapping(value, name="source competence v2 lock")
    _exact_keys(payload, _LOCK_FIELDS, name="source competence v2 lock")
    if payload["schema"] != LOCK_SCHEMA or payload["schema_version"] != VERSION:
        raise ValueError("unsupported source competence v2 lock")
    spec = {
        "source_competence_policy": payload["source_competence_policy"],
        "common_support_definition_sha256": payload[
            "common_support_definition_sha256"
        ],
        "proper_score_semantics": payload["proper_score_semantics"],
        "paired_policy": payload["paired_policy"],
        "require_complete_source_roster": payload[
            "require_complete_source_roster"
        ],
    }
    expected = build_cut3r_source_competence_v2_lock_unchecked(
        comparison,
        source_lock,
        spec,
    )
    if payload != expected:
        raise ValueError("source competence v2 lock changed from its frozen inputs")
    return cast(dict[str, Any], json.loads(_canonical_json(expected)))


def build_cut3r_source_competence_v2_lock_unchecked(
    comparison: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Internal nonrecursive lock reconstruction after outer validation."""

    policy = SourceProviderCompetencePolicyV1.from_dict(
        specification["source_competence_policy"]
    )
    if policy.to_dict() != source_lock["policy"]:
        raise ValueError("v2 source policy changed")
    semantics = _strict_string(
        specification["proper_score_semantics"],
        name="proper_score_semantics",
    )
    if semantics != PROPER_SCORE_SEMANTICS:
        raise ValueError("v2 proper score semantics changed")
    complete = _strict_boolean(
        specification["require_complete_source_roster"],
        name="require_complete_source_roster",
    )
    group_ids = _source_group_ids(source_lock)
    if complete and (
        policy.minimum_evaluable_groups != len(group_ids)
        or policy.maximum_technical_failures != 0
    ):
        raise ValueError("complete source roster policy changed")
    result: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": _sha256(
            specification["common_support_definition_sha256"],
            name="common_support_definition_sha256",
        ),
        "source_evaluation_group_ids": group_ids,
        "random_seeds": list(source_lock["random_seeds"]),
        "contrast": dict(source_lock["contrast"]),
        "source_competence_policy": policy.to_dict(),
        "proper_score_semantics": semantics,
        "paired_policy": _paired_policy(specification["paired_policy"]),
        "require_complete_source_roster": complete,
        "weighting": dict(WEIGHTING),
        "source_truth_required": True,
        "target_access": "forbidden",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["common_support_lock_id"] = _record_id(result)
    return result
