"""Content-addressed evidence cards for held-out promotion reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._heldout_promotion_common import (
    _SHA256,
    REPORT_CLAIM_BOUNDARY,
    _load_json,
    _repository,
    _revision,
)
from ._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV2,
    ProviderPromotionIdentityV1,
    promotion_lock_from_dict,
)
from ._heldout_promotion_report import promotion_report_from_dict
from ._immutable_json import plain_json
from ._selection_evidence_common import (
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_real,
    _strict_string,
)

PROMOTION_EVIDENCE_CARD_SCHEMA = "prob4d.heldout-provider-evidence-card"
PROMOTION_EVIDENCE_CARD_VERSION = 1
PROMOTION_EVIDENCE_CARD_V2_VERSION = 2

_NON_CLAIMS = (
    "No Causal4D intervention-benefit claim.",
    "No general state-of-the-art claim.",
    "No extrapolation beyond frozen objects, sessions, and source identities.",
)
_ALLOWED_ARM_ROLES = frozenset(
    {
        "physical_fallback",
        "visual_baseline",
        "rowwise_gauge_marginalized",
        "framewise_explicit_joint_gauge",
        "persistent_explicit_joint_gauge",
        "cross_window_identity_marginalized",
        "sensor_assisted",
        "diagnostic",
    }
)
_REQUIRED_ARM_ROLES = _ALLOWED_ARM_ROLES - {"diagnostic"}
_REQUIRED_FROZEN_ARTIFACTS = frozenset(
    {
        "provider_configuration",
        "gauge_calibration",
        "point_calibration",
        "source_reliability_calibration",
        "material_identity_calibration",
        "selection_lock",
        "bayesian_guard_configuration",
    }
)
_FIELDS = {
    "schema_name",
    "schema_version",
    "evidence_card_id",
    "experiment_id",
    "promotion_lock_id",
    "promotion_report_id",
    "overall_passed",
    "status",
    "statistical_unit",
    "repositories",
    "cohort",
    "comparison_arms",
    "frozen_inputs",
    "provider_gate",
    "guarded_query",
    "claim_boundary",
    "explicit_non_claims",
}
_REPOSITORY_GROUP_FIELDS_V1 = {"prob4d", "bayesian_phystwin", "motioncrafter"}
_REPOSITORY_GROUP_FIELDS_V2 = {"prob4d", "bayesian_phystwin", "provider"}
_CODE_REPOSITORY_FIELDS = {"repository", "revision"}
_MOTIONCRAFTER_FIELDS = {"revision", "model_set_id"}
_COHORT_FIELDS = {
    "development_group_count",
    "calibration_group_count",
    "target_group_count",
    "target_group_ids",
}
_ARM_FIELDS = {
    "arm_id",
    "role",
    "provider_method_id",
    "query_method_id",
    "sensor_assisted",
}
_FROZEN_INPUT_FIELDS = {
    "prediction_run_spec_id",
    "provider_evaluation_manifest_sha256",
    "frozen_artifact_ids",
    "provider_report_sha256",
    "query_results_id",
}
_PROVIDER_GATE_FIELDS = {
    "passed",
    "reference_method",
    "case_count",
    "group_count",
    "decision_policy_id",
}
_GUARDED_QUERY_FIELDS = {
    "passed",
    "primary_arm_id",
    "physical_fallback_arm_id",
    "paired_candidate_minus_fallback_mm",
    "mean_query_rmse_mm",
    "accepted_update_count",
    "rejected_update_count",
    "harmful_accepted_update_count",
    "technical_failure_count",
    "worst_group_regression_mm",
    "mean_accepted_coverage",
    "mean_accepted_width_mm",
    "exact_fallback_failure_count",
}
_PAIRED_EFFECT_FIELDS = {
    "mean",
    "ci95_lower",
    "ci95_upper",
    "group_count",
    "semantics",
}
_PAIRED_EFFECT_SEMANTICS = "paired-target-group-bootstrap-candidate-minus-physical-fallback-v1"


def _optional_string(value: Any, *, name: str) -> str | None:
    return None if value is None else _strict_string(value, name=name)


def _optional_real(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    result = _strict_real(value, name=name)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _canonical_string_list(
    value: Any,
    *,
    name: str,
    nonempty: bool = True,
) -> list[str]:
    items = _strict_list(value, name=name)
    result = [_strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(items)]
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != sorted(result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    return _strict_integer(value, name=name, minimum=0)


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _digest(value: Any, *, name: str) -> str:
    return _strict_digest(value, name=name, pattern=_SHA256)


def _optional_policy_id(value: Any) -> str | None:
    return _optional_string(value, name="provider_gate.decision_policy_id")


def _validate_artifact_ids(value: Any) -> dict[str, str]:
    mapping = _strict_mapping(value, name="frozen_inputs.frozen_artifact_ids")
    if not mapping:
        raise ValueError("frozen_inputs.frozen_artifact_ids must not be empty")
    result: dict[str, str] = {}
    for raw_key, raw_digest in mapping.items():
        key = _strict_string(
            raw_key,
            name="frozen_inputs.frozen_artifact_ids key",
        )
        result[key] = _digest(
            raw_digest,
            name=f"frozen_inputs.frozen_artifact_ids[{key!r}]",
        )
    missing = sorted(_REQUIRED_FROZEN_ARTIFACTS - set(result))
    if missing:
        raise ValueError(
            f"frozen_inputs.frozen_artifact_ids is missing required identities: {missing}"
        )
    return result


def _validate_repositories(value: Any, *, schema_version: int) -> None:
    repositories = _strict_mapping(value, name="repositories")
    expected_fields = (
        _REPOSITORY_GROUP_FIELDS_V1
        if schema_version == PROMOTION_EVIDENCE_CARD_VERSION
        else _REPOSITORY_GROUP_FIELDS_V2
    )
    _exact_keys(repositories, expected_fields, name="repositories")
    for key in ("prob4d", "bayesian_phystwin"):
        repository = _strict_mapping(
            repositories[key],
            name=f"repositories.{key}",
        )
        _exact_keys(
            repository,
            _CODE_REPOSITORY_FIELDS,
            name=f"repositories.{key}",
        )
        _repository(
            repository["repository"],
            name=f"repositories.{key}.repository",
        )
        _revision(
            repository["revision"],
            name=f"repositories.{key}.revision",
        )
    if schema_version == PROMOTION_EVIDENCE_CARD_VERSION:
        motioncrafter = _strict_mapping(
            repositories["motioncrafter"],
            name="repositories.motioncrafter",
        )
        _exact_keys(
            motioncrafter,
            _MOTIONCRAFTER_FIELDS,
            name="repositories.motioncrafter",
        )
        _revision(
            motioncrafter["revision"],
            name="repositories.motioncrafter.revision",
        )
        _digest(
            motioncrafter["model_set_id"],
            name="repositories.motioncrafter.model_set_id",
        )
        return
    ProviderPromotionIdentityV1.from_dict(repositories["provider"])


def _validate_cohort(value: Any) -> tuple[int, list[str]]:
    cohort = _strict_mapping(value, name="cohort")
    _exact_keys(cohort, _COHORT_FIELDS, name="cohort")
    _strict_integer(
        cohort["development_group_count"],
        name="cohort.development_group_count",
        minimum=1,
    )
    _strict_integer(
        cohort["calibration_group_count"],
        name="cohort.calibration_group_count",
        minimum=1,
    )
    target_count = _strict_integer(
        cohort["target_group_count"],
        name="cohort.target_group_count",
        minimum=1,
    )
    target_ids = _canonical_string_list(
        cohort["target_group_ids"],
        name="cohort.target_group_ids",
    )
    if target_count != len(target_ids):
        raise ValueError("cohort.target_group_count must match cohort.target_group_ids")
    return target_count, target_ids


def _validate_arms(value: Any) -> dict[str, Mapping[str, Any]]:
    raw_arms = _strict_list(value, name="comparison_arms")
    if not raw_arms:
        raise ValueError("comparison_arms must not be empty")
    arms: list[Mapping[str, Any]] = []
    for index, raw_arm in enumerate(raw_arms):
        arm = _strict_mapping(raw_arm, name=f"comparison_arms[{index}]")
        _exact_keys(
            arm,
            _ARM_FIELDS,
            name=f"comparison_arms[{index}]",
        )
        _strict_string(
            arm["arm_id"],
            name=f"comparison_arms[{index}].arm_id",
        )
        role = _strict_string(
            arm["role"],
            name=f"comparison_arms[{index}].role",
        )
        if role not in _ALLOWED_ARM_ROLES:
            raise ValueError(f"comparison_arms[{index}].role is not registered")
        provider_method = _optional_string(
            arm["provider_method_id"],
            name=f"comparison_arms[{index}].provider_method_id",
        )
        _strict_string(
            arm["query_method_id"],
            name=f"comparison_arms[{index}].query_method_id",
        )
        sensor_assisted = _strict_bool(
            arm["sensor_assisted"],
            name=f"comparison_arms[{index}].sensor_assisted",
        )
        if role == "physical_fallback":
            if provider_method is not None:
                raise ValueError("the physical fallback arm must not have a provider method")
        elif provider_method is None:
            raise ValueError("every non-fallback comparison arm requires a provider method")
        if sensor_assisted != (role == "sensor_assisted"):
            raise ValueError("sensor_assisted must be true exactly for the sensor-assisted role")
        arms.append(arm)
    arm_ids = [str(arm["arm_id"]) for arm in arms]
    if arm_ids != sorted(arm_ids) or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("comparison_arms must be sorted by unique arm_id")
    roles = [str(arm["role"]) for arm in arms]
    for role in _REQUIRED_ARM_ROLES:
        if roles.count(role) != 1:
            raise ValueError(f"required comparison role {role!r} must occur exactly once")
    provider_methods = [
        str(arm["provider_method_id"]) for arm in arms if arm["provider_method_id"] is not None
    ]
    query_methods = [str(arm["query_method_id"]) for arm in arms]
    if len(provider_methods) != len(set(provider_methods)):
        raise ValueError("comparison provider methods must be unique")
    if len(query_methods) != len(set(query_methods)):
        raise ValueError("comparison query methods must be unique")
    return {str(arm["arm_id"]): arm for arm in arms}


def _validate_frozen_inputs(value: Any) -> None:
    frozen = _strict_mapping(value, name="frozen_inputs")
    _exact_keys(
        frozen,
        _FROZEN_INPUT_FIELDS,
        name="frozen_inputs",
    )
    for field_name in (
        "prediction_run_spec_id",
        "provider_evaluation_manifest_sha256",
        "provider_report_sha256",
        "query_results_id",
    ):
        _digest(
            frozen[field_name],
            name=f"frozen_inputs.{field_name}",
        )
    _validate_artifact_ids(frozen["frozen_artifact_ids"])


def _validate_provider_gate(
    value: Any,
    *,
    target_count: int,
    arms_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    provider = _strict_mapping(value, name="provider_gate")
    _exact_keys(
        provider,
        _PROVIDER_GATE_FIELDS,
        name="provider_gate",
    )
    passed = _strict_bool(provider["passed"], name="provider_gate.passed")
    reference_method = _strict_string(
        provider["reference_method"],
        name="provider_gate.reference_method",
    )
    case_count = _strict_integer(
        provider["case_count"],
        name="provider_gate.case_count",
        minimum=1,
    )
    group_count = _strict_integer(
        provider["group_count"],
        name="provider_gate.group_count",
        minimum=1,
    )
    _optional_policy_id(provider["decision_policy_id"])
    if group_count != target_count:
        raise ValueError("provider_gate.group_count must match cohort.target_group_count")
    if case_count < group_count:
        raise ValueError("provider_gate.case_count cannot be smaller than its group count")
    registered_provider_methods = {
        arm["provider_method_id"]
        for arm in arms_by_id.values()
        if arm["provider_method_id"] is not None
    }
    if reference_method not in registered_provider_methods:
        raise ValueError("provider_gate.reference_method is not a registered provider method")
    return passed


def _validate_guarded_query(
    value: Any,
    *,
    target_count: int,
    arms_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    query = _strict_mapping(value, name="guarded_query")
    _exact_keys(query, _GUARDED_QUERY_FIELDS, name="guarded_query")
    passed = _strict_bool(query["passed"], name="guarded_query.passed")
    primary_id = _strict_string(
        query["primary_arm_id"],
        name="guarded_query.primary_arm_id",
    )
    fallback_id = _strict_string(
        query["physical_fallback_arm_id"],
        name="guarded_query.physical_fallback_arm_id",
    )
    if primary_id not in arms_by_id:
        raise ValueError("guarded_query.primary_arm_id is not registered")
    if fallback_id not in arms_by_id:
        raise ValueError("guarded_query.physical_fallback_arm_id is not registered")
    primary = arms_by_id[primary_id]
    fallback = arms_by_id[fallback_id]
    if fallback["role"] != "physical_fallback":
        raise ValueError("guarded_query.physical_fallback_arm_id has the wrong role")
    if primary["role"] in {
        "physical_fallback",
        "sensor_assisted",
        "diagnostic",
    }:
        raise ValueError("guarded_query.primary_arm_id must be a non-sensor candidate")

    paired = _strict_mapping(
        query["paired_candidate_minus_fallback_mm"],
        name="guarded_query.paired_candidate_minus_fallback_mm",
    )
    _exact_keys(
        paired,
        _PAIRED_EFFECT_FIELDS,
        name="guarded_query.paired_candidate_minus_fallback_mm",
    )
    _strict_real(
        paired["mean"],
        name="guarded_query.paired_candidate_minus_fallback_mm.mean",
    )
    lower = _strict_real(
        paired["ci95_lower"],
        name="guarded_query.paired_candidate_minus_fallback_mm.ci95_lower",
    )
    upper = _strict_real(
        paired["ci95_upper"],
        name="guarded_query.paired_candidate_minus_fallback_mm.ci95_upper",
    )
    if lower > upper:
        raise ValueError("guarded_query paired interval lower bound exceeds its upper bound")
    paired_count = _strict_integer(
        paired["group_count"],
        name="guarded_query.paired_candidate_minus_fallback_mm.group_count",
        minimum=1,
    )
    if paired_count != target_count:
        raise ValueError("guarded_query paired group count must match the target cohort")
    semantics = _strict_string(
        paired["semantics"],
        name="guarded_query.paired_candidate_minus_fallback_mm.semantics",
    )
    if semantics != _PAIRED_EFFECT_SEMANTICS:
        raise ValueError("guarded_query paired-effect semantics changed")

    _nonnegative_real(
        query["mean_query_rmse_mm"],
        name="guarded_query.mean_query_rmse_mm",
    )
    accepted_count = _nonnegative_integer(
        query["accepted_update_count"],
        name="guarded_query.accepted_update_count",
    )
    rejected_count = _nonnegative_integer(
        query["rejected_update_count"],
        name="guarded_query.rejected_update_count",
    )
    if accepted_count + rejected_count != target_count:
        raise ValueError("guarded_query accepted and rejected counts must cover the target cohort")
    harmful_count = _nonnegative_integer(
        query["harmful_accepted_update_count"],
        name="guarded_query.harmful_accepted_update_count",
    )
    if harmful_count > accepted_count:
        raise ValueError("guarded_query harmful accepted updates exceed accepted updates")
    technical_count = _nonnegative_integer(
        query["technical_failure_count"],
        name="guarded_query.technical_failure_count",
    )
    if technical_count > rejected_count:
        raise ValueError("guarded_query technical failures exceed rejected updates")
    _strict_real(
        query["worst_group_regression_mm"],
        name="guarded_query.worst_group_regression_mm",
    )
    _optional_real(
        query["mean_accepted_coverage"],
        name="guarded_query.mean_accepted_coverage",
        minimum=0.0,
        maximum=1.0,
    )
    _optional_real(
        query["mean_accepted_width_mm"],
        name="guarded_query.mean_accepted_width_mm",
        minimum=0.0,
    )
    _nonnegative_integer(
        query["exact_fallback_failure_count"],
        name="guarded_query.exact_fallback_failure_count",
    )
    return passed


def _validate_descriptor(card: Mapping[str, Any]) -> None:
    if card["schema_name"] != PROMOTION_EVIDENCE_CARD_SCHEMA:
        raise ValueError("unsupported promotion evidence card schema")
    schema_version = _strict_integer(
        card["schema_version"],
        name="schema_version",
        minimum=1,
    )
    if schema_version not in {
        PROMOTION_EVIDENCE_CARD_VERSION,
        PROMOTION_EVIDENCE_CARD_V2_VERSION,
    }:
        raise ValueError("unsupported promotion evidence card version")
    _strict_string(card["experiment_id"], name="experiment_id")
    _digest(card["promotion_lock_id"], name="promotion_lock_id")
    _digest(card["promotion_report_id"], name="promotion_report_id")
    overall_passed = _strict_bool(
        card["overall_passed"],
        name="overall_passed",
    )
    status = _strict_string(card["status"], name="status")
    expected_status = "PASS" if overall_passed else "FAIL"
    if status != expected_status:
        raise ValueError("promotion evidence status disagrees with overall_passed")
    _strict_string(card["statistical_unit"], name="statistical_unit")
    _validate_repositories(card["repositories"], schema_version=schema_version)
    target_count, _ = _validate_cohort(card["cohort"])
    arms_by_id = _validate_arms(card["comparison_arms"])
    _validate_frozen_inputs(card["frozen_inputs"])
    provider_passed = _validate_provider_gate(
        card["provider_gate"],
        target_count=target_count,
        arms_by_id=arms_by_id,
    )
    query_passed = _validate_guarded_query(
        card["guarded_query"],
        target_count=target_count,
        arms_by_id=arms_by_id,
    )
    if overall_passed != (provider_passed and query_passed):
        raise ValueError("overall_passed must equal the conjunction of provider and query gates")
    if card["claim_boundary"] != REPORT_CLAIM_BOUNDARY:
        raise ValueError("promotion evidence claim boundary changed")
    non_claims = _canonical_string_list(
        card["explicit_non_claims"],
        name="explicit_non_claims",
    )
    if tuple(non_claims) != tuple(sorted(_NON_CLAIMS)):
        raise ValueError("promotion evidence explicit non-claims changed")


def build_promotion_evidence_card(
    promotion_lock: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a compact paper-facing summary from a validated lock and report."""

    lock = promotion_lock_from_dict(promotion_lock)
    report = promotion_report_from_dict(promotion_report)
    if report.promotion_lock_id != lock.promotion_lock_id:
        raise ValueError("promotion report references a different promotion lock")

    audit = _strict_mapping(report.provider_audit, name="provider_audit")
    provider = _strict_mapping(
        report.provider_decision,
        name="provider_decision",
    )
    query = _strict_mapping(report.query_decision, name="query_decision")
    aggregates = _strict_mapping(
        report.query_aggregate,
        name="query_aggregate",
    )
    primary = _strict_mapping(
        aggregates.get(lock.primary_query_arm_id),
        name="primary query aggregate",
    )
    paired = _strict_mapping(
        query.get("paired_bootstrap"),
        name="paired_bootstrap",
    )
    statistical_unit = lock.metadata.get(
        "statistical_unit",
        lock.metadata.get("split_semantics"),
    )
    if type(statistical_unit) is not str or not statistical_unit:
        statistical_unit = "complete physical object or acquisition session"

    repositories: dict[str, object] = {
        "prob4d": {
            "repository": lock.source_repository,
            "revision": lock.source_revision,
        },
        "bayesian_phystwin": {
            "repository": lock.bayesian_phystwin_repository,
            "revision": lock.bayesian_phystwin_revision,
        },
    }
    if isinstance(lock, HeldoutProviderPromotionLockV2):
        evidence_version = PROMOTION_EVIDENCE_CARD_V2_VERSION
        repositories["provider"] = lock.provider_contract.to_dict()
    else:
        evidence_version = PROMOTION_EVIDENCE_CARD_VERSION
        repositories["motioncrafter"] = {
            "revision": lock.motioncrafter_revision,
            "model_set_id": lock.model_set_id,
        }

    descriptor: dict[str, Any] = {
        "schema_name": PROMOTION_EVIDENCE_CARD_SCHEMA,
        "schema_version": evidence_version,
        "experiment_id": lock.experiment_id,
        "promotion_lock_id": lock.promotion_lock_id,
        "promotion_report_id": report.report_id,
        "overall_passed": report.overall_passed,
        "status": "PASS" if report.overall_passed else "FAIL",
        "statistical_unit": statistical_unit,
        "repositories": repositories,
        "cohort": {
            "development_group_count": len(lock.development_group_ids),
            "calibration_group_count": len(lock.calibration_group_ids),
            "target_group_count": len(lock.target_group_ids),
            "target_group_ids": list(lock.target_group_ids),
        },
        "comparison_arms": [
            {
                "arm_id": arm.arm_id,
                "role": arm.role,
                "provider_method_id": arm.provider_method_id,
                "query_method_id": arm.query_method_id,
                "sensor_assisted": arm.sensor_assisted,
            }
            for arm in lock.arms
        ],
        "frozen_inputs": {
            "prediction_run_spec_id": lock.prediction_run_spec_id,
            "provider_evaluation_manifest_sha256": (lock.provider_evaluation_manifest_sha256),
            "frozen_artifact_ids": dict(lock.frozen_artifact_ids),
            "provider_report_sha256": report.provider_report_sha256,
            "query_results_id": report.query_results_id,
        },
        "provider_gate": {
            "passed": provider.get("overall_passed"),
            "reference_method": audit.get("reference_method"),
            "case_count": audit.get("case_count"),
            "group_count": audit.get("group_count"),
            "decision_policy_id": audit.get("decision_policy_id"),
        },
        "guarded_query": {
            "passed": query.get("overall_passed"),
            "primary_arm_id": lock.primary_query_arm_id,
            "physical_fallback_arm_id": query.get("physical_fallback_arm_id"),
            "paired_candidate_minus_fallback_mm": {
                "mean": paired.get("mean"),
                "ci95_lower": paired.get("ci95_lower"),
                "ci95_upper": paired.get("ci95_upper"),
                "group_count": paired.get("group_count"),
                "semantics": paired.get("semantics"),
            },
            "mean_query_rmse_mm": primary.get("mean_query_rmse_mm"),
            "accepted_update_count": primary.get("accepted_update_count"),
            "rejected_update_count": primary.get("rejected_update_count"),
            "harmful_accepted_update_count": primary.get("harmful_accepted_update_count"),
            "technical_failure_count": primary.get("technical_failure_count"),
            "worst_group_regression_mm": primary.get("worst_group_regression_mm"),
            "mean_accepted_coverage": primary.get("mean_accepted_coverage"),
            "mean_accepted_width_mm": primary.get("mean_accepted_width_mm"),
            "exact_fallback_failure_count": query.get("exact_fallback_failure_count"),
        },
        "claim_boundary": REPORT_CLAIM_BOUNDARY,
        "explicit_non_claims": list(sorted(_NON_CLAIMS)),
    }
    card = {**descriptor, "evidence_card_id": _sha256_json(descriptor)}
    return promotion_evidence_card_from_dict(card)


def promotion_evidence_card_from_dict(value: object) -> dict[str, Any]:
    card = _strict_mapping(value, name="promotion evidence card")
    _exact_keys(card, _FIELDS, name="promotion evidence card")
    _validate_descriptor(card)
    supplied = _digest(
        card["evidence_card_id"],
        name="evidence_card_id",
    )
    descriptor = {key: item for key, item in card.items() if key != "evidence_card_id"}
    if supplied != _sha256_json(descriptor):
        raise ValueError("promotion evidence card ID mismatch")
    return dict(card)


def load_promotion_evidence_card(path: str | Path) -> dict[str, Any]:
    mapping, _ = _load_json(
        Path(path),
        name="promotion evidence card",
    )
    return promotion_evidence_card_from_dict(mapping)


def render_promotion_evidence_markdown(card: Mapping[str, Any]) -> str:
    value = promotion_evidence_card_from_dict(card)
    repositories = _strict_mapping(value["repositories"], name="repositories")
    prob4d = _strict_mapping(repositories["prob4d"], name="prob4d")
    bpt = _strict_mapping(
        repositories["bayesian_phystwin"],
        name="bayesian_phystwin",
    )
    if "provider" in repositories:
        provider = ProviderPromotionIdentityV1.from_dict(repositories["provider"])
        provider_line = (
            f"- Provider: `{provider.provider_family}` "
            f"(`{provider.provider_repository}@{provider.provider_revision}`)"
        )
    else:
        motioncrafter = _strict_mapping(
            repositories["motioncrafter"],
            name="motioncrafter",
        )
        provider_line = f"- Provider: `MotionCrafter@{motioncrafter['revision']}`"
    query = _strict_mapping(value["guarded_query"], name="guarded_query")
    paired = _strict_mapping(
        query["paired_candidate_minus_fallback_mm"],
        name="paired effect",
    )
    return "\n".join(
        [
            "# Held-out Prob4D promotion evidence card",
            "",
            f"Evidence card: `{value['evidence_card_id']}`.",
            f"Promotion report: `{value['promotion_report_id']}`.",
            f"Overall decision: **{value['status']}**.",
            "",
            f"- Prob4D: `{prob4d['repository']}@{prob4d['revision']}`",
            f"- BayesianPhysTwin: `{bpt['repository']}@{bpt['revision']}`",
            provider_line,
            "",
            "| Guarded-query result | Value |",
            "| --- | ---: |",
            f"| Primary arm | `{query['primary_arm_id']}` |",
            (f"| Mean candidate-minus-fallback RMSE | {float(paired['mean']):.6g} mm |"),
            (f"| Harmful accepted updates | {query['harmful_accepted_update_count']} |"),
            (f"| Exact fallback failures | {query['exact_fallback_failure_count']} |"),
            "",
            str(value["claim_boundary"]),
            "",
        ]
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish complete text without replacing a concurrently created path."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            plain_json(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write_text(path, encoded)


def write_promotion_evidence_card(
    card: Mapping[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    value = promotion_evidence_card_from_dict(card)
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    if json_destination == markdown_destination:
        raise ValueError("JSON and Markdown evidence-card paths must differ")
    if json_destination.exists():
        raise FileExistsError(json_destination)
    if markdown_destination.exists():
        raise FileExistsError(markdown_destination)
    _atomic_write_json(json_destination, value)
    try:
        _atomic_write_text(
            markdown_destination,
            render_promotion_evidence_markdown(value),
        )
    except Exception:
        json_destination.unlink(missing_ok=True)
        raise


__all__ = [
    "PROMOTION_EVIDENCE_CARD_SCHEMA",
    "PROMOTION_EVIDENCE_CARD_VERSION",
    "PROMOTION_EVIDENCE_CARD_V2_VERSION",
    "build_promotion_evidence_card",
    "load_promotion_evidence_card",
    "promotion_evidence_card_from_dict",
    "render_promotion_evidence_markdown",
    "write_promotion_evidence_card",
]
