"""Replayable provider and guarded-query evaluation for held-out promotion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ._heldout_promotion_common import (
    _SHA256,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_lock import HeldoutProviderPromotionLockV1
from ._heldout_promotion_query import (
    HeldoutPromotionQueryResultsV1,
    PromotionQueryRowV1,
)


def _provider_report_audit(
    lock: HeldoutProviderPromotionLockV1,
    provider_report: Mapping[str, Any],
) -> tuple[dict[str, object], Mapping[str, Any]]:
    if provider_report.get("schema_name") != "prob4d.provider-evaluation-report":
        raise ValueError("provider report has the wrong schema")
    schema_version = provider_report.get("schema_version")
    if type(schema_version) is not int or schema_version < 3:
        raise ValueError("provider report must retain a preregistered decision")
    if provider_report.get("primary_mode") == "oracle_aligned":
        raise ValueError("promotion provider report cannot use oracle alignment")
    if provider_report.get("primary_support") != "common_across_registered_methods":
        raise ValueError("promotion provider report must use common primary support")
    if provider_report.get("legacy_artifacts_allowed") is not False:
        raise ValueError("promotion provider report cannot admit legacy artifacts")
    manifest_sha = _strict_digest(
        provider_report.get("source_manifest_sha256"),
        name="provider source_manifest_sha256",
        pattern=_SHA256,
    )
    if manifest_sha != lock.provider_evaluation_manifest_sha256:
        raise ValueError("provider report uses a different frozen evaluation manifest")
    if provider_report.get("bootstrap_resamples") != lock.bootstrap_resamples:
        raise ValueError("provider report bootstrap resamples differ from the lock")
    if provider_report.get("bootstrap_seed") != lock.bootstrap_seed:
        raise ValueError("provider report bootstrap seed differs from the lock")
    if provider_report.get("reference_method") != lock.provider_reference_method_id:
        raise ValueError("provider report changed the registered reference method")

    raw_records = _strict_list(provider_report.get("cases"), name="provider report cases")
    expected_methods = set(lock.provider_method_ids)
    expected_groups = set(lock.target_group_ids)
    observed_groups: set[str] = set()
    methods_by_case: dict[str, set[str]] = {}
    group_by_case: dict[str, str] = {}
    observed_case_methods: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(raw_records):
        record = _strict_mapping(raw_record, name=f"provider report cases[{index}]")
        case_id = _strict_string(record.get("case_id"), name="provider case_id")
        group_id = _strict_string(record.get("group_id"), name="provider group_id")
        method_id = _strict_string(record.get("method_id"), name="provider method_id")
        case_method = (case_id, method_id)
        if case_method in observed_case_methods:
            raise ValueError(
                "provider report contains a duplicate case/method record: "
                f"{case_method}"
            )
        observed_case_methods.add(case_method)
        previous_group = group_by_case.setdefault(case_id, group_id)
        if previous_group != group_id:
            raise ValueError(
                f"provider case {case_id!r} is assigned to multiple target groups"
            )
        observed_groups.add(group_id)
        methods_by_case.setdefault(case_id, set()).add(method_id)
    if observed_groups != expected_groups:
        raise ValueError("provider report target groups differ from the promotion lock")
    incomplete_cases = sorted(
        case_id
        for case_id, methods in methods_by_case.items()
        if methods != expected_methods
    )
    if incomplete_cases:
        raise ValueError(
            "provider report has incomplete or changed method sets for cases: "
            f"{incomplete_cases}"
        )
    observed_methods = set().union(*methods_by_case.values()) if methods_by_case else set()
    if observed_methods != expected_methods:
        raise ValueError("provider report methods differ from the promotion lock")
    decision = _strict_mapping(provider_report.get("decision"), name="provider decision")
    provider_passed = decision.get("overall_passed") is True
    return (
        {
            "schema_version": schema_version,
            "primary_mode": provider_report.get("primary_mode"),
            "reference_method": provider_report.get("reference_method"),
            "case_count": len(methods_by_case),
            "group_count": len(observed_groups),
            "registered_methods": sorted(observed_methods),
            "bootstrap_resamples": lock.bootstrap_resamples,
            "bootstrap_seed": lock.bootstrap_seed,
            "decision_policy_id": decision.get("policy_id"),
            "decision_passed": provider_passed,
            "structural_validation_passed": True,
        },
        decision,
    )


def _paired_bootstrap(values: np.ndarray, *, resamples: int, seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("paired bootstrap values must be one finite nonempty vector")
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, array.size, size=(resamples, array.size))
    bootstrap_means = np.mean(array[sampled], axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
    return {
        "mean": float(np.mean(array)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "group_count": int(array.size),
        "resamples": resamples,
        "seed": seed,
        "semantics": "paired-target-group-bootstrap-candidate-minus-physical-fallback-v1",
    }


def _query_evaluation(
    lock: HeldoutProviderPromotionLockV1,
    results: HeldoutPromotionQueryResultsV1,
) -> tuple[dict[str, object], dict[str, object]]:
    if results.promotion_lock_id != lock.promotion_lock_id:
        raise ValueError("query results reference a different promotion lock")
    expected_keys = {
        (group_id, arm.arm_id)
        for group_id in lock.target_group_ids
        for arm in lock.arms
    }
    rows_by_key = {(row.group_id, row.arm_id): row for row in results.rows}
    if set(rows_by_key) != expected_keys:
        missing = sorted(expected_keys - set(rows_by_key))
        extra = sorted(set(rows_by_key) - expected_keys)
        raise ValueError(
            "query result matrix differs from the frozen group-by-arm design: "
            f"missing={missing}, extra={extra}"
        )

    fallback_arm = lock.physical_fallback_arm_id
    fallback_by_group: dict[str, PromotionQueryRowV1] = {}
    for group_id in lock.target_group_ids:
        row = rows_by_key[(group_id, fallback_arm)]
        if row.accepted is not None or row.exact_fallback_reproduced is not None:
            raise ValueError("physical fallback rows must not carry guard decisions")
        if row.deployed_artifact_id != row.fallback_artifact_id:
            raise ValueError("physical fallback row artifact identities must match")
        if row.technical_failure:
            raise ValueError("physical fallback row cannot be a feeder technical failure")
        fallback_by_group[group_id] = row

    aggregate: dict[str, object] = {}
    exact_fallback_failures: list[dict[str, str]] = []
    for arm in lock.arms:
        rows = [rows_by_key[(group_id, arm.arm_id)] for group_id in lock.target_group_ids]
        deltas: list[float] = []
        accepted_count = 0
        rejected_count = 0
        harmful_accepted_count = 0
        technical_failure_count = 0
        exact_fallback_count = 0
        coverages: list[float] = []
        widths: list[float] = []
        for row in rows:
            fallback = fallback_by_group[row.group_id]
            if row.fallback_artifact_id != fallback.deployed_artifact_id:
                raise ValueError(
                    f"query row {row.group_id}/{row.arm_id} changed the physical fallback"
                )
            delta = row.query_rmse_mm - fallback.query_rmse_mm
            deltas.append(delta)
            if arm.role == "physical_fallback":
                continue
            if row.accepted is None:
                raise ValueError("non-fallback query rows require an accept/reject decision")
            if row.accepted:
                accepted_count += 1
                if delta > lock.harmful_update_margin_mm:
                    harmful_accepted_count += 1
                if row.accepted_coverage is not None:
                    coverages.append(row.accepted_coverage)
                if row.accepted_width_mm is not None:
                    widths.append(row.accepted_width_mm)
            else:
                rejected_count += 1
                observed_exact = (
                    row.exact_fallback_reproduced is True
                    and row.deployed_artifact_id == fallback.deployed_artifact_id
                    and row.query_rmse_mm == fallback.query_rmse_mm
                )
                if observed_exact:
                    exact_fallback_count += 1
                else:
                    exact_fallback_failures.append(
                        {"group_id": row.group_id, "arm_id": row.arm_id}
                    )
            technical_failure_count += int(row.technical_failure)
        delta_array = np.asarray(deltas, dtype=np.float64)
        aggregate[arm.arm_id] = {
            "role": arm.role,
            "query_method_id": arm.query_method_id,
            "group_count": len(rows),
            "mean_query_rmse_mm": float(np.mean([row.query_rmse_mm for row in rows])),
            "mean_delta_from_physical_fallback_mm": float(np.mean(delta_array)),
            "worst_group_regression_mm": float(np.max(delta_array)),
            "accepted_update_count": accepted_count,
            "rejected_update_count": rejected_count,
            "harmful_accepted_update_count": harmful_accepted_count,
            "technical_failure_count": technical_failure_count,
            "exact_fallback_count": exact_fallback_count,
            "mean_accepted_coverage": (
                None if not coverages else float(np.mean(coverages))
            ),
            "mean_accepted_width_mm": None if not widths else float(np.mean(widths)),
        }

    primary_rows = [
        rows_by_key[(group_id, lock.primary_query_arm_id)]
        for group_id in lock.target_group_ids
    ]
    primary_deltas = np.asarray(
        [
            row.query_rmse_mm - fallback_by_group[row.group_id].query_rmse_mm
            for row in primary_rows
        ],
        dtype=np.float64,
    )
    bootstrap = _paired_bootstrap(
        primary_deltas,
        resamples=lock.bootstrap_resamples,
        seed=lock.bootstrap_seed,
    )
    primary_summary = cast(Mapping[str, Any], aggregate[lock.primary_query_arm_id])
    coverage_threshold = lock.minimum_mean_accepted_coverage
    observed_coverage = primary_summary["mean_accepted_coverage"]
    coverage_passed = (
        True
        if coverage_threshold is None
        else observed_coverage is not None
        and float(observed_coverage) >= coverage_threshold
    )
    exact_fallback_passed = not exact_fallback_failures
    decision = {
        "primary_query_arm_id": lock.primary_query_arm_id,
        "physical_fallback_arm_id": fallback_arm,
        "minimum_target_group_count": lock.minimum_target_group_count,
        "observed_target_group_count": len(lock.target_group_ids),
        "target_group_count_passed": (
            len(lock.target_group_ids) >= lock.minimum_target_group_count
        ),
        "paired_bootstrap": bootstrap,
        "query_superiority_margin_mm": lock.query_superiority_margin_mm,
        "query_superiority_passed": (
            float(bootstrap["ci95_upper"]) <= -lock.query_superiority_margin_mm
        ),
        "harmful_update_margin_mm": lock.harmful_update_margin_mm,
        "maximum_harmful_accepted_updates": lock.maximum_harmful_accepted_updates,
        "observed_harmful_accepted_updates": primary_summary[
            "harmful_accepted_update_count"
        ],
        "harmful_accepted_updates_passed": (
            int(primary_summary["harmful_accepted_update_count"])
            <= lock.maximum_harmful_accepted_updates
        ),
        "maximum_worst_group_regression_mm": (
            lock.maximum_worst_group_regression_mm
        ),
        "observed_worst_group_regression_mm": primary_summary[
            "worst_group_regression_mm"
        ],
        "worst_group_regression_passed": (
            float(primary_summary["worst_group_regression_mm"])
            <= lock.maximum_worst_group_regression_mm
        ),
        "maximum_technical_failures": lock.maximum_technical_failures,
        "observed_technical_failures": primary_summary["technical_failure_count"],
        "technical_failures_passed": (
            int(primary_summary["technical_failure_count"])
            <= lock.maximum_technical_failures
        ),
        "minimum_mean_accepted_coverage": coverage_threshold,
        "observed_mean_accepted_coverage": observed_coverage,
        "accepted_coverage_passed": coverage_passed,
        "exact_fallback_failure_count": len(exact_fallback_failures),
        "exact_fallback_failures": exact_fallback_failures,
        "exact_fallback_passed": exact_fallback_passed,
        "decision_semantics": (
            "The upper paired target-group bootstrap bound must clear the frozen "
            "superiority margin; harmful accepted updates, worst-group regression, "
            "technical failures, accepted-update coverage, and exact fallback are "
            "separate conjunctive gates."
        ),
    }
    decision["overall_passed"] = all(
        decision[name] is True
        for name in (
            "target_group_count_passed",
            "query_superiority_passed",
            "harmful_accepted_updates_passed",
            "worst_group_regression_passed",
            "technical_failures_passed",
            "accepted_coverage_passed",
            "exact_fallback_passed",
        )
    )
    return aggregate, decision

