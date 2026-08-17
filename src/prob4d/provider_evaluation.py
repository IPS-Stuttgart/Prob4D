"""Evaluate causally valid Prob4D provider artifacts across held-out groups."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from ._heldout_promotion_lock import load_promotion_lock
from ._provider_evaluation_decision import evaluate_provider_decision_policy
from ._provider_evaluation_manifest import (
    PROVIDER_EVALUATION_DECISION_VERSION,
    PROVIDER_EVALUATION_SCHEMA,
    PROVIDER_EVALUATION_VERSION,
    EvaluationModeName,
    ProviderEvaluationCase,
    ProviderEvaluationDecisionPolicy,
    ProviderEvaluationDecisionRule,
    load_provider_evaluation_plan,
    validate_finite_json,
)
from ._provider_evaluation_output import write_provider_evaluation_outputs
from ._provider_evaluation_provider_neutral import (
    aggregate_provider_records,
    evaluate_provider_cases,
)
from .metrics import DEFAULT_EVALUATION_CHUNK_SIZE
from .provider_evaluation_target_authorization import (
    PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD,
    PROVIDER_EVALUATION_TARGET_AUTHORIZED_REPORT_VERSION,
    ProviderEvaluationManifestSnapshotV1,
    build_provider_evaluation_target_authorization,
    load_provider_evaluation_manifest_snapshot,
)
from .target_provider_admission import load_target_provider_admission


def _load_authorized_plan(
    source: Path,
    *,
    promotion_lock_path: str | Path,
    target_provider_admission_path: str | Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    allow_legacy_artifacts: bool,
) -> tuple[
    list[ProviderEvaluationCase],
    EvaluationModeName,
    str,
    dict[str, Any],
    ProviderEvaluationDecisionPolicy,
    dict[str, object],
    str,
    ProviderEvaluationManifestSnapshotV1,
]:
    """Authorize exact manifest bytes before resolving any target paths."""

    snapshot = load_provider_evaluation_manifest_snapshot(source)
    lock = load_promotion_lock(promotion_lock_path)
    admission = load_target_provider_admission(target_provider_admission_path)
    authorization = build_provider_evaluation_target_authorization(
        snapshot,
        lock,
        admission,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        legacy_artifacts_allowed=allow_legacy_artifacts,
    )
    with snapshot.materialize_execution_manifest() as execution_manifest:
        cases, primary_mode, reference_method, metadata, decision_policy = (
            load_provider_evaluation_plan(execution_manifest)
        )
    if decision_policy is None:
        raise AssertionError("authorized schema-v2 provider evaluation lost its decision policy")
    if decision_policy.to_dict() != snapshot.decision_policy.to_dict():
        raise ValueError("provider decision policy changed during private snapshot loading")
    if metadata != dict(snapshot.metadata):
        raise ValueError("provider-evaluation metadata changed during private snapshot loading")
    if tuple(sorted({case.group_id for case in cases})) != snapshot.target_group_ids:
        raise ValueError("provider-evaluation target groups changed during snapshot loading")
    observed_methods = tuple(sorted(cases[0].predictions))
    if observed_methods != snapshot.method_ids:
        raise ValueError("provider-evaluation methods changed during snapshot loading")
    return (
        cases,
        primary_mode,
        reference_method,
        metadata,
        decision_policy,
        authorization,
        snapshot.source_manifest_sha256,
        snapshot,
    )


def run_provider_evaluation(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
    allow_legacy_artifacts: bool = False,
    evaluation_chunk_size: int = DEFAULT_EVALUATION_CHUNK_SIZE,
    promotion_lock_path: str | Path | None = None,
    target_provider_admission_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run paired multi-case evaluation with equal-group bootstrap aggregation."""

    if isinstance(bootstrap_resamples, bool) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if isinstance(evaluation_chunk_size, bool) or evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    normalized_seed = int(seed)
    if normalized_seed != seed:
        raise ValueError("seed must be an integer")
    if (promotion_lock_path is None) != (target_provider_admission_path is None):
        raise ValueError(
            "promotion_lock_path and target_provider_admission_path must be supplied together"
        )

    source = Path(manifest_path).resolve()
    target_authorization: dict[str, object] | None = None
    manifest_snapshot: ProviderEvaluationManifestSnapshotV1 | None = None
    if promotion_lock_path is None:
        cases, primary_mode, reference_method, manifest_metadata, decision_policy = (
            load_provider_evaluation_plan(source)
        )
        source_manifest_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        assert target_provider_admission_path is not None
        (
            cases,
            primary_mode,
            reference_method,
            manifest_metadata,
            decision_policy,
            target_authorization,
            source_manifest_sha256,
            manifest_snapshot,
        ) = _load_authorized_plan(
            source,
            promotion_lock_path=promotion_lock_path,
            target_provider_admission_path=target_provider_admission_path,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=normalized_seed,
            allow_legacy_artifacts=allow_legacy_artifacts,
        )

    records, method_metadata = evaluate_provider_cases(
        cases,
        allow_legacy_artifacts=allow_legacy_artifacts,
        evaluation_chunk_size=evaluation_chunk_size,
    )
    aggregate, comparisons = aggregate_provider_records(
        records,
        reference_method=reference_method,
        bootstrap_resamples=bootstrap_resamples,
        seed=normalized_seed,
    )
    decision = (
        None
        if decision_policy is None
        else evaluate_provider_decision_policy(
            decision_policy,
            aggregate=aggregate,
            comparisons=comparisons,
            primary_mode=primary_mode,
            reference_method=reference_method,
        )
    )
    if manifest_snapshot is not None:
        manifest_snapshot.assert_source_unchanged()
    clean_records = [
        {key: value for key, value in record.items() if key != "_numeric"} for record in records
    ]
    schema_version = 2 if decision_policy is None else 3
    if target_authorization is not None:
        schema_version = PROVIDER_EVALUATION_TARGET_AUTHORIZED_REPORT_VERSION
    report: dict[str, Any] = {
        "schema_name": "prob4d.provider-evaluation-report",
        "schema_version": schema_version,
        "source_manifest": str(source),
        "source_manifest_sha256": source_manifest_sha256,
        "primary_mode": primary_mode,
        "primary_support": "common_across_registered_methods",
        "secondary_support": "native_per_method",
        "reference_method": reference_method,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": normalized_seed,
        "legacy_artifacts_allowed": allow_legacy_artifacts,
        "evaluation_chunk_size": evaluation_chunk_size,
        "manifest_metadata": manifest_metadata,
        "method_metadata": method_metadata,
        "cases": clean_records,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "support_semantics": (
            "Primary metrics use the intersection of truth support and every "
            "registered method within each case. Native per-method metrics and "
            "retention fractions are reported separately to expose selective "
            "missingness."
        ),
        "claim_boundary": (
            "This report measures held-out observation competence. It does not by itself "
            "establish Bayesian-PhysTwin acceptance, physical-prediction benefit, or "
            "Causal4D intervention benefit."
        ),
    }
    if decision_policy is not None:
        report["decision_policy"] = decision_policy.to_dict()
        report["decision"] = decision
    if target_authorization is not None:
        report[PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD] = target_authorization
    validate_finite_json(report, name="provider-evaluation report")
    write_provider_evaluation_outputs(
        output_directory,
        report=report,
        records=records,
        primary_mode=primary_mode,
        reference_method=reference_method,
        aggregate=aggregate,
        comparisons=comparisons,
        decision=decision,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--promotion-lock",
        type=Path,
        help=(
            "frozen promotion lock for a claim-bearing target evaluation; must be "
            "supplied together with --target-provider-admission"
        ),
    )
    parser.add_argument(
        "--target-provider-admission",
        type=Path,
        help=(
            "metadata-only admission checked against the exact lock and manifest "
            "before truth or prediction artifacts are opened"
        ),
    )
    parser.add_argument(
        "--evaluation-chunk-size",
        type=int,
        default=DEFAULT_EVALUATION_CHUNK_SIZE,
        help=(
            "maximum spatial samples materialized per metric/covariance chunk; "
            "changes execution memory, not estimator semantics"
        ),
    )
    parser.add_argument(
        "--allow-legacy-artifacts",
        action="store_true",
        help=(
            "permit historical fused NPZ files whose covariance and dependence "
            "semantics were not embedded; use only for labelled diagnostics"
        ),
    )
    parser.add_argument(
        "--require-decision-pass",
        action="store_true",
        help=(
            "require a schema-v2 preregistered decision policy and return exit code "
            "3 when its independent-group or paired metric gates do not pass"
        ),
    )
    arguments = parser.parse_args(argv)
    if (arguments.promotion_lock is None) != (arguments.target_provider_admission is None):
        parser.error("--promotion-lock and --target-provider-admission must be supplied together")
    report = run_provider_evaluation(
        arguments.manifest,
        arguments.output_dir,
        bootstrap_resamples=arguments.bootstrap_resamples,
        seed=arguments.seed,
        allow_legacy_artifacts=arguments.allow_legacy_artifacts,
        evaluation_chunk_size=arguments.evaluation_chunk_size,
        promotion_lock_path=arguments.promotion_lock,
        target_provider_admission_path=arguments.target_provider_admission,
    )
    print(arguments.output_dir / "provider_evaluation.json")
    if arguments.require_decision_pass:
        decision = report.get("decision")
        if not isinstance(decision, dict):
            parser.error("--require-decision-pass requires a schema-v2 manifest decision_policy")
        if decision.get("overall_passed") is not True:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROVIDER_EVALUATION_DECISION_VERSION",
    "PROVIDER_EVALUATION_SCHEMA",
    "PROVIDER_EVALUATION_TARGET_AUTHORIZED_REPORT_VERSION",
    "PROVIDER_EVALUATION_VERSION",
    "ProviderEvaluationCase",
    "ProviderEvaluationDecisionPolicy",
    "ProviderEvaluationDecisionRule",
    "run_provider_evaluation",
]
