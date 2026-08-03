"""Evaluate causally valid Prob4D provider artifacts across held-out groups."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from ._provider_evaluation_compute import (
    aggregate_provider_records,
    evaluate_provider_cases,
)
from ._provider_evaluation_decision import evaluate_provider_decision_policy
from ._provider_evaluation_manifest import (
    PROVIDER_EVALUATION_DECISION_VERSION,
    PROVIDER_EVALUATION_SCHEMA,
    PROVIDER_EVALUATION_VERSION,
    ProviderEvaluationCase,
    ProviderEvaluationDecisionPolicy,
    ProviderEvaluationDecisionRule,
    load_provider_evaluation_plan,
    validate_finite_json,
)
from ._provider_evaluation_output import write_provider_evaluation_outputs
from .metrics import DEFAULT_EVALUATION_CHUNK_SIZE


def run_provider_evaluation(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
    allow_legacy_artifacts: bool = False,
    evaluation_chunk_size: int = DEFAULT_EVALUATION_CHUNK_SIZE,
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
    source = Path(manifest_path).resolve()
    cases, primary_mode, reference_method, manifest_metadata, decision_policy = (
        load_provider_evaluation_plan(source)
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
    clean_records = [
        {key: value for key, value in record.items() if key != "_numeric"}
        for record in records
    ]
    report: dict[str, Any] = {
        "schema_name": "prob4d.provider-evaluation-report",
        "schema_version": 2 if decision_policy is None else 3,
        "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
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
    report = run_provider_evaluation(
        arguments.manifest,
        arguments.output_dir,
        bootstrap_resamples=arguments.bootstrap_resamples,
        seed=arguments.seed,
        allow_legacy_artifacts=arguments.allow_legacy_artifacts,
        evaluation_chunk_size=arguments.evaluation_chunk_size,
    )
    print(arguments.output_dir / "provider_evaluation.json")
    if arguments.require_decision_pass:
        decision = report.get("decision")
        if not isinstance(decision, dict):
            parser.error(
                "--require-decision-pass requires a schema-v2 manifest decision_policy"
            )
        if decision.get("overall_passed") is not True:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROVIDER_EVALUATION_DECISION_VERSION",
    "PROVIDER_EVALUATION_SCHEMA",
    "PROVIDER_EVALUATION_VERSION",
    "ProviderEvaluationCase",
    "ProviderEvaluationDecisionPolicy",
    "ProviderEvaluationDecisionRule",
    "run_provider_evaluation",
]
