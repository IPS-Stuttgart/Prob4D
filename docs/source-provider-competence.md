# Source-only provider competence

`prob4d.source_provider_competence` evaluates whether an observation provider has
useful source-side means and sufficiently reliable identities before a richer
point-covariance model can be considered.

The registered statistical unit is one complete physical object or acquisition
session. Frames, pixels, cameras, tracks, and points remain nested observations;
they are not counted as independent source replicates.

## Why this gate exists

A downstream physical update can fail for several different reasons:

1. the provider mean is inaccurate or drifts;
2. material identities or associations are unreliable;
3. the shared gauge/dependence model is wrong;
4. conditional point covariance is miscalibrated despite useful means and
   identities; or
5. the downstream physical query is not identifiable or is insensitive to the
   observation.

Only case 4 authorizes richer point-uncertainty development. The source report
therefore produces two separate decisions:

- `mean_quality_status`; and
- `identity_reliability_status`.

A failing mean-quality gate must not be relabelled as a covariance problem.

## Group record

Each `SourceProviderGroupResultV1` contains equal-unit metrics for one complete
object/session:

- candidate and baseline proper scores;
- point and endpoint RMSE;
- absolute drift slope;
- overlap-seam RMSE;
- association precision;
- identity retention; and
- support retention.

A technical-failure group contains only its predeclared failure code and
metadata. It cannot mix a failure disposition with scored metrics.

## Frozen policy

`SourceProviderCompetencePolicyV1` freezes:

- the minimum number of evaluable groups;
- permitted technical-failure codes and their maximum count;
- mean proper-score, point-RMSE, endpoint-RMSE, drift, and seam limits;
- a worst-group point-RMSE limit;
- mean-quality group-pass coverage;
- association, identity-retention, and support-retention minima; and
- identity group-pass coverage.

All aggregate quantities are equal-group means. Large objects, long videos, or
dense cameras cannot dominate by contributing more rows.

## Example

```python
from prob4d.source_provider_competence import (
    SourceProviderCompetencePolicyV1,
    SourceProviderCompetenceReportV1,
    SourceProviderGroupResultV1,
    write_source_provider_competence,
)

policy = SourceProviderCompetencePolicyV1(
    minimum_evaluable_groups=8,
    maximum_technical_failures=0,
    permitted_technical_failure_codes=(),
    maximum_mean_proper_score_delta=0.0,
    maximum_mean_point_rmse_ratio=1.0,
    maximum_mean_endpoint_rmse_ratio=1.0,
    maximum_worst_group_point_rmse_ratio=1.2,
    maximum_mean_absolute_drift_slope_m_per_frame=0.001,
    maximum_mean_seam_rmse_m=0.01,
    minimum_mean_quality_group_pass_fraction=0.75,
    minimum_mean_association_precision=0.9,
    minimum_mean_identity_retention=0.8,
    minimum_mean_support_retention=0.8,
    minimum_identity_group_pass_fraction=0.75,
)

report = SourceProviderCompetenceReportV1(
    provider_manifest_id=provider_manifest_id,
    cohort_binding_id=source_cohort_binding_id,
    group_definition="complete-physical-object-v1",
    policy=policy,
    groups=tuple(
        SourceProviderGroupResultV1(
            group_id=result.object_id,
            candidate_proper_score=result.provider_proper_score,
            baseline_proper_score=result.baseline_proper_score,
            candidate_point_rmse_m=result.provider_point_rmse_m,
            baseline_point_rmse_m=result.baseline_point_rmse_m,
            candidate_endpoint_rmse_m=result.provider_endpoint_rmse_m,
            baseline_endpoint_rmse_m=result.baseline_endpoint_rmse_m,
            absolute_drift_slope_m_per_frame=abs(result.drift_slope_m_per_frame),
            seam_rmse_m=result.seam_rmse_m,
            association_precision=result.association_precision,
            identity_retention=result.identity_retention,
            support_retention=result.support_retention,
            metadata={"session_ids": result.session_ids},
        )
        for result in frozen_source_results
    ),
)
write_source_provider_competence("source-competence.json", report)
```

Construction rejects target payload or target outcome access. Loading the JSON
recomputes every aggregate, decision, reason code, and content identity.

## Readiness integration

`prob4d.fresh_provider_readiness.source_competence_gates()` converts a report
into the ordered source-mean and identity/reliability gates. When mean quality
fails, identity is left `not-evaluated`; downstream evidence cannot be used to
rescue the terminal source-mean-negative result.

## Claim boundary

A passing source report establishes only source-side competence under the exact
provider, cohort, and frozen policy. It does not prove target transfer,
calibrated uncertainty, BayesianPhysTwin benefit, Causal4D benefit, deployment
safety, or state of the art.
