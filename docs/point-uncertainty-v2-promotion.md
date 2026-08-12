# Point uncertainty v2 promotion gate

`PointUncertaintyCalibrationV2` is intentionally experimental. Fitting it after a
`point-covariance-localized` source diagnostic is not enough to replace the
production ray/lateral covariance. This gate adds the missing **disjoint
object/session validation step**.

The evaluator compares one frozen v2 calibration against one content-addressed
production `PointUncertaintyCalibrationV1` artifact on complete validation groups
that were not used to fit either model. It is source-validation evidence only;
it does not change provider-v2 export or open a target cohort.

## Information order

Before validation residuals are inspected:

1. freeze the source covariance localization that authorized v2 development;
2. freeze the passing gauge-propagation readiness artifact used by the v2 fit;
3. fit and seal the candidate `PointUncertaintyCalibrationV2` on its complete
   source/calibration group roster;
4. freeze the matched production-v1 point calibration using the **same independent
   training groups**;
5. freeze the disjoint validation object/session roster and prediction-only
   feature contract; and
6. freeze a promotion policy with all proper-score, calibration, width, and
   worst-group thresholds.

The evaluator rejects a baseline/candidate training-roster mismatch and rejects
any validation group that appears in the training roster. Rows and pixels are
never promoted to independent statistical units.

## Matched production-v1 replay

The validation bundle does not supply arbitrary baseline covariances. The
production-v1 conditional covariance is recomputed from the sealed v1 artifact:

- the calibrated ray/lateral floors and depth coefficients;
- the calibrated disagreement gain and variance scales;
- validation `depth_squared`;
- validation overlap parallel disagreement; and
- validation overlap lateral disagreement.

This prevents a convenient precomputed baseline from silently changing the
comparison. The v2 candidate uses the same residual rows, viewing rays, and its
frozen prediction-only feature contract. Its content identity already binds the exact
`gauge_propagation_readiness_id` admitted during fitting. Shared `Sim(3)` gauge
covariance remains outside both local point-covariance models.

## Validation bundle

The NPZ contains exactly:

- `residual_xyz`: `(N, 3)` source-validation residuals;
- `ray_directions`: `(N, 3)` viewing rays;
- `tangent_reference`: `(N, 3)` prediction-only tangent references;
- `features`: `(N, F)` frozen v2 prediction-only covariates;
- `feature_names`: `(F,)`, exactly matching the v2 calibration;
- `group_ids`: `(N,)` physical object/session IDs;
- `depth_squared`: `(N,)` production-v1 depth term;
- `disagreement_parallel_mean`: `(N,)` production-v1 overlap term;
- `disagreement_lateral_mean`: `(N,)` production-v1 overlap term;
- `provider_manifest_id`: scalar SHA-256 string; and
- `cohort_binding_id`: scalar SHA-256 string.

The complete NPZ bytes are SHA-256-bound into the promotion artifact.

## Metrics

For every validation row and each covariance `Sigma`, the evaluator computes the
3-D Gaussian negative log likelihood

\[
\frac{1}{2}\left(3\log(2\pi)+\log|\Sigma|+e^T\Sigma^{-1}e\right),
\]

the normalized energy `e^T Sigma^-1 e / 3`, 90% three-dimensional ellipsoid
coverage using the fixed chi-square threshold for three degrees of freedom, and
RMS covariance width `sqrt(trace(Sigma) / 3)`.

Rows are first reduced within each complete object/session. The reported overall
metrics then give every validation group equal mass regardless of its row count.

## Promotion policy

A policy JSON freezes these thresholds before validation outcomes are used:

```json
{
  "minimum_group_count": 8,
  "minimum_rows_per_group": 64,
  "minimum_mean_nll_improvement": 0.0,
  "minimum_group_win_fraction": 0.6,
  "maximum_coverage_error_increase": 0.02,
  "maximum_worst_group_coverage_error_increase": 0.05,
  "maximum_mean_width_ratio": 1.25,
  "maximum_worst_group_width_ratio": 1.5,
  "maximum_worst_group_nll_regression": 0.1
}
```

These are illustrative values, not a protocol recommendation. A real experiment
must preregister its own thresholds from source-side design considerations.

The candidate is promotable only when all checks pass simultaneously:

- the v2 fit itself converged;
- enough independent validation groups are present;
- every group has the declared minimum row support;
- equal-group mean Gaussian NLL improves by the required margin;
- the required fraction of groups wins on NLL;
- aggregate 90% coverage error does not worsen beyond the budget;
- the worst group's coverage error does not worsen beyond its budget;
- mean covariance width stays inside its budget;
- every group's covariance width stays inside its budget; and
- no group exceeds the allowed NLL regression.

A failure is a valid negative result and leaves production v1 unchanged.

## Run and replay

```bash
python -m prob4d.point_uncertainty_v2_promotion evaluate \
  --calibration outputs/point-uncertainty-v2.json \
  --baseline-calibration outputs/point-uncertainty-v1.json \
  --validation outputs/point-uncertainty-v2-validation.npz \
  --policy protocols/point-uncertainty-v2-promotion-policy.json \
  --output outputs/point-uncertainty-v2-promotion.json
```

Verification requires all original inputs and deterministically recomputes the
entire report rather than trusting stored decision fields:

```bash
python -m prob4d.point_uncertainty_v2_promotion verify \
  --artifact outputs/point-uncertainty-v2-promotion.json \
  --calibration outputs/point-uncertainty-v2.json \
  --baseline-calibration outputs/point-uncertainty-v1.json \
  --validation outputs/point-uncertainty-v2-validation.npz \
  --policy protocols/point-uncertainty-v2-promotion-policy.json
```

Exit status `0` means the source-validation promotion criteria pass. Status `2`
means a valid negative promotion decision. Invalid bindings, overlapping groups,
changed feature contracts, malformed inputs, or replay mismatches raise errors.

## Claim boundary

Passing this gate means only that the frozen v2 conditional point covariance
outperformed the matched production-v1 covariance under the preregistered,
disjoint source-validation protocol. It does not establish provider competence
on fresh target objects, validate shared gauge uncertainty, enable provider-v2
export automatically, authorize a BayesianPhysTwin update, establish Causal4D
intervention benefit, deployment safety, or state of the art.
