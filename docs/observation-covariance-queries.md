# Structured observation covariance queries

Prob4D observation-factor stacks represent the joint row covariance as

\[
\Sigma_y = \operatorname{blockdiag}(R_1,\ldots,R_M)
           + J\Sigma_gJ^\top,
\]

where `R_i` is conditional point uncertainty and `Sigma_g` is the shared
cross-window gauge prior. The stored per-row marginal blocks include each row's
own gauge contribution, but not the cross-row covariance induced by the common
gauge state. Summing those marginal blocks is therefore wrong for a physical
query that combines several rows.

## Matrix-free query projection

The additive query API supports dense, sparse, and tree-sparse factor stacks:

```python
from prob4d.api.v2 import project_observation_covariance

# A has shape (query_dimension, observation_count, 3).
projection = project_observation_covariance(stacked, A)

conditional = projection.conditional_covariance
gauge = projection.gauge_covariance
marginal = projection.marginal_covariance
```

The result is exactly `A @ Sigma_y @ A.T`. Only the requested query covariance
is materialized; the tree-sparse route keeps the gauge prior matrix-free. A
single query may be passed with shape `(M, 3)`, and a conventional flattened
Jacobian with shape `(Q, 3M)` is accepted as well.

For covariance actions and scalar covariance diagnostics:

```python
from prob4d.api.v2 import (
    observation_covariance_action,
    observation_covariance_quadratic,
)

sigma_v = observation_covariance_action(stacked, v)
energy = observation_covariance_quadratic(stacked, v)
```

The `component` argument can select `"conditional"`, `"gauge"`, or
`"marginal"`. This decomposition is suitable for a same-mean covariance-value
certificate: downstream code can hold the observation mean and physical query
fixed while changing only the admitted uncertainty representation.

## Structured Gaussian solves and proper scores

Likelihood evaluation needs the inverse action and log determinant of the same
joint covariance, not a sum of independent rowwise marginal scores. Build one
cached operator when several residuals or right-hand sides share the same factor
stack:

```python
from prob4d.api.v2 import build_observation_gaussian_operator

operator = build_observation_gaussian_operator(stacked)

precision_residual = operator.solve(residual_xyz_m)
mahalanobis_squared = operator.precision_quadratic(residual_xyz_m)
log_determinant = operator.log_determinant
gaussian_nll = operator.gaussian_nll(residual_xyz_m)

factor_bytes = operator.factor_storage_nbytes
dense_bytes = operator.dense_covariance_nbytes
storage_ratio = operator.factor_storage_ratio_to_dense
```

`solve` accepts shape `(M, 3)` or batched shape `(M, 3, R)`. Convenience
functions are also available for one-off calls:

```python
from prob4d.api.v2 import (
    observation_gaussian_nll,
    observation_log_determinant,
    observation_precision_quadratic,
    solve_observation_covariance,
)
```

Dense and sparse stacks use a covariance-root Woodbury factorization. The gauge
prior may be positive semidefinite; zero-variance gauge directions are retained
as a rank-deficient covariance rather than regularized silently. Tree-sparse
stacks combine the row information with the causal gauge-tree precision and
eliminate seven-dimensional blocks from leaves to the root. This computes the
exact inverse action and determinant without materializing either the full
`3M x 3M` observation covariance or a dense tree-gauge covariance.

The conditional row covariances `R_i` must be strictly positive definite for a
proper nonsingular Gaussian likelihood. A singular conditional block fails
closed instead of receiving an implicit jitter. The inverse covariance does not
admit an additive `conditional`/`gauge` component interpretation, so these
functions always operate on the complete marginal covariance.

The storage properties report the cached numerical factors only and exclude the
already-owned immutable input stack. For the tree-sparse backend this storage is
linear in observation and gauge count: one conditional `3 x 3` factor per row
and two `7 x 7` blocks per gauge. `factorization_backend` records which exact
implementation was selected:

- `dense-gauge-root-woodbury-v1`;
- `sparse-gauge-root-woodbury-v1`; or
- `tree-block-information-v1`.

## Group-aware pathwise calibration

Marginal 95% point coverage does not imply that every admitted trajectory from a
physical object or acquisition session is covered jointly. The first array axis
may still enumerate material-point trajectories, but trajectories are not
independent calibration units. Every trajectory must therefore be assigned to a
complete, frozen object or session group:

```python
from prob4d.api.v2 import (
    fit_pathwise_maximum_calibration,
    pathwise_uncertainty_diagnostics,
)

calibration = fit_pathwise_maximum_calibration(
    calibration_residuals,       # (P_cal, T, 3)
    calibration_covariances,     # (P_cal, T, 3, 3)
    group_ids=calibration_group_ids,  # one object/session ID per trajectory
    independent_unit="physical-object",
    valid_mask=calibration_valid,
    miscoverage=0.05,
)

diagnostics = pathwise_uncertainty_diagnostics(
    target_residuals,
    target_covariances,
    group_ids=target_group_ids,
    independent_unit="physical-object",
    valid_mask=target_valid,
    calibration=calibration,
)
```

`independent_unit` must be either `"physical-object"` or
`"acquisition-session"`. IDs must be stable, opaque identifiers from the frozen
cohort lock, not track IDs and not labels derived after outcomes are opened. The
calibration artifact stores the complete trajectory-to-group assignment and its
SHA-256 digest. Target evaluation rejects any group ID that also appears in the
calibration artifact.

For each independent group, the calibration statistic is the maximum squared
Mahalanobis error over every admitted trajectory and valid step in that group.
The split-conformal rank is computed from the number of distinct groups. Adding
more tracks from an existing object or session cannot increase the finite-sample
sample size. At 5% miscoverage, at least 19 independent groups are required for a
finite threshold.

Target coverage, maximum-error quantiles, valid-support fractions, and Gaussian
score use equal group weighting. Fields describing longest failure or unsupported
runs remain explicitly per-trajectory diagnostics. `all_groups_inside_marginal_*`
uses a pointwise chi-squared threshold throughout each group and remains a
diagnostic rather than a simultaneous confidence statement. A calibrated
simultaneous group-coverage result is emitted only when an independently fitted
`PathwiseMaximumCalibrationV1` is supplied.

## Ownership boundary

Prob4D supplies generic observation-space moments, structured inverse actions,
proper Gaussian scores, matrix-free projections, and group-aware
source/calibration diagnostics. BayesianPhysTwin owns the physical residual or
query Jacobian, update guard, and exact physical fallback. Causal4D consumes the
accepted BayesianPhysTwin belief and owns intervention contrasts. These APIs do
not by themselves establish provider competence, physical benefit, causal
benefit, deployment safety, or scientific promotion.
