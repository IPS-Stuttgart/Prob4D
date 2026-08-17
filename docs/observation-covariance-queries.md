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

For iterative algorithms and scalar diagnostics:

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

## Pathwise calibration

Marginal 95% point coverage does not imply that an entire material-point track
is covered with probability 95%. Prob4D therefore exposes equal-trajectory
pathwise diagnostics:

```python
from prob4d.api.v2 import (
    fit_pathwise_maximum_calibration,
    pathwise_uncertainty_diagnostics,
)

calibration = fit_pathwise_maximum_calibration(
    calibration_residuals,       # (P_cal, T, 3)
    calibration_covariances,     # (P_cal, T, 3, 3)
    valid_mask=calibration_valid,
    miscoverage=0.05,
)

diagnostics = pathwise_uncertainty_diagnostics(
    target_residuals,
    target_covariances,
    valid_mask=target_valid,
    calibration=calibration,
)
```

The calibration statistic is the maximum squared Mahalanobis error along one
trajectory. Its split-conformal order statistic is fitted on source/calibration
trajectories treated as complete exchangeability units, including their frozen
support masks, then frozen before target residuals or outcomes are opened. The
target report includes calibrated simultaneous coverage, maximum
whitened-error quantiles, longest contiguous marginal-coverage failure, support
retention, longest unsupported run, and an equal-trajectory Gaussian score.

Fields named `all_steps_inside_marginal_*` deliberately remain diagnostic. They
apply a pointwise chi-squared threshold at every step and are not labelled as
simultaneous confidence coverage. A simultaneous result is emitted only when an
independently fitted `PathwiseMaximumCalibrationV1` is supplied.

## Ownership boundary

Prob4D supplies generic observation-space moments and matrix-free projections.
BayesianPhysTwin owns the physical residual/query Jacobian, update guard, and
exact physical fallback. Causal4D consumes the accepted BayesianPhysTwin belief
and owns intervention contrasts. These APIs provide infrastructure and
diagnostics; they do not by themselves establish provider competence, physical
benefit, causal benefit, or scientific promotion.
