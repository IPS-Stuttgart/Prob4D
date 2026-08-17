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

## Ownership and evidence boundary

Prob4D supplies generic observation-space moments and matrix-free projections.
BayesianPhysTwin owns the physical residual or query Jacobian, update guard, and
exact physical fallback. Causal4D consumes the accepted BayesianPhysTwin belief
and owns intervention contrasts.

These functions propagate an already admitted covariance representation. They
do not calibrate uncertainty, define simultaneous trajectory coverage, select
exchangeability units, promote a provider, or establish physical or causal
benefit. Any future finite-sample calibration must preserve complete physical
objects or acquisition sessions as the independent calibration and evaluation
units; tracks, points, frames, views, and taxels must not be counted as
independent replicates.
