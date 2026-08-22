# Local Gaussian query conditioning

Prob4D can now carry a correlation-aware observation covariance into a local
Bayesian physical-query update without materializing a dense `3M x 3M`
covariance.

The implementation is intentionally split at the repository boundary:

- Prob4D supplies a cached `ObservationGaussianOperator` for the provider
  covariance \(R\).
- BayesianPhysTwin supplies the physical prior, the frozen local
  linearization, and the registered physical query.
- The new `prob4d.query_posterior` helpers evaluate the resulting structured
  Gaussian update.
- Causal4D should consume only an accepted downstream posterior or query
  result, not raw provider evidence.

## Innovation covariance

For a fixed local model

\[
x \sim \mathcal N(m, P), \qquad
y = Hx + \epsilon, \qquad
\epsilon \sim \mathcal N(0, R),
\]

let \(P^{1/2}\) be any covariance root and define

\[
F = H P^{1/2}.
\]

The innovation covariance is

\[
S = R + HPH^\top = R + FF^\top.
\]

`augment_observation_gaussian_operator` represents this update with a second
Woodbury factorization:

```python
from prob4d.query_posterior import augment_observation_gaussian_operator

innovation_operator = augment_observation_gaussian_operator(
    prob4d_noise_operator,
    (H @ prior_root).reshape(observation_count, 3, -1),
)
```

The resulting operator supports batched solves, log determinants, precision
quadratics, and Gaussian negative log likelihoods. It retains only the base
Prob4D factorization, the low-rank factor, its base-precision response, and the
small Woodbury core.

## Posterior of a registered query

For a query \(q=Lx\), define

\[
m_q = Lm,\qquad
P_q = LPL^\top,\qquad
C_{qy} = LPH^\top,
\]

and the innovation \(\nu=y-Hm\). The exact local Gaussian posterior is

\[
m_{q\mid y} = m_q + C_{qy}S^{-1}\nu,
\]

\[
P_{q\mid y} = P_q - C_{qy}S^{-1}C_{qy}^\top.
\]

The implementation evaluates both expressions with one batched structured
solve:

```python
from prob4d.query_posterior import condition_gaussian_query

posterior = condition_gaussian_query(
    prior_mean=L @ state_mean,
    prior_covariance=L @ P @ L.T,
    innovation=(observation - H @ state_mean).reshape(observation_count, 3),
    query_observation_cross_covariance=(
        L @ P @ H.T
    ).reshape(query_dimension, observation_count, 3),
    innovation_operator=innovation_operator,
)

query_mean = posterior.posterior_mean
query_covariance = posterior.posterior_covariance
```

Only query-sized matrices are materialized. The result also records the mean
shift, covariance reduction, innovation precision quadratic, log determinant,
and joint Gaussian negative log likelihood.

## Validity boundary

The result is exact for the supplied fixed local linear-Gaussian model. It is
not a globally sufficient message for an arbitrary nonlinear or robust
posterior.

For nonlinear or robust inference, callers must freeze and identify:

- the state and query linearization;
- the physical prior or covariance root;
- the effective observation covariance;
- any IRLS, mixture, gating, or robust weights.

A new message must be constructed whenever any of these quantities changes.
The helper fails closed when the supplied prior query covariance and
query-observation cross covariance imply a non-positive-semidefinite posterior.

This module is currently an experimental integration surface. It is not yet
part of `prob4d.api.v2`; promotion should follow cross-repository parity tests
with BayesianPhysTwin.
