# Query-preserving shared-covariance compression

Prob4D represents shared observation uncertainty with a factor

```text
U in R^(3N x R),        C_shared = U U^T.
```

A conventional rank reduction retains directions with the most observation-space
energy. That is safe as an average covariance approximation, but it can discard a
small observation-space mode that dominates a particular BayesianPhysTwin or
Causal4D query.

`prob4d.query_preserving_compression` is an experimental, source-side diagnostic
that compresses the shared factor against one or more caller-supplied local query
Jacobians. It does not alter a claim-bearing observation artifact and is not used
by the calibrated provider-v2 exporter.

## Registered query factors

For query `h` with block Jacobian `J_h`, define

```text
B_h = J_h U,            C_h = B_h B_h^T.
```

The implementation accepts the same row-block convention as the query-covariance
relevance diagnostic:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `low_rank_factor_m` | `N x 3 x R` | shared observation factor |
| scalar query Jacobian | `N x 3` | one scalar query |
| vector query Jacobian | `Q x N x 3` | one `Q`-dimensional query |

Queries are named in a mapping. Names are sorted before processing, and optional
query weights must cover exactly the registered names.

## Deterministic latent score

The compressor searches only within the original latent column space. It forms

```text
G_obs = U^T U,
G_h   = B_h^T B_h,

S = w_obs G_obs / tr(G_obs)
    + sum_h w_h G_h / tr(G_h),
```

omitting a zero-trace term. Trace normalization prevents a query from dominating
merely because it has more coordinates or different physical units. The weights
therefore state the intended relative importance explicitly.

The eigenvectors of `S` order latent directions. Numerically repeated
eigenvalues are treated as one indivisible eigenspace, and a projector-derived
canonical basis avoids dependence on an arbitrary eigensolver basis. A rank cap
is never allowed to select only part of a repeated eigenspace.

For a candidate latent projection `V_k`, the retained factor is

```text
U_k = U V_k.
```

## Admission limits

`QueryPreservingCompressionPolicyV1` freezes three independent requirements:

1. minimum retained observation trace;
2. maximum relative query-covariance trace loss for every registered query; and
3. maximum relative query-covariance spectral loss for every registered query.

For query `h`, the discarded covariance is positive semidefinite:

```text
Delta C_h = C_h - B_h V_k V_k^T B_h^T.
```

The reported losses are

```text
trace_loss_h    = tr(Delta C_h) / tr(C_h),
spectral_loss_h = ||Delta C_h||_2 / ||C_h||_2.
```

A zero query covariance reports zero loss rather than an undefined ratio.
Candidate ranks are evaluated from smallest to largest, but only at complete
score-eigenspace boundaries.

## Example

```python
import numpy as np

from prob4d.query_preserving_compression import (
    QueryPreservingCompressionPolicyV1,
    compress_shared_factor_for_queries,
)

policy = QueryPreservingCompressionPolicyV1(
    minimum_observation_trace_fraction=0.99,
    maximum_query_trace_loss_fraction=0.01,
    maximum_query_spectral_loss_fraction=0.05,
    maximum_rank=16,
    observation_weight=1.0,
    query_weights={
        "early_endpoint": 1.0,
        "late_endpoint": 2.0,
    },
)

result = compress_shared_factor_for_queries(
    low_rank_factor_m,
    {
        "early_endpoint": early_query_jacobian,
        "late_endpoint": late_query_jacobian,
    },
    policy=policy,
)

factor_for_diagnostic_use = result.compressed_factor_m
print(result.summary())
```

Every retained array is an immutable copy. The summary reports the original and
retained ranks, observation trace fraction, per-query trace and spectral losses,
the frozen policy, and the explicit claim boundary.

## Exact full-rank fallback

If no strict reduction satisfies every policy limit within `maximum_rank`, the
routine does not return an underqualified approximation. It returns the exact
caller-supplied full factor with the identity latent projection and records

```text
no-admissible-reduction-within-rank-cap
```

as the fallback reason. If the full rank is required even without a cap, it
records `full-rank-required`. This fallback preserves the original shared
covariance exactly and makes a failed compression attempt observable.

## Intended use and boundary

Use this diagnostic only after a provider has useful source means and identities,
adequate gauge/dependence calibration, and a downstream query whose shared
covariance is demonstrably relevant. It is not a substitute for the ordered
provider-readiness gates.

BayesianPhysTwin owns the query Jacobian, practical-equivalence limits, guarded
update, and exact physical fallback. Causal4D consumes only the selected
BayesianPhysTwin belief. A successful compression result does not establish
provider competence, uncertainty calibration on a fresh cohort, physical-query
benefit, intervention benefit, deployment safety, or state of the art.
