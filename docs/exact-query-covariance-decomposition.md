# Exact registered-query covariance decomposition

Prob4D represents shared observation uncertainty by a factor

```text
U in R^(3N x R),        C_shared = U U^T.
```

A downstream physical or diagnostic query may depend on only a lower-dimensional
part of the latent uncertainty. The experimental
`prob4d.exact_query_covariance` module identifies that part without silently
deleting the remaining observation uncertainty.

## Registered query factors

For each registered linear query with Jacobian `J_h`, define

```text
B_h = J_h U.
```

Stack every query factor by rows:

```text
B = stack_h(B_h).
```

Let `V_q` be an orthonormal basis for the row space of `B`, and let `V_n` be an
orthonormal basis for its orthogonal complement. The returned factors are

```text
U_q = U V_q,            U_n = U V_n.
```

`U_q` is query-coupled. `U_n` is query-orthogonal for the registered projections.
They form a partition, not a lossy truncation.

## Exact decomposition theorem

Because `[V_q, V_n]` is an orthogonal basis of the original latent space,

```text
U U^T = U_q U_q^T + U_n U_n^T.
```

Furthermore, `B V_n = 0`. Therefore, for every pair of registered queries `h`
and `g`,

```text
J_h U U^T J_g^T = J_h U_q U_q^T J_g^T.
```

The query-coupled rank is minimal. Suppose another orthogonal projection `P`
preserves the stacked registered-query covariance. Then

```text
B (I - P) B^T = 0.
```

The left-hand side equals

```text
[B (I - P)] [B (I - P)]^T,
```

so `B (I - P) = 0`. The row space of `B` must therefore lie inside the retained
subspace, and every exact retained subspace has rank at least `rank(B)`.
Consequently,

```text
minimum exact registered-query rank = rank(B).
```

The implementation computes this rank numerically by SVD and records the exact
threshold and singular values used in the decision.

## Why the query-orthogonal factor must remain

`J_h U_n = 0` does **not** make `U_n` irrelevant to a Gaussian observation
likelihood. It still contributes

```text
U_n U_n^T
```

to the observation covariance and therefore generally changes its inverse, log
determinant, innovation score, and posterior conditioning. Dropping `U_n` would
usually make the observation overconfident and can change a BayesianPhysTwin
update even though the registered forward projection cannot see it directly.

Safe uses retain both components, for example:

- keep `U_n` inside a base observation operator and expose `U_q` as the small
  query-coupled Woodbury block;
- diagnose how much latent rank is directly coupled to registered physical
  queries;
- verify whether an approximate compressor retained the mathematically required
  query subspace; or
- store the two factors separately while reconstructing their covariance sum
  before likelihood evaluation.

The decomposition alone is not a license to discard nuisance uncertainty.

## API

```python
from prob4d.exact_query_covariance import (
    decompose_shared_factor_for_exact_queries,
)

result = decompose_shared_factor_for_exact_queries(
    low_rank_factor_m,
    {
        "early_endpoint": early_query_jacobian,
        "late_endpoint": late_query_jacobian,
    },
    maximum_query_rank=16,
    relative_rank_tolerance=0.0,
)

query_factor = result.query_coupled_factor_m
nuisance_factor = result.query_orthogonal_factor_m
print(result.summary())
```

Accepted shapes follow the existing query-projection convention:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `low_rank_factor_m` | `N x 3 x R` | shared observation factor |
| scalar query Jacobian | `N x 3` | one scalar query |
| vector query Jacobian | `Q x N x 3` | one `Q`-dimensional query |

Query names are sorted before processing. Returned factors, projections, and
singular values are immutable copies.

## Numerical rank and diagnostics

The numerical threshold is

```text
max(
    eps * max(B.shape) * largest_singular_value,
    relative_rank_tolerance * largest_singular_value,
).
```

The default relative tolerance is zero, so the standard floating-point matrix
rank threshold is the only truncation. A nonzero tolerance is an explicit caller
choice and is reported in the result. Every query receives two diagnostics:

- relative covariance error after retaining only the query-coupled factor; and
- relative norm of the query-orthogonal factor after projection by that query.

These values expose the finite-precision residual rather than labelling it
mathematically zero by convention.

## Fail-closed rank cap

`maximum_query_rank` is an admission limit, not a request to truncate the exact
subspace. If the minimum numerical query rank exceeds the cap, the routine
returns

```text
query_coupled_factor_m = original full factor
query_orthogonal_factor_m = empty factor
fallback_reason = exact-query-rank-exceeds-cap
```

Thus a caller never receives an under-ranked object labelled exact.

## Relation to approximate query-preserving compression

`prob4d.query_preserving_compression` solves a different problem. It searches for
a low-rank approximation satisfying frozen observation-trace and per-query loss
limits. That routine can trade a declared amount of covariance fidelity for rank
reduction and fails closed to the full factor when no approximation is admitted.

The exact decomposition instead gives the minimum registered-query subspace and
its complete orthogonal complement. It introduces no covariance approximation
when both components are retained. The exact result can be used as a lower-bound
check on an approximate rank budget, but it does not replace the approximate
compressor's observation-space admission limits.

## Scientific and repository boundary

This module is an experimental source-side diagnostic and is intentionally not
part of `prob4d.api.v2` or the calibrated provider-v2 exporter. BayesianPhysTwin
owns the physical query, fixed local linearization, prior, guarded update, and
fallback. Causal4D consumes only an accepted downstream belief.

A successful decomposition does not establish provider mean accuracy, material
identity, uncertainty calibration, fresh-object transfer, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
