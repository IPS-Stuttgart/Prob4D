# Query-space covariance relevance

Prob4D preserves conditional observation covariance and shared covariance modes
separately. A complete joint representation can be scientifically important even
when it has little effect on one downstream physical query. Conversely, a shared
mode that appears modest in observation space can dominate the uncertainty of a
query that is sensitive to that mode.

`prob4d.query_covariance_relevance` provides a neutral, experimental diagnostic
for this distinction. The downstream consumer supplies its own local query
Jacobian; Prob4D projects the covariance but does not define the query, choose a
covariance treatment, or authorize an update.

## Covariance decomposition

For `N` three-dimensional observation rows, let

```text
C_y = blockdiag(D_1, ..., D_N) + U U^T,
```

where `D_i` is the conditional covariance of row `i` and `U` contains shared
uncertainty modes. For a `Q`-dimensional query with block Jacobians `J_i`, the
module computes

```text
C_conditional = sum_i J_i D_i J_i^T,
U_query       = sum_i J_i U_i,
C_shared      = U_query U_query^T,
C_total       = C_conditional + C_shared.
```

The implementation accepts:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `query_jacobian` | `Q x N x 3` | local derivative blocks for the query |
| `local_covariance_m2` | `N x 3 x 3` | conditional row covariance |
| `low_rank_factor_m` | `N x 3 x R` | shared observation-space covariance factor |

A two-dimensional `N x 3` Jacobian denotes a scalar query. Rank zero is valid.
The implementation never materializes the dense `3N x 3N` observation covariance
or a dense `Q x 3N` Jacobian.

## Example

```python
import numpy as np

from prob4d.query_covariance_relevance import (
    project_joint_covariance_to_query,
)

query_jacobian = np.array([[1.0, 0.0, 0.0]])
local_covariance = np.diag([1.0, 2.0, 3.0])[None, ...]
shared_factor = np.array([[[2.0], [0.0], [0.0]]])

projection = project_joint_covariance_to_query(
    query_jacobian,
    local_covariance,
    shared_factor,
)

assert projection.shared_trace_fraction == 0.8
summary = projection.summary()
```

The returned `QueryCovarianceProjectionV1` owns immutable copies of all retained
arrays and replays every derived rank, trace, fraction, and covariance identity
when constructed directly.

## Streaming large observations

Dense physical queries can have many observation rows even though their output
dimension `Q` and shared rank `R` are small. Use
`project_joint_covariance_blocks_to_query` to avoid retaining a complete
`Q x N x 3` Jacobian in memory:

```python
from prob4d.query_covariance_relevance import (
    project_joint_covariance_blocks_to_query,
)

projection = project_joint_covariance_blocks_to_query(
    (
        (query_jacobian[:, start:stop], local_covariance[start:stop], shared_factor[start:stop])
        for start, stop in row_blocks
    )
)
```

Each block is validated independently and must use the same query dimension and
shared-factor rank. The iterable is consumed exactly once. Only compensated
`Q x Q` and `Q x R` accumulators are retained, so peak memory is independent of
the total row count apart from the caller-owned current block. Empty streams,
malformed blocks, and dimension or rank drift fail closed.

## Reported relevance measures

The compact summary contains:

- conditional, shared, and total query-covariance traces;
- shared trace fraction;
- shared-to-total Frobenius-norm fraction;
- one shared variance fraction per query coordinate;
- effective total and shared ranks; and
- minimum, mean, and maximum directional shared fractions on the numerically
  supported query subspace.

The directional quantities are the generalized eigenvalues of the shared
component relative to total query covariance after whitening the supported
subspace. The maximum therefore identifies whether any query direction is
strongly controlled by shared uncertainty, even when the trace-average fraction
is small. A query with zero total variance reports fractions as `None` rather
than inventing a value.

These measures answer different questions. Trace fraction is an average variance
share in the supplied coordinates. Coordinate fractions are useful for named
outputs but depend on the coordinate system. Directional fractions are invariant
to orthogonal changes of query basis. None of them is a selection threshold by
itself.

## Repository boundary

Prob4D owns the observation covariance decomposition and this projection. A
BayesianPhysTwin consumer owns the physical query Jacobian, practical-equivalence
margin, source/calibration study, computational trade-off, and exact fallback
rule. Causal4D should consume only the accepted physical belief rather than use
this diagnostic to bypass BayesianPhysTwin admission.

This is mechanism and diagnostic infrastructure. It does not establish provider
competence, calibrated uncertainty on a fresh cohort, BayesianPhysTwin physical
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
