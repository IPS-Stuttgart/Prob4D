# Prior-anchored query messages and unknown-dependence fusion

Prob4D can already condition a fixed Gaussian query directly through a structured
innovation operator, and it can reduce one shared covariance factor while
preserving that posterior. This module adds the downstream representation needed
when several overlapping prediction windows produce query posteriors whose
cross-window dependence is not known well enough to justify information
addition.

The implementation is in `src/prob4d/query_message.py`.

## Exact query message

Let a fixed registered query have anchor prior

\[
q \sim \mathcal N(m_0,P_0)
\]

and let one full, correlation-aware update produce

\[
q\mid y \sim \mathcal N(m_1,P_1).
\]

Assume the query coordinates have been reduced to an independent basis, so
\(P_0\) and \(P_1\) are positive definite. Define the prior-relative natural
parameter increment

\[
\Lambda = P_1^{-1}-P_0^{-1},\qquad
\eta = P_1^{-1}m_1-P_0^{-1}m_0.
\]

Then applying the anchor prior once gives

\[
P_{\mathrm{msg}} =
  \left(P_0^{-1}+\Lambda\right)^{-1}=P_1,
\]

\[
m_{\mathrm{msg}} =
  P_{\mathrm{msg}}\left(P_0^{-1}m_0+\eta\right)=m_1.
\]

`compress_gaussian_query_posterior` constructs this message from
`GaussianQueryPosterior` and independently reconstructs the posterior before
returning it. The retained payload is one \(Q\times Q\) symmetric information
increment and one \(Q\)-vector. The explicit anchor adds one prior covariance and
mean.

The message is deliberately prior-bound and query-bound. Applying it to another
prior would generally be wrong because nuisance variables eliminated by the
full update can couple the observation to the query through the original prior.
The implementation therefore stores the anchor and rejects a supplied prior
unless its float64 arrays are byte-identical.

## Covariance intersection in message coordinates

Suppose messages \(i=1,\ldots,M\) share the same anchor and query but their
cross-message dependence is unknown. Each component posterior has

\[
J_i=P_i^{-1}=J_0+\Lambda_i,\qquad
h_i=J_i m_i=h_0+\eta_i,
\]

where \(J_0=P_0^{-1}\) and \(h_0=J_0m_0\).

For nonnegative weights \(w_i\) with \(\sum_i w_i\leq 1\), assign the remaining
weight

\[
w_0=1-\sum_i w_i
\]

to the unchanged anchor prior. Covariance intersection over the component
posteriors and the prior gives

\[
J_{\mathrm{CI}}
  = w_0J_0+\sum_i w_iJ_i
  = J_0+\sum_i w_i\Lambda_i,
\]

\[
h_{\mathrm{CI}}
  = w_0h_0+\sum_i w_ih_i
  = h_0+\sum_i w_i\eta_i.
\]

Thus the common prior is counted exactly once. No high-dimensional state,
measurement covariance, or nuisance block must be reconstructed after the
individual query messages have been formed.

An important idempotence check follows immediately. If all component messages
are byte-identical and the message weights sum to one, the fused posterior is
exactly the single-message posterior. Repeating one overlapping-window result
cannot create additional confidence. In contrast, adding the message
information twice produces

\[
J_{\mathrm{naive}}=J_0+2\Lambda,
\]

which is spuriously more precise unless \(\Lambda=0\).

`fuse_gaussian_query_messages_covariance_intersection` implements caller-supplied
weights. `select_pairwise_covariance_intersection` provides a deterministic
grid search minimizing query-covariance log determinant or trace. Weight
selection is part of the scientific protocol; target outcomes must not be used
to tune it.

## Example

```python
from prob4d.query_message import (
    apply_gaussian_query_message,
    compress_gaussian_query_posterior,
    fuse_gaussian_query_messages_covariance_intersection,
)
from prob4d.query_posterior import condition_gaussian_query

first_posterior = condition_gaussian_query(...)
second_posterior = condition_gaussian_query(...)

first = compress_gaussian_query_posterior(
    first_posterior,
    query_id="rope-endpoint-at-horizon-8",
    prior_id="physical-prior-frame-120",
    evidence_ids=("cut3r-window-112-120",),
)
second = compress_gaussian_query_posterior(
    second_posterior,
    query_id="rope-endpoint-at-horizon-8",
    prior_id="physical-prior-frame-120",
    evidence_ids=("cut3r-window-116-120",),
)

fused = fuse_gaussian_query_messages_covariance_intersection(
    (first, second),
    weights=(0.5, 0.5),
)
query_belief = apply_gaussian_query_message(fused)
```

The two windows may overlap completely, partially, or not at all. The CI call
does not assert independence. A scientifically justified independence model
could be sharper, but it must be represented and validated separately rather
than inferred from distinct window identifiers.

## Relationship to shared-factor compression

The existing `posterior_preserving_compression` module answers a different
interface question:

- keep all measurement rows;
- keep the structured Woodbury consumer;
- replace one shared factor \(U\) by a lower-rank \(UV\);
- preserve one fixed query posterior.

A query message is preferable when the downstream consumer needs only that
fixed query and does not need to re-score the observation, change the query, or
reuse the factor under another prior. It is smaller and cheaper online because
the structured solve has already been completed.

A compressed shared factor is preferable when the factor interface itself must
remain resident, several compatible operations still consume it, or the query
may change within the factor's registered validity boundary.

The two operations compose:

1. use the exact or posterior-preserving structured factor to obtain
   `GaussianQueryPosterior`;
2. convert the result into a prior-anchored query message;
3. fuse overlapping-window messages with registered CI weights.

## Complexity and storage

For query dimension \(Q\) and \(M\) messages:

- message payload: \(Q^2+Q\) float64 values;
- explicit anchor: another \(Q^2+Q\) values;
- message application: \(O(Q^3)\);
- fixed-weight CI construction: \(O(MQ^2)\) plus one \(O(Q^3)\) application;
- pairwise grid selection with \(G\) weights: \(O(GQ^3)\).

The expensive full correlation-aware conditioning is still performed once per
source message. This layer compresses and composes its query consequence; it
does not make the provider covariance free.

## Claim boundary

This implementation establishes Gaussian algebra only.

It does not establish:

- correctness or calibration of the provider covariance;
- unbiasedness required for a statistical covariance-intersection guarantee;
- that a chosen query is sufficient for another query or action;
- preservation of the observation marginal likelihood;
- conditional independence between messages;
- validity under a different prior, row identity, linearization, or robust
  weighting;
- nonlinear-query accuracy beyond the supplied local Gaussian approximation;
- robot safety or state-of-the-art prediction.

Singular or duplicated query coordinates must first be reduced to an independent
basis. Invalid anchors, negative information increments, incompatible query
identities, incompatible priors, and weights summing above one fail closed.

## Registered source-only DOT protocol

The terminal routed-camera study found the intended rank-6 structure in only
one of ten R11--R20 source sequences. The content-addressed development protocol
`protocols/dot-r11-r20-query-message-ci-source-v1.json` therefore does not
condition eligibility on a fixed rank defect.

The protocol binds:

- the exact sealed routed CUT3R provider artifact from run `33552798863`;
- the already-open source cohort R11--R20 and the unchanged R11--R20 archive
  checksum;
- the two overlapping windows and their identity-bound common anchor prior;
- a gauge-insensitive centroid query and a gauge-sensitive adverse query;
- equal-weight query-space CI as the primary source method, selected without
  source outcomes;
- dense/continuous, single-window, naive-independent, diagonal, pairwise-CI,
  and exact-fallback comparators;
- complete sequence as the independent unit; and
- posterior parity, duplicate idempotence, RMSE, joint NLL, coverage and width,
  harmful-update, byte, latency, and factor-rank-stratified endpoints.

Protocol ID:

```text
50feac697139ab4b61942df8b4e32611fc0f41da2d29884ed81753dd73e77706
```

The protocol contains no execution request. R21--R30 confirmation and R31--R70
reserve remain closed. Source completion cannot authorize confirmation; a
separate one-shot protocol must be frozen before any confirmation access.

The positive paper target is not “rank 6 occurs.” It is that a fixed
query-message interface retains useful decision-relevant uncertainty across
changing factor ranks while avoiding the overconfidence caused by overlapping
evidence.
