# Tracking Cloth query-portfolio and cache break-even study

This experiment addresses the practical objection to posterior-preserving
shared-noise-factor compression: for a single immutable Gaussian query, directly
caching the gain and posterior covariance is simpler and usually faster.

The study uses recording-disjoint folds from the complete public Tracking Cloth
Deformation dataset. For each cloth size it fits one source-side joint Gaussian
model for all marker future displacements and the causal marker-motion
observation. It then registers deterministic portfolios of 1, 2, 4, 8, 12, and
(where available) 20 marker queries.

For every portfolio it reports:

- full-versus-compressed gain, covariance, and realized-mean parity;
- retained shared rank;
- full and compressed shared-factor payloads;
- a direct cached joint-query gain/covariance payload;
- compressed-factor payload with projection metadata;
- construction time and per-window update time under a structured Woodbury
  implementation; and
- a positive-definite generic baseline obtained by a low-rank-plus-isotropic
  spectral approximation of the complete joint covariance.

The primary payload comparison assumes that the query prior, query/observation
cross covariance, conditional covariance, and means are already resident at the
consumer. The output explicitly states that a cached query message is preferable
when those blocks are not resident or when only one immutable query is needed.

This is a real-motion-capture representation and systems-cost study. It does not
evaluate a learned 4-D provider, BayesianPhysTwin physical benefit, Causal4D
intervention benefit, deployment calibration, or state of the art.
