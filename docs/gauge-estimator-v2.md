# Gauge Estimator V2: order-invariant graph inference

Prob4D's portable causal observation stream deliberately retains the frozen
`sequential_joint_spanning_tree_v1` behavior. This document describes an additive
estimator and alignment path for controlled ablations before any production
contract is changed.

## Motivation

The original sequential estimator processes an explicit temporal window order,
but it fuses several already-initialized parent candidates through pairwise
covariance intersection. Pairwise CI is not associative, so permuting otherwise
identical constraints can change the result. The strict observation exporter also
selects one parent for each new window greedily and does not use non-tree overlap
edges except as recorded metadata.

Dense overlap alignment has a related sampling problem. When more than the
configured maximum number of correspondences are available, an unstratified
random subset can retain many redundant pixels while omitting a small region that
provides useful scale, rotation, or depth leverage.

## Additive APIs

`prob4d.gauge_graph` provides:

- `OrderInvariantSequentialGaugeEstimator`, which maps all parent candidates to a
  right-invariant local `Sim(3)` tangent and fuses them jointly with generalized
  covariance intersection;
- `select_uncertainty_volume_spanning_tree`, a deterministic global Kruskal
  selection using covariance log-volume, correspondence count, residual RMS, and
  content-addressed tie breakers;
- `estimate_joint_gauge_tree`, which propagates the selected tree into one full
  cross-window covariance;
- `loop_closure_diagnostics`, which evaluates non-tree edges against the
  tree-derived posterior without refitting it; and
- `constraint_content_id`, an orientation-independent digest for overlap edges.

`prob4d.stratified_alignment` provides:

- `stratified_overlapping_correspondences`, which allocates an exact sample budget
  over absolute-frame/spatial-tile clusters and samples each cluster across its
  depth range; and
- `align_windows_stratified`, which feeds that sample to the existing robust
  `Sim(3)` estimator and retains the frame/tile sandwich covariance semantics.

The stratified alignment path fails closed when fewer than eight independent
clusters survive. A pointwise or IID covariance fallback requires explicit
`fallback_policy="pointwise"` and is recorded in the returned alignment result.

## Example

```python
from prob4d.gauge_graph import OrderInvariantSequentialGaugeEstimator
from prob4d.stratified_alignment import align_windows_stratified

alignment = align_windows_stratified(
    reference_window,
    moving_window,
    max_correspondences=100_000,
    spatial_tile_size=32,
)

estimates = OrderInvariantSequentialGaugeEstimator().estimate(
    ordered_window_ids,
    relative_constraints,
    initial_transform=metric_anchor.global_from_local,
    initial_covariance=metric_anchor.covariance,
)
```

## Statistical boundary

These APIs improve determinism and expose graph diagnostics. They do not establish
that all dense overlap edges are independent. Generalized CI is used when several
parent estimates can have unknown correlation. The globally selected tree assumes
independent selected edge errors only for covariance propagation. Non-tree edge
NIS values additionally use an independence approximation and must be interpreted
as diagnostics until calibrated on independent sequence families.

The local coordinates use the repository's
`[log scale, shortest rotation vector, translation]` convention applied to a
right-invariant relative transform. This avoids subtracting two global rotation
vectors, but it is still a local first-order covariance approximation rather than
a new exact `Sim(3)` probability distribution.

## Promotion gate

Do not replace the portable stream's frozen gauge mode solely because these unit
tests pass. A production promotion requires a registered sequence-family-held-out
comparison covering:

1. permutation invariance and loop-fault localization;
2. gauge scale, rotation, and translation error;
3. low-dimensional gauge NIS and coverage;
4. seam error and long-horizon drift;
5. downstream Bayesian-PhysTwin acceptance, harmful-update frequency, and exact
   fallback; and
6. peak memory and runtime.

The existing stream-contract version and artifact IDs remain unchanged in this
implementation.
