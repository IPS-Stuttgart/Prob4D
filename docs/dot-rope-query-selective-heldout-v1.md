# DOT R11-R30 query-selective learned-provider experiment

## Question

Can one source-frozen, learned CUT3R relative-gauge factor be used for a
registered physical query that is insensitive to its unresolved rope-axis
rotation while returning the complete prior/fallback for an off-axis query that
remains sensitive to that direction?

This is the learned-provider continuation of the held-out DLO4/DLO5
query-observability result. It is not another covariance tournament.

## Prerequisite

R11-R30 cannot be opened unless the independently verified R04-R10 dependence
confirmation has decision `heldout-strong-positive`. The later execution request
must bind the exact workflow run, artifact ID/name/digest, evaluation ID,
marker-support ID, protocol ID, and source calibration ID. A directional,
mixed, support-negative, technical, missing, or expired prerequisite leaves
R11-R70 closed.

R04-R10 is never reused to tune the R11-R30 factor, rank threshold, prior, query,
gate, method, score, or decision threshold.

## Frozen target scope

- R11-R20: `R11-20.zip`, MD5 `23ce3e7067465d3edabe20b4c7cfa388`.
- R21-R30: `R21-30.zip`, MD5 `8aee77f79d1aff6e1f3fd21886b251a0`.
- R31-R70: reserved and unopened.
- Independent statistical unit: one complete DOT sequence.

## Prediction-first custody

1. On the protected `gpuserver4090` runner, CUT3R opens only normal-view images
   and seals point maps for R11-R30. No marker payload is opened.
2. A hosted factor-seal process opens only the frozen 2-D marker locations on
   overlap frames 3-5. Those locations select corresponding points from the
   already sealed window-A and window-B point maps. The process fits the partial
   Sim(3) factor, constructs both query points, makes every admission decision,
   and seals every mean/covariance prediction. It never reads a 3-D marker.
3. A separate hosted process verifies the seal, then opens the registered 3-D
   marker outcomes on metric-fit frames 1-2 and 6-7 and scores the already sealed
   predictions once.

The factor construction is therefore learned-provider and 3-D-outcome-blind,
but 2-D-marker-localized. It is not described as fully marker-free.

## Queries and arms

The factor maps window-B coordinates to window-A coordinates. The frozen rank
threshold is `0.01`, the prior standard deviations in the centroid-normalized
local chart are `[0.05, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15]`, and the query gate
requires direct observability fraction at least `0.90`.

The two queries are:

- `centerline_centroid`: the provider overlap centroid, expected to be locally
  invariant to axial rotation;
- `off_axis_probe`: the centroid plus one provider cloud scale in a
  deterministic normal direction, expected to depend on axial rotation.

The frozen arms are physical fallback, full-rank-only use, unconditional
observable-subspace use, query-aware use with exact fallback, and deliberately
invalid full-rank completion.

## Registered outcome

At least 18 of 20 complete sequences must be supported. A strong positive
requires at least 90% centroid admission, at least 90% off-axis rejection,
positive complete-sequence bootstrap lower bounds for both centroid RMSE and
normalized Gaussian NLL improvement versus fallback, exact mean and covariance
fallback for every rejection, and a directional off-axis failure of
unconditional partial-factor use. A bounded positive omits only the last
negative-control requirement. Every other outcome is retained as mixed,
negative, or insufficient support; no target-side change is permitted.

## Claim boundary

A positive result supports selective use of a rank-deficient learned-provider
factor for the registered query under this DOT protocol. It does not establish
fully marker-free factor construction, raw covariance calibration,
BayesianPhysTwin or Causal4D benefit, deployment safety, arbitrary-DLO transfer,
or state of the art.
