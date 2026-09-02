# Tracking Cloth approximate-orbit tube v1

Status: **prospectively registered; target trajectory values unopened**.

## Question

Can a trajectory-level calibrated Euclidean tube around an estimated axial
orbit absorb departures from exact symmetry while remaining sharper and more
decision-useful than a generic point-centered conformal ball?

This is the prospective experiment corresponding to the approximate-orbit
corollary in the focused query-identifiability manuscript. It is not a learned
visual-provider experiment.

## Fresh cohort

The previous Tracking Cloth support audit inspected marker headers for 27 A2
cotton/denim/wool Self-collisions recordings but parsed no trajectory values.
Those exact 27 content-bound group IDs form this study's complete cohort.

Within each material, file names are ordered by a frozen SHA-256 rule:

- six recordings enter calibration;
- three recordings remain target-only.

This yields 18 calibration and nine target recordings. Target trajectory values
may be opened only after a content-addressed calibration artifact exists.

## Estimated orbit

For consecutive registered frames, two selected anchors define the current
axis. The previous probe state supplies:

- normalized axial fraction along the anchor segment;
- normalized radial distance from the segment.

Applying those normalized quantities to the current anchor geometry defines an
estimated circular orbit. The probe's angle around the axis remains unresolved.

The point baseline transports the previous radial direction by the deterministic
minimal rotation between consecutive anchor axes. The orbit and point baseline
therefore share the same axial/radial prediction and differ only in whether the
unresolved angle is retained.

## Calibration

For each complete calibration recording:

- orbit score: maximum point-to-estimated-circle distance;
- point score: maximum Euclidean error of the transported canonical point.

The calibration radius is the split-conformal order statistic over the 18
complete-recording maxima at requested miscoverage 0.10. Frames and marker
coordinates are nested observations, not independent calibration samples.

## Registered query

The scalar query is the probe coordinate along the current anchor axis relative
to a calibration-only threshold. The threshold is the median normalized axial
query center over calibration pairs and is sealed before target values are
opened. For this rotation-invariant affine query, an orbit tube of radius `rho`
gives the exact interval

```text
query_center ± rho.
```

A generic point ball gives the same center but radius `R`. Any width or
threshold-decision difference is therefore caused by preserving the residual
gauge rather than by changing the point prediction.

## Comparators

1. exact estimated orbit (`rho = 0`);
2. group-calibrated approximate-orbit tube;
3. group-calibrated canonical-point ball.

The primary endpoints are complete-recording simultaneous coverage, scalar
interval width, threshold-decision admission, and harmful accepted decisions.

## Claim boundary

A positive outcome would establish a public real-trajectory calibration result
for an estimated orbit under a controlled observation restriction. It would not
establish visual discovery of the orbit, complete physical-state recovery,
BayesianPhysTwin or Causal4D benefit, closed-loop robot performance, deployment
safety, or state of the art.
