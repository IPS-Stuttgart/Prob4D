# CUT3R Bayesian prefix development comparison

This is a new **exploratory** experiment on already-open public DOT R01-R03.
It reuses immutable native CUT3R continuous predictions from run 33329701704;
it does not rerun or modify any closed experiment, access R04-R70, decode new
RGB, or access BayesianPhysTwin/Causal4D/held-v8/DLO4/DLO5 artifacts.

## Question

Can sparse early 3D measurements make later CUT3R reconstruction more accurate,
and does accounting for shared observation error improve Bayesian conditioning?
This is observed-frame reconstruction with sparse prefix supervision, **not**
unseen-future simulation, fully marker-free tracking, or a fresh-object result.
The current-frame RGB and released 2D query locations are common to every arm.

## Fixed comparison

- Initial metric alignment: robust proper Sim(3) using frames 1-2.
- Bayesian residual observations: frames 3-5 only.
- Scored 3D marker measurements: frames 6-7, opened only after all three
  predictions or technical dispositions have been sealed by this new runner.
- Full-prefix deterministic control: Sim(3) refit using all frames 1-5.
- Last-residual control: each identity's last valid residual anywhere in frames
  1-5, with zero correction for previously unsupported identities.
- Bayesian arms: the same zero-mean spatial residual GP, either independent
  observation noise or a shared camera-bias nuisance with correlation 0.8.
- GP covariance: `0.1^2 * (0.5 + 0.5 * exp(-distance^2 / (2 * 0.25^2)))`.
- Observation standard deviation: 0.02. Both likelihoods have the same row
  marginal variance; only their off-diagonal dependence differs.
- Coordinates are normalized by the frame-1/2 marker bounding-box diagonal.
  No unverified metre/millimetre conversion is claimed.
- No hyperparameter search, rank-defect requirement, outcome-selected camera,
  confidence-to-probability conversion, residual clipping, or automatic promotion.

The existing `condition_gaussian_query` and structured covariance operators
perform the Bayesian update. The covariance is a model assumption for this
development test, not a pre-established calibration guarantee. Deterministic
controls receive the same fixed Gaussian wrapper as the initial prior; reported
NLL does not imply that vanilla CUT3R natively outputs calibrated covariance.
The shared camera nuisance is not identified separately from the residual field.

## Safeguards and metrics

Inputs and implementations are hash-bound before measurement access. Duplicate
frame/identity observations are collapsed if exact and rejected if conflicting.
Missing early observations are not fabricated. Insufficient residual-update
support retains the exact initial-alignment prediction and covariance; insufficient
alignment or later scoring support is a retained technical failure, not replacement.

Report 3D point RMSE / prefix span, marginal 3D Gaussian NLL per coordinate,
NEES / 3, 90% ellipsoid coverage, and 90% marginal coordinate full width / span.
Aggregate equally over the two supported score frames, then over the three
sequences. Every arm uses exactly the same scored rows. A missing sequence means
only descriptive subset metrics, never a complete-denominator claim.

The spatial kernel uses provider geometry, not scored 3D truth. Marker identities
follow the existing released row-order convention. Results are temporal transfer
on previously seen identities, not a disjoint-hidden-identity test. With only
three previously opened source sequences, no confirmatory CI or SOTA claim is made.
Any positive result must beat the full-prefix alignment and last-residual controls
before it motivates a distinct, larger public-data evaluation.

## Completed result

The [retained result](../evidence/cut3r-bayesian-prefix-dev-v1/README.md) covers
all three sequences. Shared-error Bayes improves RMSE by 78.38% over full-prefix
metric alignment, but is 10.21% worse than last-residual correction overall.
The strongest simple point baseline therefore remains unbeaten in this test.
