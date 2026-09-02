# Group-level conformal risk control for selective physical updates

## Purpose

Compatibility-set coverage and safe selective use are different properties. A
continuous ambiguity tube can contain the next trajectory while abstention is
still harmful because the caller's fallback is weak. The first Tracking Cloth
continuous-SO(2) experiment exposed exactly that boundary.

This module controls a task loss directly. For calibration groups
`g=1,...,n` and a fixed candidate family ordered from least to most
conservative, let `L_g(lambda_j)` lie in `[0,B]` and be nonincreasing in `j`.
The corrected empirical risk is

```text
R_plus(lambda_j)
  = n/(n+1) * mean_g L_g(lambda_j) + B/(n+1).
```

Select the first candidate satisfying `R_plus <= alpha`. Under exchangeability
of the calibration groups and next group, and with the entire model, score,
fallback, candidate grid and group definition frozen before calibration, the
standard conformal-risk-control argument bounds the expected next-group loss by
`alpha`.

For selective physical revision, one useful bounded group loss is the fraction
of cases in a complete recording for which the update is both accepted and
worse than the registered fallback. Increasing a robust-advantage margin gives
a nested family because accepted sets can only shrink.

## What it does not guarantee

The result is not:

- a high-probability bound on the realized test-set risk;
- conditional coverage or risk conditional on acceptance;
- protection after choosing the model or grid on calibration data;
- evidence that the fallback is competent;
- a deployment-safety certificate.

The finite-sample correction also imposes a hard floor `B/(n+1)`. With 12
calibration recordings and `B=1`, the floor is `1/13 = 0.0769`. A target risk
below that value cannot be certified even with zero observed calibration loss.
This is deliberate fail-closed behavior.

## Relation to continuous symmetry

The intended prospective pipeline is:

1. freeze an identity-free physical-axis estimator on source-only labelled data;
2. calibrate the continuous axial support on one recording-disjoint cohort;
3. construct shared-angle candidate-versus-fallback advantage bounds;
4. calibrate a nested robust-margin family on a second cohort using this module;
5. evaluate exactly once on an untouched confirmation cohort;
6. return the complete registered fallback on every rejection.

Support calibration and risk control answer different questions and should be
reported separately.
