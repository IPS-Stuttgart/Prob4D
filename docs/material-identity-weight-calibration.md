# Source calibration of material-identity weights

`prob4d.material_identity_weight_calibration` fits the source-side probabilities
used by `MaterialIdentityMixtureV1`. It closes the gap between retaining
cross-window identity hypotheses and requiring every mixture weight to be
supplied by an external calibration script.

The model is a conditional logit over one mandatory null candidate and one or
more linked source candidates. It uses only frozen prefix features and complete
source/calibration objects or acquisition sessions as independent groups. It
never consumes target outcomes and does not decide whether BayesianPhysTwin
accepts the resulting update.

## Calibration records

Each `MaterialIdentityCalibrationExampleV1` contains one labelled candidate set:

- one complete object/session `group_id`;
- exactly one null candidate and one or more linked candidates;
- one ordered feature row per candidate;
- the source-labelled true candidate ID; and
- finite metadata identifying the source label evidence.

Candidate order is canonicalized before fitting. The calibration-data identity
binds the complete canonical examples, labels, features, and feature-name order.
Changing a value, label, candidate, or source group changes the data identity.

The feature vector is deliberately generic. A real protocol can freeze features
such as source association score, overlap support, geometric residual, margin to
the second-best candidate, visibility, track age, or termination state. The
`feature_schema_id` must identify the exact feature definitions and units.

## Group-balanced cross-fitting

The fitter minimizes group-balanced conditional negative log likelihood with a
strictly positive ridge penalty. Every complete group receives equal total
weight, so a group with many tracks or candidate sets cannot dominate the fit.
Groups are assigned deterministically to cross-fitting folds.

Every training fold must contain both true-null and true-linked examples. Empty,
class-deficient, non-converged, or numerically invalid folds fail closed rather
than emitting a calibration artifact.

The retained report includes:

- source group, example, candidate, feature, and fold counts;
- optimizer iterations and convergence settings;
- cross-fitted log loss and its advantage over a uniform candidate prior;
- Brier score, top-1 accuracy, mean true-candidate probability, and top-choice
  expected calibration error;
- observed and predicted null frequency; and
- worst source-group log loss.

These are source diagnostics, not a target-calibration claim. The artifact does
not pass or fail scientific promotion by itself; the source protocol must freeze
its own acceptance thresholds before evaluation.

## Fit and replay

Start from the checked-in source-only example:

```bash
prob4d identity fit-calibration \
  docs/examples/material-identity-weight-calibration-input.json \
  --output outputs/material-identity-weight-calibration.json

prob4d identity validate-calibration \
  outputs/material-identity-weight-calibration.json
```

The output is strict, content-addressed, immutable after loading, and published
atomically without replacing different existing evidence.

## Build a calibrated mixture

Apply the retained model to prefix-only candidate features:

```bash
prob4d identity calibrate-mixture \
  outputs/material-identity-weight-calibration.json \
  docs/examples/material-identity-calibrated-mixture-config.json \
  --output outputs/material-identity-mixture.json

prob4d identity validate-mixture \
  outputs/material-identity-mixture.json
```

Application requires exact agreement in the feature schema, feature order,
association rule, tracklet-producer revision, and association revision. The
resulting mixture uses the calibration artifact ID as its `calibration_id` and
records that identity in its immutable metadata. Candidate rows are aligned only
by their local endpoint and association identities; provider-v2 point IDs are not
rewritten.

A null-only candidate set remains valid and produces probability one on the
newest-window reference. Rejected or unavailable links therefore retain the
existing exact reference behavior.

## Scientific boundary

A fitted source model establishes deterministic calibration software and
source-side diagnostic evidence only. It does not establish real provider
competence, target association calibration, BayesianPhysTwin physical-query
benefit, Causal4D intervention benefit, deployment safety, or state of the art.

Promotion still requires disjoint association-calibration, prefix-likelihood,
guard-calibration, and target groups. The decisive comparison should retain the
newest-window reference, hard-link, identity-marginalized, oracle, and exact
fallback arms. A negative held-out result is complete evidence and must not be
retuned on the same target partition.
