# Material-identity command line

`prob4d identity` exposes append-only identity hypotheses, source fitting of
identity weights, portable identity mixtures, and exact downstream
marginalization without promoting global material IDs or changing provider-v2
observation identities.

The commands never accept a BayesianPhysTwin update. Calibration must use
complete declared source/calibration objects or sessions and must remain separate
from target outcomes.

## Fit source-side weights

Fit and strictly validate a group-balanced conditional-logit calibration:

```bash
prob4d identity fit-calibration \
  docs/examples/material-identity-weight-calibration-input.json \
  --output material-identity-weight-calibration.json

prob4d identity validate-calibration \
  material-identity-weight-calibration.json
```

The input contains labelled candidate sets, one physical object or acquisition
session ID per example, the exact feature schema and association identities, and
an explicit `uses_target_outcomes=false` declaration. The artifact reports
cross-fitted proper scores and calibration diagnostics while retaining the
complete canonical calibration-data digest. See
[source calibration of material-identity weights](material-identity-weight-calibration.md).

## Build and validate a calibrated mixture

Apply a retained source model to one prefix-only candidate set:

```bash
prob4d identity calibrate-mixture \
  material-identity-weight-calibration.json \
  docs/examples/material-identity-calibrated-mixture-config.json \
  --output material-identity-mixture.json

prob4d identity validate-mixture material-identity-mixture.json
```

`calibrate-mixture` requires exact feature-schema, feature-order, association-rule,
tracklet-producer, and association-implementation compatibility. It computes all
candidate log weights from the retained model and binds the model artifact ID as
the mixture calibration identity.

For a frozen externally calibrated model, the lower-level configuration remains
available:

```bash
prob4d identity build-mixture \
  docs/examples/material-identity-mixture-config.json \
  --output material-identity-mixture.json
```

Every mixture requires exactly one null hypothesis. Linked endpoints retain their
original `(window_id, track_id)` identities, must precede the target window, and
bind their source association evidence. Publication is atomic and no-clobber by
default.

Validate an append-only multi-window stream independently with:

```bash
prob4d identity validate-stream material-identities.json
```

## Likelihood marginalization

Create a JSON input whose candidate IDs exactly match the mixture order:

```json
{
  "candidate_ids": ["<null candidate ID>", "<linked candidate ID>"],
  "log_likelihoods": [-2.1, -0.4],
  "likelihood_power": 1.0
}
```

Then run:

```bash
prob4d identity marginalize material-identity-mixture.json likelihoods.json
```

The output contains the stable log-sum-exp marginal likelihood and posterior
identity probabilities. Candidate-ID alignment prevents a downstream likelihood
row from being assigned to the wrong local endpoint.

## Gaussian moment matching

For consumers that require one Gaussian approximation, provide candidate-aligned
means, covariances, and probabilities:

```bash
prob4d identity moment-match material-identity-mixture.json hypotheses.json
```

The output reports the within-hypothesis covariance, the between-hypothesis
covariance, their law-of-total-covariance sum, identity entropy, and effective
hypothesis count. A null-only mixture reproduces the null mean and covariance
exactly.

## Claim boundary

The CLI makes the experimental identity path reproducible and inspectable. It
does not establish cross-window association precision, real provider competence,
BayesianPhysTwin benefit, or Causal4D benefit. Promotion still requires the
object/session-held-out comparison in the held-out provider protocol.
