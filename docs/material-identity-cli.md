# Material-identity command line

`prob4d identity` exposes append-only hypothesis streams, source-calibrated
local mixtures, and bounded exact joint posteriors without promoting global
material IDs or changing provider-v2 observation identities.

The commands do not fit weights and do not accept BayesianPhysTwin updates.
Calibration must be completed on declared source/calibration objects or sessions
before target access.

## Build and validate a mixture

```bash
cp docs/examples/material-identity-mixture-config.json mixture-config.json
prob4d identity build-mixture mixture-config.json \
  --output material-identity-mixture.json
prob4d identity validate-mixture material-identity-mixture.json
```

`build-mixture` requires exactly one null hypothesis. Every linked endpoint keeps
its original `(window_id, track_id)` identity, belongs to a window before the
target, and binds its association result, source score, externally calibrated log
weight, producer revision, association revision, rule ID, and calibration ID.
The output is written atomically and is rejected rather than overwritten unless
`--overwrite` is supplied.

Validate an append-only multi-window stream independently with:

```bash
prob4d identity validate-stream material-identities.json
```

## Joint multi-window consistency

Several local mixtures can be conditioned on the exact window-unique forest
constraint before a downstream physical likelihood is evaluated:

```bash
cp docs/examples/joint-material-identity-posterior-config.json joint-config.json
prob4d identity build-joint joint-config.json \
  --output joint-material-identity.json
prob4d identity validate-joint joint-material-identity.json
```

The joint artifact embeds every source mixture, rejects assignments whose
connected component would contain two endpoints from one prediction window, and
fails before enumeration when the declared Cartesian-product bound is exceeded.
It retains the mandatory null candidate for every target mixture.

Evaluate assignment-aligned downstream likelihoods with:

```bash
prob4d identity marginalize-joint \
  joint-material-identity.json \
  joint-assignment-likelihoods.json
```

The output includes exact assignment probabilities and candidate-aligned local
marginals after global conditioning. See the
[joint material-identity posterior](joint-material-identity-posterior.md) for the
constraint, replay, and claim boundaries.

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
