# Controlled Gaussian moment-collapse diagnostic

Prob4D observation contracts preserve means, covariance, explicit gauge
uncertainty, association uncertainty, source reliability, and declared dependence.
A different question is whether a genuinely multimodal point likelihood would be
reduced to only its first two moments before BayesianPhysTwin consumes it.

`prob4d.moment_collapse_diagnostic` provides a controlled analytic mechanism test.
It does not claim that any real provider is multimodal and does not change an
observation schema.

## Symmetric reference mixture

For one 3-D offset `d` and positive-definite component covariance `Sigma`, the
controlled likelihood is

```text
p(x) = 0.5 N(x; -d, Sigma) + 0.5 N(x; +d, Sigma).
```

Its moment-matched Gaussian is

```text
q(x) = N(x; 0, Sigma + d d^T).
```

Both representations have the same mean and covariance, but they can assign very
different probability to the physically implausible midpoint. The diagnostic
projects onto the Fisher discriminant direction and reports:

- squared Mahalanobis offset `d^T Sigma^-1 d`;
- component-mean separation in component standard deviations;
- mixture density at the midpoint relative to a component mean;
- mixture and moment-Gaussian probability inside a frozen midpoint interval;
- central-mass inflation under moment matching; and
- moment-matched projected excess kurtosis.

A case is labelled material only when every frozen threshold passes. The result is
fully analytic and deterministic; no Monte Carlo tolerance or row order enters its
identity.

## Running the diagnostic

Use the raw configuration in
[`examples/moment-collapse-diagnostic-input.json`](examples/moment-collapse-diagnostic-input.json):

```bash
python -m prob4d.moment_collapse_diagnostic build \
  docs/examples/moment-collapse-diagnostic-input.json \
  --output outputs/moment-collapse-diagnostic.json

python -m prob4d.moment_collapse_diagnostic verify \
  outputs/moment-collapse-diagnostic.json
```

The artifact is content-addressed, strictly validates all numeric JSON members,
returns immutable NumPy arrays, rejects non-positive-definite component
covariance, and publishes without clobbering different retained evidence.

## Decision rule for future schema work

A positive controlled case establishes only that first-two-moment transport can
lose qualitative information for that constructed mixture. Adding a multimodal
likelihood payload is scientifically justified only after opened source diagnostics
show all of the following:

1. a real provider has repeatable multimodal residual structure;
2. the modes are not already explained by explicit gauge or association nuisance
   variables;
3. moment collapse materially changes a preregistered physical query or proper
   score; and
4. source/calibration support is sufficient to freeze mixture complexity before a
   target is opened.

Without that evidence, the diagnostic remains a negative control against
unnecessary contract growth.
