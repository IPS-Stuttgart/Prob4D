# Gauge-propagation readiness

Prob4D distinguishes two valid ways to carry uncertain `Sim(3)` gauges into a
physical query:

1. retain the explicit gauge latent; or
2. marginalize it through a frozen first-order approximation.

The second route is an approximation and must not be admitted merely because the
source covariance decomposition looks calibrated. A nonlinear gauge posterior can
make a Jacobian-propagated covariance inaccurate even when the underlying shared
and conditional covariance components are otherwise well specified.

`prob4d.gauge_propagation_readiness` binds that missing prerequisite into the
prospective fresh-provider information order.

## Information order

For a first-order-marginalized provider, use the following order before any
protected target is opened:

1. pass support feasibility and source mean/identity competence;
2. run source covariance localization;
3. require the source `gauge-dependence` result to pass;
4. generate a source-only nonlinear-versus-linearized `Sim(3)` certificate for
   the exact frozen query projection;
5. bind the certificate to the provider, cohort, source covariance localization,
   source group roster, and query definition; and
6. compose the propagation result with the source covariance gates.

If the propagation result fails or is technically incomplete, the
`point-covariance` gate remains `not-evaluated`. This prevents nonlinear gauge
error from being misclassified as evidence authorizing a richer conditional point
model.

An explicit-gauge-latent implementation does not need a linearization certificate.
It still receives a content-addressed propagation record so the readiness request
states explicitly that no first-order marginalization was used.

## Strict certificate verification

The original diagnostic emits a content-addressed JSON certificate. The strict
loader replays all fields and its identity, rejects duplicate JSON keys, validates
point and query diagnostic rosters, and detects tampering:

```bash
python -m prob4d.diagnostics.sim3_linearization_certificate \
  source-linearization.json
```

Add `--fail-on-inadequate` when a valid but inadequate certificate should return
status 2.

## Required source-only binding

A certificate used for fresh-provider readiness must include this exact nested
metadata object:

```json
{
  "fresh_provider_readiness_binding": {
    "provider_manifest_id": "<sha256>",
    "cohort_binding_id": "<sha256>",
    "query_definition_id": "<sha256>",
    "source_covariance_localization_id": "<sha256>",
    "source_group_ids": ["object-or-session-a", "object-or-session-b"],
    "causal_prefix_only": true,
    "target_residuals_used": false,
    "target_outcomes_used": false
  }
}
```

The builder rejects mismatched provider, cohort, query, localization, or source
roster identities as invalid evidence. It also rejects noncausal bindings and any
claim that target residuals or outcomes were used. Those are not scientific
negative results.

## First-order policy

Freeze the policy before producing the certificate. For example:

```json
{
  "propagation_mode": "first-order-marginalized",
  "expected_perturbation_side": "left",
  "expected_parameter_order": ["scale", "rotation", "translation"],
  "minimum_sample_count": 4096,
  "require_query_projection": true,
  "require_supplied_jacobian_validation": true
}
```

The policy verifies the exact perturbation convention and block order, minimum
Monte Carlo sample count, presence of query-space diagnostics, and—when required—
that a supplied analytic Jacobian passed the independent finite-difference check.

Build and verify the propagation artifact:

```bash
python -m prob4d.gauge_propagation_readiness build \
  --localization source-covariance-localization.json \
  --policy gauge-propagation-policy.json \
  --query-definition-id <sha256> \
  --certificate source-linearization.json \
  --output gauge-propagation-readiness.json

python -m prob4d.gauge_propagation_readiness verify \
  --artifact gauge-propagation-readiness.json
```

A first-order certificate can produce three terminal results:

- `first-order-adequate`: permit only the declared first-order marginalization;
- `first-order-inadequate`: retain the explicit gauge latent or exact fallback;
- `technical-failure`: evidence is valid but incomplete for the frozen policy,
  such as too few samples, a missing query projection, or an unvalidated supplied
  Jacobian.

Neither failure authorizes target-side covariance inflation.

## Explicit-latent policy

The explicit route uses this canonical policy:

```json
{
  "propagation_mode": "explicit-gauge-latent",
  "expected_perturbation_side": null,
  "expected_parameter_order": [],
  "minimum_sample_count": 0,
  "require_query_projection": false,
  "require_supplied_jacobian_validation": false
}
```

No certificate is supplied. The resulting pass means only that the declared
provider retains the gauge latent rather than relying on the tested approximation.
It does not establish calibration or physical-query benefit.

## Gate composition

Use `compose_source_covariance_readiness_gates` with the exact
`FreshProviderCohortLockV1`. It verifies the cohort and query bindings and applies
the strict stage order:

```python
from prob4d.gauge_propagation_readiness import (
    compose_source_covariance_readiness_gates,
)

gauge_gate, point_gate = compose_source_covariance_readiness_gates(
    cohort_lock,
    source_covariance_localization,
    gauge_propagation_readiness,
)
```

Only a passing source gauge/dependence result and a passing propagation result
allow the conditional point-covariance result to enter the final fresh-provider
readiness request. A source gauge/dependence failure remains terminal and cannot
be overridden by a separately adequate linearization certificate.

## Scientific boundary

These artifacts are source-only readiness and failure-localization evidence. They
do not establish real-provider competence, target calibration, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
They do not reopen the already-used MotionCrafter interactions or Deform360 source
objects, and they do not authorize a new point-uncertainty model unless the
remaining conditional point-covariance gate is localized after every earlier gate
passes.
