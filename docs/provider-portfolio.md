# Evidence-first provider portfolio budget

Prob4D can now record a content-addressed portfolio of provider families through
`prob4d.provider_portfolio`. The artifact prevents provider breadth from growing
without an explicit scientific budget and preserves the ordered readiness gates:

```text
support
  -> means
  -> identity
  -> gauge-dependence
  -> conditional-covariance
  -> query-value
```

The default policy allows at most one active primary provider and one active
alternative. An alternative may be active only while a primary is active. Other
providers must be parked, promoted, rejected, or archived.

## Build and verify

Start from the checked-in
[portfolio specification](examples/provider-portfolio-spec.json):

```bash
python -m prob4d.provider_portfolio build \
  docs/examples/provider-portfolio-spec.json \
  --output outputs/provider-portfolio.json

python -m prob4d.provider_portfolio verify \
  outputs/provider-portfolio.json

python -m prob4d.provider_portfolio summarize \
  outputs/provider-portfolio.json
```

Publication is atomic and no-clobber. Repeating an identical write is
idempotent; a different artifact at the same destination is rejected.

## Gate semantics

Every provider has exactly one record for every gate. Decisions are one of:

- `not-started`, without an evidence digest;
- `in-progress`, without an evidence digest;
- `passed`, with the exact evidence digest; or
- `failed`, with the exact negative-result digest.

A provider may advance only after every preceding gate has passed. An active
provider has exactly one in-progress gate. A rejected provider has exactly one
failed gate and no later decisions. Parked and archived providers retain a
completed prefix but no active or failed gate.

Conditional point-covariance development has an additional stop rule. It may be
in progress only when
`point_covariance_development_authorized=true`, which is intended to bind the
existing source-localization result. A support, mean, identity, or
gauge/dependence failure therefore cannot be converted into permission for a
more complex point model.

## Information boundary

A valid portfolio establishes governance, ordering, and content identity only.
It does not run a provider, evaluate accuracy, authorize protected target access,
calibrate uncertainty, select a BayesianPhysTwin update, or establish Causal4D
benefit. Negative provider evidence remains a complete terminal result and must
not be rescued by activating another unbudgeted branch on the same opened target.
