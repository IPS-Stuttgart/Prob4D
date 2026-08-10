# Provider-v2 factor-contract conformance corpus

Prob4D packages a data-only normative corpus for the advanced provider-v2
factor boundary under:

```text
prob4d/contract_data/provider_v2_factors_v1
```

The corpus fixes one minimal valid joint-gauge/tree-sparse vector and ten
adversarial mutations. Its content-addressed bundle identity is:

```text
fe0374f46319287e3709497de9cbb73f7497286cf4f157f246096f2c352e4446
```

Producer materialization uses the stable `prob4d.api.v2` surface now available
on `main`; consumers can validate the same bytes independently without relying
on Prob4D implementation-module layout.

The valid vector covers:

- provider API version 2 and provider-factor API version 2;
- schema-v4 `ObservationFactorBundle` rows;
- two ordered `Sim(3)` gauges with nonzero cross-window covariance;
- an equivalent causal square-root gauge tree;
- distinct association probability and source reliability;
- an exclusive causal frame cutoff;
- sparse and tree-sparse row stacking;
- immutable numerical arrays;
- a fixed tree-prior identity; and
- a fixed execution-stack digest.

The invalid corpus covers future-dependent rows, duplicate point identities,
invalid probabilities, indefinite point covariance, unknown gauges, drifted
joint-covariance marginals, inconsistent correlation-group settings, invalid
tree parents, dense/tree covariance mismatch, and gauge-order substitution.

## Verification

Inspect the installed corpus with:

```bash
python -m prob4d.provider_v2_contract_bundle --compact
```

The verifier checks every member digest, the aggregate bundle identity, the
valid vector's prior and execution identities, and all producer-owned
construction paths. Downstream repositories can carry the same JSON bytes while
retaining independent validators and expected accept/reject decisions.

A schema or interpretation change requires a new corpus version and bundle
identity. Editing only one repository's validator is intentionally insufficient
for a future cross-repository promotion.

## Boundary

The corpus is contract and interoperability evidence. Passing it does not
establish observation accuracy, covariance calibration, object/session
transfer, BayesianPhysTwin benefit, Causal4D intervention benefit, deployment
safety, or state of the art.
