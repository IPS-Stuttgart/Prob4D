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
- dense, sparse, and tree-sparse row stacking;
- immutable numerical arrays;
- a fixed tree-prior identity; and
- a fixed structural stack-semantic identity.

The invalid corpus covers future-dependent rows, duplicate point identities,
invalid probabilities, indefinite point covariance, unknown gauges, drifted
joint-covariance marginals, inconsistent correlation-group settings, invalid
tree parents, dense/tree covariance mismatch, and gauge-order substitution.

## Exact and numerical conformance

Cross-runtime conformance deliberately separates exact contract identity from
floating-point replay:

1. The corpus JSON bytes, every member digest, the aggregate bundle identity,
   row identities, factor/gauge order, probability fields, causal cutoff, dtypes,
   shapes, and tree-prior identity are exact.
2. Dense, sparse, and tree-sparse materializations must agree with absolute
   tolerance `1e-12` and relative tolerance `1e-10`. The gauge tree must also
   reconstruct the complete joint gauge covariance at those tolerances.
3. The exact byte digest of derived floating-point arrays is reported only as a
   same-runtime diagnostic. It is not a cross-runtime acceptance criterion,
   because supported NumPy or BLAS implementations may differ in final bits
   without changing any admitted provider-v2 semantics.

The valid vector retains the original reference runtime digest in its historical
`expected.stack_sha256` field. The installed verifier reports both that reference
and the observed runtime digest, but acceptance depends on the exact structural
identity and the explicit numerical checks above.

## Verification

Inspect the installed corpus with:

```bash
python -m prob4d.provider_v2_contract_bundle --compact
```

The verifier checks every member digest, the aggregate bundle identity, the
valid vector's prior and structural identities, dense/sparse/tree parity, the
joint-covariance/tree equivalence, and every producer-owned invalid path.
Downstream repositories can carry the same JSON bytes while retaining
independent validators and expected accept/reject decisions.

A schema or interpretation change requires a new corpus version and bundle
identity. Editing only one repository's validator is intentionally insufficient
for a future cross-repository promotion.

## Boundary

The corpus is contract and interoperability evidence. Passing it does not
establish observation accuracy, covariance calibration, object/session
transfer, BayesianPhysTwin benefit, Causal4D intervention benefit, deployment
safety, or state of the art.
