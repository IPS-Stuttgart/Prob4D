# Semantic compatibility versus frozen evidence

Prob4D now distinguishes ordinary consumer compatibility from exact scientific
reproduction.

## Semantic compatibility

`prob4d prediction compatibility` describes the minimum interface semantics a
consumer requires:

- the stable `prob4d.api.v2` major and provider API majors;
- the observation-belief and provider-v2 contract identities and versions;
- named mandatory valid conformance vectors and their individual content IDs;
- required capabilities such as causal source lineage, joint gauge covariance,
  strict artifact loading, exact invalid fallback, and tree-sparse factors.

It deliberately does **not** bind the complete corpus digest or the total number
of valid and invalid vectors. A provider may add new conformance vectors and new
capabilities without invalidating an older consumer requirement. Existing named
vectors may not disappear or change.

```bash
prob4d prediction compatibility build \
  --output outputs/semantic-compatibility.json

prob4d prediction compatibility verify \
  outputs/semantic-compatibility.json \
  --require-current
```

Compare a consumer requirement against a provider manifest with:

```bash
prob4d prediction compatibility check \
  consumer-required.json \
  provider-available.json
```

The command returns success only when the provider contains every required
named vector with the same digest and every required capability.

## Claim-bearing evidence pin

A semantic-compatibility result is infrastructure evidence only. A frozen
scientific run must additionally bind:

- exact Prob4D, BayesianPhysTwin, and Causal4D revisions;
- exact wheel or source-distribution hashes;
- complete normative contract-corpus identities;
- provider/model/runtime and calibration identities;
- protocol, input, output, and decision-artifact digests.

This separation prevents ordinary development from becoming brittle when an
additional adversarial test vector is added, while preserving byte-exact
reproduction for every claim-bearing result.

## Additive rule

Compatibility is directional. A provider manifest satisfies a consumer manifest
when the API and contract major identities match and the provider is a superset
of the consumer's named vectors and capabilities. Equality of the complete
manifest is neither required nor sufficient to replace the frozen evidence pin.
