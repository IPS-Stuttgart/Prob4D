# Provider-v2 explicit-gauge factor facade

New explicit-gauge experiments should import their producer contracts from one
versioned surface:

```python
from prob4d.provider_v2_factors import (
    load_claim_bearing_observation_factor_bundle,
    stack_sparse_observation_factors,
)

validated = load_claim_bearing_observation_factor_bundle(
    "outputs/case-a/factors.claim.json"
)
stacked = stack_sparse_observation_factors(validated.bundle)
```

`prob4d.provider_v2_factors` collects three additive layers:

1. the neutral schema-v4 `ObservationFactorBundle` contract and checksum-bound
   JSON/NPZ I/O;
2. the claim-bearing provider-v2 envelope that binds causal lineage, calibration,
   provider attestation, and runtime provenance; and
3. the sparse in-memory stack that preserves one local `3 x 7` gauge Jacobian and
   one gauge index per row.

The facade does not create another artifact schema. Neutral factor bundles remain
usable for frozen and explicitly exploratory work. New prospective admission must
use the strict envelope before a downstream physical innovation is formed.

## Downstream covariance rule

A Bayesian estimator that keeps gauge errors explicit must consume:

```text
conditional point covariance
+ local gauge Jacobian and row gauge index
+ complete joint gauge prior
```

It must not add the gauge-marginal point covariance as another independent noise
term. Association probability, source-side prior reliability, nominal-component
probability, and composite information weight also remain separate inputs.

## Compatibility

`prob4d.provider_v2` remains the stable surface for observation-belief export,
strict observation-belief loading, provider manifests, calibration artifacts,
and factor streams. `prob4d.provider_v2_factors` is the corresponding focused
surface for explicit-gauge factor production and execution. Both report provider
API version 2.

The facade imports only the NumPy core and does not load Torch, Diffusers, or
Decord. Wheel and source-distribution builds include it automatically through
the existing package discovery.

A valid factor envelope and sparse stack establish integrity, causal ordering,
and algebraic parity. They do not establish physical-state identifiability,
target calibration, accepted-update safety, Bayesian-PhysTwin benefit, or
Causal4D benefit.
