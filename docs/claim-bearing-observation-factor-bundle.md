# Claim-bearing observation-factor bundles

`ObservationFactorBundle` schema v4 is a neutral, reusable representation of
unfused Prob4D observations. It stores conditional point covariance, explicit
`Sim(3)` gauge Jacobians, and one ordered joint gauge covariance, but ordinary
schema validation alone does not establish that a bundle belongs to a calibrated,
causal provider-v2 run.

New prospective factor experiments can add that admission boundary with
`prob4d.provider_v2_factor_bundle`:

```python
from prob4d.provider_v2_factor_bundle import (
    load_claim_bearing_observation_factor_bundle,
    write_claim_bearing_observation_factor_bundle,
)

validated = write_claim_bearing_observation_factor_bundle(
    bundle,
    "outputs/case-a/factors.claim.json",
    causal_selection=selection,
    gauge_covariance_calibration=gauge_calibration,
    point_uncertainty_calibration=point_calibration,
)

reloaded = load_claim_bearing_observation_factor_bundle(
    "outputs/case-a/factors.claim.json"
)
stacked = reloaded.bundle.stack()
```

The neutral bundle manifest and NPZ payload remain normal schema-v4 artifacts.
The additional JSON envelope is path-independent in content identity and binds:

- the exact bundle-manifest and payload SHA-256 digests;
- schema, case, stream, sequence, revision, cutoff, factor count, observation
  count, and ordered gauge identities;
- `joint-cross-window` covariance semantics;
- independently decoded causal source-window lineage with zero future-payload
  access;
- the complete content-addressed provider-v2 manifest and attestation;
- exact gauge and point calibration artifact IDs; and
- independently verified runtime-revision evidence.

The local relative path to the neutral manifest is retrieval metadata and is not
part of the envelope artifact ID. Moving an unchanged envelope, manifest, and
payload tree therefore preserves identity. Path traversal, manifest changes,
payload changes, contradictory duplicated provenance, marginal-only covariance,
and factors outside their source-window interval all fail closed.

## Production and orchestration surfaces

`write_claim_bearing_observation_factor_bundle` is the production entry point. It
checks prediction/calibration compatibility from the selected prediction manifest
and requires an installed VCS revision or clean source checkout matching the
bundle revision before it writes output files.

`seal_claim_bearing_observation_factor_bundle` is a lower-level orchestration
surface. It accepts an already constructed claim-bearing provider attestation and
is useful for tests or a producer that has already performed the same validation.
It still validates the complete attestation, lineage, covariance semantics, and
bundle/envelope equality.

Consumers should use `load_claim_bearing_observation_factor_bundle`, not the
neutral loader, before treating factors as claim-bearing evidence. The returned
object exposes the validated provider-manifest ID, calibration IDs, runtime
source, neutral bundle, and envelope artifact ID.

## Downstream covariance rule

A Bayesian estimator retaining gauge errors as explicit nuisance variables must
use:

```text
conditional_world_covariance_m2
+ gauge_jacobian
+ gauge_prior_covariance
```

It must not also use `marginal_world_covariance_m2`, because that would add the
same gauge uncertainty twice. Association probability, source-side prior
reliability, nominal-component probability, and composite information weight
remain separate inputs.

## Claim boundary

A valid envelope establishes producer identity, byte integrity, causal ordering,
calibration identities, joint gauge-covariance representation, and independently
verified runtime provenance. It does not establish state identifiability, target
calibration, safe assimilation, accepted-update benefit, or improved Causal4D
intervention prediction. Bayesian-PhysTwin remains responsible for the physical
linearization, nuisance-aware update, baseline-relative guard, and exact fallback.
