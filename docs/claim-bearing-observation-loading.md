# Strict claim-bearing observation loading

`prob4d.provider_v2_loading` separates ordinary contract loading from admission into a
claim-bearing experiment. `load_observation_belief_export` continues to validate the
neutral `ObservationBeliefV1` content address and remains appropriate for frozen or
exploratory artifacts. New prospective experiments should instead use:

```python
from prob4d.provider_v2_loading import load_claim_bearing_observation_belief

validated = load_claim_bearing_observation_belief(
    "outputs/held-out/observation_belief.npz"
)
observation = validated.observation
```

The strict loader rejects the artifact unless all of the following agree:

- the source repository and strict Prob4D causal stream identity;
- causal-stream contract version 2 and the exclusive frame cutoff;
- selected source-window lineage with no future payload access;
- canonical shared joint-gauge factor names and one factor group;
- the sequential joint spanning-tree gauge model with preserved cross-window
  covariance and no approximate fixed-lag boundary;
- calibrated covariance metadata and the same two calibration artifact IDs in the
  provider attestation;
- a calibrated provider-v2 attestation using canonical covariance roots and analytic
  composition Jacobians; and
- an independently verified runtime revision equal to the observation source
  revision.

The returned `ValidatedClaimBearingObservation` also exposes the validated provider
manifest ID, gauge and point calibration IDs, runtime revision, and observation
artifact ID. It does not replace downstream Bayesian-PhysTwin validation or its
baseline-relative accept/fallback decision.

## Immutable metadata

`ObservationBeliefExportV1.metadata` is normalized as finite JSON and recursively
frozen after construction. Nested values remain compatible with ordinary
`isinstance(value, dict)` and `isinstance(value, list)` checks, but every in-place
mutation raises `TypeError`. Caller-owned inputs are defensively copied, so modifying
the original metadata cannot change an existing observation or its content address.

`copy.copy` and `copy.deepcopy` return ordinary mutable JSON containers. This keeps
existing workflows such as `metadata = deepcopy(observation.metadata)` followed by
`dataclasses.replace(...)` available without weakening the immutable artifact.

## Compatibility

The neutral observation schema and provider-v1 behavior are unchanged. Artifact IDs
for unchanged metadata remain byte-for-byte stable because descriptors still contain
ordinary canonical JSON. The strict loader is additive and must be selected
explicitly by new claim-bearing workflows.
