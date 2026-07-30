# Strict claim-bearing observation loading

`prob4d.provider_v2` separates ordinary contract loading from admission into a
claim-bearing experiment. `load_observation_belief_export` continues to validate
the neutral `ObservationBeliefV1` content address and remains appropriate for
frozen or explicitly exploratory artifacts. New prospective experiments should
instead use:

```python
from prob4d.provider_v2 import load_claim_bearing_observation_belief

validated = load_claim_bearing_observation_belief(
    "outputs/held-out/observation_belief.npz"
)
observation = validated.observation
```

The implementation remains in `prob4d.provider_v2_loading`, but the canonical
public import is the safe-by-default provider-v2 namespace. This keeps export and
admission semantics discoverable from one versioned surface.

The strict loader rejects the artifact unless all of the following agree:

- the source repository and strict Prob4D causal stream identity;
- causal-stream contract version 2 and the exclusive frame cutoff;
- selected source-window lineage with no future payload access;
- canonical shared joint-gauge factor names and one factor group;
- the sequential joint spanning-tree gauge model with preserved cross-window
  covariance and no approximate fixed-lag boundary;
- complete gauge and point calibration metadata;
- calibration of every admitted gauge alignment;
- no permission for uncalibrated covariance or pointwise fallback;
- no recorded covariance-fallback use;
- the same calibration artifact IDs in the observation and provider attestation;
- canonical covariance roots and analytic composition Jacobians; and
- an independently verified runtime revision equal to the observation source
  revision.

The returned `ValidatedClaimBearingObservation` also exposes the validated
provider manifest ID, gauge and point calibration IDs, runtime revision, and
observation artifact ID. It does not replace independent validation in
Bayesian-PhysTwin, nor the baseline-relative accept/fallback decision.

## Incremental factor streams

For repeated causal observation times, use
`ObservationFactorStreamV1` from the same provider-v2 namespace. The stream binds
non-overlapping schema-v4 factor bundles through a previous-update hash chain and
revalidates every referenced manifest and payload. See
[append-only observation-factor streams](observation-factor-stream.md).

## Immutable metadata

`ObservationBeliefExportV1.metadata` is normalized as finite JSON and recursively
frozen after construction. Nested values remain compatible with ordinary
`isinstance(value, dict)` and `isinstance(value, list)` checks, but every in-place
mutation raises `TypeError`. Caller-owned inputs are defensively copied, so
modifying the original metadata cannot change an existing observation or its
content address.

`copy.copy` and `copy.deepcopy` return ordinary mutable JSON containers. This
keeps workflows such as `metadata = deepcopy(observation.metadata)` followed by
`dataclasses.replace(...)` available without weakening the immutable artifact.

## Compatibility

The neutral observation schema and provider-v1 behavior are unchanged. Artifact
IDs for unchanged metadata remain byte-for-byte stable because descriptors still
contain ordinary canonical JSON. Strict provider-v2 loading and factor streams
are additive and must be selected explicitly by new claim-bearing workflows.
