# Provider-neutral prediction manifests

Prob4D can admit decoded 4-D predictions without making MotionCrafter-specific
fields part of every downstream interface. The portable
`PredictionProviderManifestV1` binds the provider, exact model and loader
identities, stochastic and dependence semantics, canonical prediction payload
bytes, and the source frames used by every output frame.

This contract is additive. Existing MotionCrafter `predictions.json` manifests,
provider-v1/provider-v2 observation artifacts, and frozen experiment identities
remain unchanged.

## Information boundary

A neutral manifest contains:

- an exact provider repository and 40- or 64-character revision;
- content identities for the provider run, model set, and executed loader;
- declared coordinate, point, flow, and ray semantics;
- one or more versioned `PredictionWindow` NPZ payloads;
- shared dependence groups such as model set, input video, or stochastic member;
- exact output-frame identities; and
- an exclusive source-frame interval plus contributing source IDs for each output.

For an exclusive causal cutoff `c`, a payload is admissible only when every
output row satisfies

```text
source_frame_stop_exclusive <= c.
```

The output frame number itself is not sufficient. A frame labelled 120 can still
be inadmissible when its provider used source frame 121.

Payload paths are retrieval metadata and are deliberately excluded from the
portable payload and manifest identities. The payload SHA-256, byte count,
frame lineage, stochastic member, dependence groups, and semantic declarations
are identity-bearing. Loaders nevertheless require safe relative paths confined
to the manifest directory and reject symbolic-link traversal.

## Importing MotionCrafter output

First generate an integrity-bound MotionCrafter bundle through the existing safe
producer. Then place the neutral manifest inside the same bundle tree so every
referenced payload has a confined relative path:

```bash
prob4d prediction import-motioncrafter \
  outputs/sequence-a/predictions.json \
  outputs/sequence-a/provider-neutral.json \
  --sequence-id sequence-a \
  --view-id camera-0

prob4d prediction validate \
  outputs/sequence-a/provider-neutral.json \
  --causal-frame-stop 134
```

The importer first replays MotionCrafter's existing artifact-integrity checks. It
requires an exact immutable model-set identity, exact loader-module identity,
complete run-spec identity, member hashes and byte counts, and the recorded seed
schedule. Legacy unbound prediction trees are not promoted through this route.

The imported dependence groups distinguish shared model/input-video dependence
from stochastic-member identity. Different seeds from the same model are not
silently described as independent providers.

## Importing another provider

An external provider can use the same canonical contract without adding its
runtime or model-loading stack to Prob4D. It must write versioned
`PredictionWindow` NPZ payloads and a strict import specification. A complete
example is retained at
[`examples/provider-neutral-import-spec.json`](examples/provider-neutral-import-spec.json).

```bash
prob4d prediction import-spec \
  outputs/sequence-a/provider-import.json \
  outputs/sequence-a/provider-neutral.json

prob4d prediction validate \
  outputs/sequence-a/provider-neutral.json \
  --causal-frame-stop 134
```

The specification declares the provider repository and revision, provider-run,
model-set and loader identities, coordinate and field semantics, dependence
groups, and one complete source-frame interval for every output frame. Prob4D
derives each payload SHA-256, byte count, storage precision, and optional-field
presence from the canonical NPZ bytes rather than accepting those values from the
caller.

The specification and all payloads must share a confined bundle root with the
output manifest. Absolute paths, parent traversal, symlink traversal, duplicate
JSON keys, non-finite values, unknown fields, malformed identities, frame-lineage
mismatches, and caller attempts to set importer-owned metadata fail closed.
Payloads and the specification are hashed and statted before and after loading;
a concurrent mutation therefore cannot be admitted under the earlier identity.

## Alternative-provider semantics

An adapter for VGGT or another 4-D model must declare one of the supported
coordinate semantics:

- `window-local-sim3`;
- `sequence-local-sim3`;
- `camera-local-metric`; or
- `metric-world`.

An adapter must not relabel camera-local or arbitrary-gauge output as metric
world coordinates. It must also state whether scene flow is absent or denotes
forward point displacement and whether stored rays are absent or unit camera
rays.

A model-family disagreement diagnostic can use the neutral provider and
shared-dependence fields. Such disagreement remains a source-calibrated
reliability feature; it is not automatically an independent likelihood or a
fusion weight.

## Persistence and verification

Canonical manifests are written atomically and idempotently. A writer lock
serializes concurrent creation, and an existing different manifest is never
replaced. Validation checks the manifest byte snapshot as well as every payload
before and after `PredictionWindow` loading, then checks the window ID, frame
grid, storage precision, and optional-field declarations.

This closes a time-of-check/time-of-use gap without changing portable manifest or
payload identities. The path remains retrieval metadata; the verified byte and
scientific contracts remain identity-bearing.

## Python API

```python
from prob4d.prediction_provider_import import (
    import_prediction_provider_specification,
)
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    load_prediction_provider_manifest,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)
```

`verify_prediction_provider_manifest` checks the content identity and, by
default, reopens every exact payload byte with `allow_pickle=False`, validates
its closed versioned `PredictionWindow` schema, and checks window ID, frame grid,
storage precision, optional field declarations, and snapshot stability.

## Repository boundary and nonclaims

The neutral manifest standardizes prediction provenance and causal admission. It
does not establish that an alternative provider is accurate, calibrated, less
biased, statistically independent, or useful to BayesianPhysTwin. Provider
competence and the guarded physical-query result remain separate held-out gates.
Causal4D should continue to consume only the accepted or exact-fallback
BayesianPhysTwin belief, not raw prediction-provider output.
