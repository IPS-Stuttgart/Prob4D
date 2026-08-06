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

## Integrity-bound VGGT export and import

The official VGGT baseline now writes a closed version-2 run record. Remote
checkpoints require an exact immutable revision, which is passed through to
`VGGT.from_pretrained`; local checkpoints are bound by their file SHA-256. The
run record additionally binds the exact VGGT checkout, executed Prob4D loader
module bytes, preprocessing mode, input-video bytes, and every cached prediction
archive. Claim-bearing export rejects modified or untracked files in the VGGT
checkout. With `--resume`, complete cached samples are validated and sealed
without importing Torch or loading the VGGT model.

```bash
prob4d vggt baseline \
  --dataset-root /data/Sintel_video \
  --output-root outputs/vggt \
  --vggt-root /opt/vggt \
  --checkpoint facebook/VGGT-1B \
  --checkpoint-revision <exact-40-character-revision> \
  --resume
```

Convert one registered sample into canonical provider-neutral payloads without
rerunning the model:

```bash
prob4d prediction import-vggt \
  outputs/vggt/run-part-00.json \
  outputs/vggt/provider/scene-a.json \
  --sequence-id scene-a \
  --sample-id scene-a/video.mp4 \
  --dataset-root /data/Sintel_video \
  --prediction-root outputs/vggt

prob4d prediction validate \
  outputs/vggt/provider/scene-a.json
```

By default the adapter exports both official VGGT point constructions,
`world_points` and `depth_unprojected`. They receive the same model-set,
input-video, provider-run, and deterministic-member dependence groups. They are
alternative constructions from one model execution, not independent sensor
likelihoods and not two votes for a downstream update.

VGGT processes the complete supplied sequence jointly. Every exported output
frame therefore records the full sequence as its source interval. For a clip
starting at absolute frame 109, pass `--frame-start 109`; a causal cutoff admits
the payload only after the complete supplied clip ends. A predictive experiment
that needs an earlier cutoff must run VGGT on that prefix rather than relabel a
full-sequence result as causal.

The adapter declares `sequence-local-sim3`, no scene flow, and no stored camera
rays. It verifies that the two official constructions share the same frame grid,
camera extrinsics, and intrinsics before placing them in one manifest. Non-finite
points are retained as an explicit invalid mask and replaced by zero only in the
inactive canonical payload entries.

## Alternative providers

Another 4-D model should emit the same neutral contract from independently
validated source bytes. It must declare one of the supported coordinate
semantics:

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

## Python API

```python
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    load_prediction_provider_manifest,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)
from prob4d.vggt_provider_adapter import import_vggt_prediction_manifest
```

`verify_prediction_provider_manifest` checks the content identity and, by
default, reopens every exact payload byte with `allow_pickle=False`, validates
its closed versioned `PredictionWindow` schema, and checks window ID, frame grid,
storage precision, and optional field declarations.

## Repository boundary and nonclaims

The neutral manifest standardizes prediction provenance and causal admission. It
does not establish that an alternative provider is accurate, calibrated, less
biased, statistically independent, or useful to BayesianPhysTwin. Provider
competence and the guarded physical-query result remain separate held-out gates.
Causal4D should continue to consume only the accepted or exact-fallback
BayesianPhysTwin belief, not raw prediction-provider output.
