# Provider-neutral prediction-window bundles

`ProviderWindowBundleV1` is the portable boundary between an external 4-D
prediction model and Prob4D's gauge, uncertainty, identity, and observation
export code. It binds exact independently decoded window archives to the
provider implementation, immutable model set, source sequence, and complete
causal source intervals.

The bundle deliberately does **not** import or execute the provider. A consumer
can inspect its metadata without opening dense arrays, or verify every payload
before using it.

## Contract

Every admitted window is a `PredictionWindow` NPZ with:

- a stable `window_id`;
- absolute source-frame IDs;
- one local `Sim(3)` gauge;
- a point map and validity mask;
- optional scene flow and viewing rays; and
- one complete source-frame interval containing every source frame that could
  influence the decoded window.

A bundle records the exact SHA-256 and byte count of every NPZ, its frame-index
digest, resolution, storage precision, capabilities, and archive schema. All
windows must share resolution, storage precision, and optional-field
capabilities. Window records use canonical source-interval ordering and unique
IDs, paths, and payload hashes.

The fixed semantics in version 1 are:

```text
coordinate_semantics:      independent-window-sim3
frame_index_semantics:     absolute-source-frame
source_lineage_semantics:  complete-source-interval
```

Provider, model-set, and source identities must be exact `git:`, `sha256:`, or
`oci:sha256:` identities. Human-readable version strings do not replace those
identities.

A valid bundle establishes interoperability and provenance only. It does not
establish target calibration, independence between providers, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.

## Generic external-provider ingest

External providers should write versioned windows with
`PredictionWindow.to_npz`. Prepare a strict ingest specification such as
[`examples/provider-window-ingest-spec.json`](examples/provider-window-ingest-spec.json):

```json
{
  "schema_name": "prob4d.provider-window-ingest-spec",
  "schema_version": 1,
  "provider_name": "Example4D",
  "provider_version": "1.2.3",
  "implementation_identity": "git:0123456789abcdef0123456789abcdef01234567",
  "model_set_identity": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_identity": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "coordinate_semantics": "independent-window-sim3",
  "frame_index_semantics": "absolute-source-frame",
  "source_lineage_semantics": "complete-source-interval",
  "allow_legacy_window_archives": false,
  "windows": [
    {
      "window_id": "window_0000",
      "path": "windows/window_0000.npz",
      "source_frame_start": 0,
      "source_frame_stop_exclusive": 25
    }
  ],
  "metadata": {
    "adapter": "example4d-prob4d-v1"
  }
}
```

Paths are canonical POSIX paths relative to the payload root. Absolute paths,
parent traversal, noncanonical paths, and symlink members are rejected.
Duplicate JSON keys, non-finite values, coercion-dependent integers, unknown
fields, and malformed identities also fail closed.

Build and immediately verify the bundle:

```bash
prob4d provider ingest \
  provider-spec.json \
  provider-bundle.json \
  --payload-root /path/to/provider-output
```

When `--payload-root` is omitted, paths are resolved relative to the ingest
specification. The output location does not alter the content identity.

Versioned prediction-window archives are the default. A legacy unversioned NPZ
is admitted only when the specification explicitly sets
`allow_legacy_window_archives` to `true`; the resulting window record remains
labelled `legacy-unversioned`.

## MotionCrafter compatibility adapter

An integrity-bound `prob4d motioncrafter` output can be adapted directly:

```bash
prob4d provider ingest-motioncrafter \
  outputs/sequence/predictions.json \
  outputs/sequence/provider-bundle.json
```

The adapter first runs the existing MotionCrafter integrity verifier. It then
binds:

- the exact MotionCrafter Git object;
- the pinned model-set SHA-256;
- the input-video SHA-256;
- the complete overlap-window payloads and source intervals;
- the MotionCrafter run-spec identity;
- the original prediction-manifest identity; and
- the recorded stochastic seed policy.

Legacy MotionCrafter window archives remain explicitly labelled; the adapter
does not silently rewrite their bytes or claim a newer archive schema.

## Validation and metadata-only inspection

Verify a bundle and all payload bytes:

```bash
prob4d provider validate provider-bundle.json \
  --payload-root /path/to/provider-output
```

If the bundle lives next to its payload paths, `--payload-root` can be omitted.
For orchestration that must select sources before opening dense arrays, validate
only the closed manifest and content identity:

```bash
prob4d provider validate provider-bundle.json --metadata-only
```

Python callers use the same separation:

```python
from prob4d.provider_bundle import (
    load_provider_window_bundle,
    verify_provider_window_bundle,
)

bundle = load_provider_window_bundle("provider-bundle.json")
report = verify_provider_window_bundle(
    bundle,
    payload_root="/path/to/provider-output",
)
```

Loading a bundle never opens prediction payloads. Full verification reopens each
NPZ, checks its exact hash and byte count, revalidates the `PredictionWindow`, and
compares every recorded shape, frame, precision, capability, archive, and source
interval field.

## Persistence and race behavior

Bundle manifests are canonical, content-addressed JSON. Writing is atomic,
serialized by a fail-closed lock, and idempotent for identical bytes. An existing
different bundle is never overwritten. Payloads are hashed before and after
window validation, and admission fails if a file changes during inspection.

## Downstream use

This contract is intentionally upstream of the existing Prob4D provider-v2
observation contract:

```text
external 4-D provider
    -> ProviderWindowBundleV1
    -> Prob4D gauge / uncertainty / identity estimation
    -> ObservationBeliefV1 or ObservationFactorBundle
    -> BayesianPhysTwin guarded update and exact fallback
    -> Causal4D intervention analysis
```

Provider-specific inference and model loading stay outside the NumPy-only Prob4D
core. BayesianPhysTwin remains responsible for physical-state updates and guards;
Causal4D remains responsible for interventions on an accepted belief.
