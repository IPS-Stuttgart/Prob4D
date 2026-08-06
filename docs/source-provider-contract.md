# Provider-neutral windowed 4-D source contract

Prob4D's estimator consumes independently decoded, overlapping 4-D prediction
windows. MotionCrafter is the first producer, but the estimator-side assumptions
are not specific to one neural architecture. `prob4d.source` therefore provides
an additive, content-addressed normalization boundary for MotionCrafter and
future 4-D providers.

The source contract is deliberately **upstream** of `prob4d.provider_v1` and
`prob4d.provider_v2`:

```text
provider-specific prediction manifest
    -> prob4d.source.Windowed4DSourceManifestV1
    -> Prob4D source selection, calibration, and fusion
    -> provider-v2 observation belief or factor bundle
```

It does not modify existing `predictions.json` files, provider-v1/v2 observation
artifacts, calibration identities, or frozen run manifests. Adapting a source
manifest establishes only structural and provenance compatibility. It does not
establish accuracy, uncertainty calibration, physical-query benefit, or Causal4D
intervention benefit.

## Stable import surface

```python
from prob4d.source import (
    adapt_motioncrafter_prediction_manifest,
    load_motioncrafter_source_manifest,
    load_windowed_4d_source_manifest,
    save_windowed_4d_source_manifest,
)
```

`load_motioncrafter_source_manifest` opens only the JSON manifest. Prediction
payloads remain unopened until a later causal selector admits them.

```python
source = load_motioncrafter_source_manifest(
    "outputs/sequence_name/predictions.json"
)

print(source.source_provider_id)
print(source.model_set_id)
print(source.artifact_id)
print(source.claim_ready_source_identity)
```

The normalized artifact can be persisted without replacing existing evidence:

```python
save_windowed_4d_source_manifest(
    source,
    "outputs/sequence_name/windowed_4d_source.json",
)
reloaded = load_windowed_4d_source_manifest(
    "outputs/sequence_name/windowed_4d_source.json"
)
assert reloaded.artifact_id == source.artifact_id
```

Saving identical bytes is idempotent. Attempting to reuse the path for different
content raises `FileExistsError`.

## Schema version 1

`Windowed4DSourceManifestV1` records:

- provider ID and exact provider revision;
- optional immutable model-set identity;
- canonical source-manifest SHA-256 and, when loaded from a file, exact file-byte
  SHA-256;
- provider-native coordinate frame and length-unit semantics;
- nominal window size, overlap, and source-frame stride;
- point, flow, ray, and uncertainty representation semantics;
- source-only stochastic policy and schedule identity;
- temporal lineage sufficient to audit output-to-input dependencies;
- one record per independently decoded window, with absolute source-frame bounds;
- optional payload SHA-256 and byte count; and
- finite, recursively immutable provider metadata.

The artifact ID covers the complete normalized descriptor. Window IDs and
payload paths must be unique. Paths must be safe POSIX-relative members; absolute
paths, parent traversal, backslashes, duplicate identities, invalid frame bounds,
partial payload identities, non-finite metadata, and coercion-dependent scalar
aliases fail closed.

## MotionCrafter adapter

The version-1 adapter consumes the historical MotionCrafter prediction-manifest
schema without rewriting it. It validates the registered stochastic seed
schedule, carries the exact temporal-lineage declaration, normalizes model-set
identity when present, and maps integrity-bound overlap-window descriptors into
provider-neutral payload identities.

The adapter intentionally excludes MotionCrafter's disjoint and latent-linear
baselines from the normalized window list. Their portable paths remain in
provider metadata because they are comparison products, not independently
admitted source factors.

For legacy manifests without model-set or artifact-integrity records, adaptation
remains possible for reproduction and exploratory diagnostics. Such a normalized
artifact reports `claim_ready_source_identity == False`.

## Identity readiness is not scientific admission

`claim_ready_source_identity` is true only when all of the following are present:

1. an exact lowercase 40- or 64-character provider revision;
2. a model-set identity;
3. the exact original manifest-file SHA-256; and
4. SHA-256 plus byte count for every normalized source window.

This property is a provenance-completeness diagnostic. Claim-bearing Prob4D use
still requires the provider-v2 calibration compatibility checks, causal prefix
selection, exact runtime attestation, downstream BayesianPhysTwin guard, and
exact fallback behavior.

## Adding another provider

A new adapter should remain provider-specific while producing the same neutral
classes. It must:

1. bind an exact provider and model revision;
2. expose absolute source-frame bounds before opening payloads;
3. state point, flow, ray, uncertainty, coordinate-frame, and unit semantics;
4. preserve stochastic dependence rather than labelling different seeds as
   independent by assumption;
5. retain payload byte identities where available;
6. reject unsafe paths and ambiguous or duplicate identities; and
7. leave provider-v1/v2 observation schemas unchanged.

A provider with incompatible window, coordinate, or lineage semantics should use
a new source-contract version rather than overloading version 1.
