# Memory-mapped prediction execution stores

MotionCrafter prediction members are portable compressed NPZ artifacts. NumPy
cannot memory-map arrays inside a compressed NPZ archive, so repeated large
experiments otherwise decompress and retain complete windows in process memory.
Prob4D now provides a separate, explicit execution-cache format based on
uncompressed NPY members.

The execution store does **not** replace or reinterpret the portable prediction,
provider-v1, provider-v2, calibration, or observation artifacts. It binds the
exact source prediction-manifest SHA-256 and records that provider semantics are
unchanged.

## Materialize once

```bash
prob4d storage materialize \
  outputs/sequence/predictions.json \
  outputs/sequence/prediction-store \
  --dense-storage-dtype float32
```

Materialization first verifies the MotionCrafter bundle and every registered
member. It then processes overlap windows and baselines one at a time, writes
NPY members into a temporary tree, hashes every file, writes content-addressed
window and bundle manifests, fsyncs them, and atomically publishes the complete
directory. An existing destination is rejected instead of being partially
updated.

## Load and validate

```python
from prob4d.prediction_store import load_prediction_bundle_store

bundle = load_prediction_bundle_store(
    "outputs/sequence/prediction-store",
    verify_hashes=True,
)
```

The loader rejects:

- manifest or member symlinks and path escapes;
- missing, extra, aliased, or malformed fields;
- byte-count, SHA-256, dtype, shape, or content-address mismatches;
- inconsistent scene-flow/deformation-mask pairs;
- invalid frame identities, masks, vectors, or non-normalized stored rays; and
- any store that does not declare execution-cache-only semantics.

Dense arrays are opened read-only with `mmap_mode="r"`. The returned
`MMapPredictionWindow` is a `PredictionWindow` subclass, so existing alignment,
uncertainty, and fusion routines consume it without changing estimator
semantics. Validation is chunked and no second retained dense copy is created.

Validate and summarize a store independently with:

```bash
prob4d storage validate outputs/sequence/prediction-store
```

## Matched process benchmark

Run each backend in a separate fresh process:

```bash
prob4d storage benchmark outputs/sequence/predictions.json \
  --backend eager_npz \
  --dense-storage-dtype float32 \
  --output-json outputs/sequence/eager-loading.json

prob4d storage benchmark outputs/sequence/prediction-store \
  --backend mmap_npy \
  --dense-storage-dtype float32 \
  --output-json outputs/sequence/mmap-loading.json
```

The reports bind the source-manifest identity, store ID where applicable,
backend, dtype, Python/NumPy/platform versions, loading-and-validation time,
retained dense-vector accounting, and peak process RSS. Compare only matched
fresh processes on the same host and repository revision.

The benchmark is engineering evidence. It does not establish reconstruction
accuracy, calibration, BayesianPhysTwin benefit, or Causal4D performance.

## Integrity and lifecycle

The original verified NPZ bundle remains the portable source of truth. The NPY
store may be deleted and regenerated from that exact bundle. It should be placed
under an ignored output or cache directory rather than committed. New provider
or paper evidence must continue to bind the original source manifest and the
normal Prob4D provider artifacts; the execution-store ID is additional runtime
provenance only.
