# Dense-memory execution

Prob4D separates **dense storage precision** from **estimation precision**. The
serialized MotionCrafter arrays are already `float32`; eagerly converting every
point map, scene-flow field, and optional ray field to `float64` can double the
retained prediction memory without adding information.

## Explicit storage mode

`PredictionWindow` and `load_prediction_bundle` accept
`dense_storage_dtype="float32"` or `"float64"`. The default on the reusable
provider-facing loader remains `float64` for frozen compatibility. The pinned
benchmark deliberately selects `float32` and binds that choice into its config
and every fused prediction artifact:

```bash
prob4d benchmark \
  --dataset-dir /data/Sintel_video \
  --output-dir outputs/benchmark \
  --upstream-root /opt/MotionCrafter \
  --cache-dir /models/cache \
  --unet-path TencentARC/MotionCrafter \
  --unet-revision <exact-revision> \
  --vae-path TencentARC/MotionCrafter \
  --vae-revision <exact-revision> \
  --base-pipeline-revision <exact-revision> \
  --dense-storage-dtype float32
```

The benchmark report stores a deterministic `prediction_dense_storage` summary
for each completed sample: retained bytes, the all-`float64` equivalent, field
count, window count, and retained fraction. This storage accounting is separate
from process RSS because allocator, decompression, and NumPy temporaries depend
on the host runtime.

## Numerical boundary

Dense point and flow payloads may remain `float32`, while low-dimensional
alignment, gauge, and fusion routines continue to promote their selected inputs
to `float64`. The mode changes retained storage rather than the declared gauge
or covariance model. Fused artifacts nevertheless record the mode so held-out
evaluation cannot silently combine benchmark runs produced under different
execution contracts.

## Frame-local rays and structured covariance

Use `PredictionWindow.rays_at(local_index)` when only one frame is needed.
`rays()` remains available for compatibility, but builds its output frame by
frame rather than creating another full-size normalization temporary.
Cross-fitted overlap disagreement retains only rays for valid overlap rows and
does not materialize complete ray fields for both windows.

Dense fusion preserves structured ray-parallel/lateral covariance until a
representative covariance-intersection sample or active spatial tile is needed.
CI weights are still optimized once per complete frame and contributor-mask
pattern, so `fusion_tile_size` changes temporary memory rather than estimator
semantics. See [tiled dense fusion](tiled-fusion.md).

## Memory-mapped prediction stores

Portable prediction members remain compressed NPZ artifacts. For repeated large
experiments, `prob4d storage materialize` creates a separate content-addressed
execution cache of ordinary NPY members:

```bash
prob4d storage materialize \
  outputs/sequence/predictions.json \
  outputs/sequence/prediction-store \
  --dense-storage-dtype float32

prob4d storage validate outputs/sequence/prediction-store
```

The store verifies the source bundle and every member, publishes atomically,
and exposes read-only `MMapPredictionWindow` arrays. It does not replace the
portable provider artifact and may be deleted and regenerated. See
[memory-mapped prediction execution stores](prediction-store.md).

Run matched loading measurements in separate fresh processes:

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

## Bounded export and evaluation

The bounded fused-prediction writer stages point and flow arrays in chunks and
packs symmetric covariance directly into six values per sample. It preserves the
existing fused-prediction field semantics while trading temporary disk space for
lower export allocations. See [bounded fused export](bounded-fused-export.md).

Engineering parity and ablation checks may use
`PredictionWindowTruthView` to reuse one already validated immutable prediction
window as an explicitly declared internal reference. External truth continues to
use the defensive `TruthSequence` contract. See
[zero-copy prediction references](prediction-window-truth-view.md).

Provider metrics and evaluation modes process active rows in bounded chunks. The
configured chunk size changes temporary memory and timing, not the declared
metric or estimator semantics. See [streaming evaluation](streaming-evaluation.md).

## Completed full-resolution measurement

Issue #50's frozen real-bundle comparison completed on exact evidence head
`1e02c868db5e94fb59d60ebab63c0e439a814c81`. It used the non-protected
Deform360 calibration sequence `002-rope-silk-ep0008`, source frames `[0,59)` at
`320 x 640`, three 25-frame windows, and two uniform eight-frame overlaps. Eager
NPZ and mmap NPY arms ran in matched fresh processes.

| Quantity | Eager NPZ | mmap NPY | Change |
|---|---:|---:|---:|
| Source-loading peak RSS | 1,766.43 MiB | 1,026.20 MiB | −740.24 MiB (−41.91%) |
| Maximum process RSS | 5,493.54 MiB | 5,352.92 MiB | −140.62 MiB (−2.56%) |
| Source-loading time | 6.415 s | 1.257 s | −5.158 s |
| Total arm wall time | 198.257 s | 188.344 s | −9.913 s (−5.00%) |
| Persistent input bytes | 855,636,705 | 1,027,712,841 | +172,076,136 bytes |

The total-process peak passed the preregistered 128 MiB absolute-reduction gate
but not the 10% relative gate. The source-loading peak passed both its 64 MiB
absolute and 10% relative gates. The locked decision was therefore:

```text
material_memory_benefit
```

Every declared semantic comparison passed exactly: source arrays, disagreement
and uncertainty fields were byte-identical; point, flow, covariance, gauge,
seam, calibration-state, and provider-evaluation maximum absolute differences
were zero; and eager/mmap provider exports had equal semantic content and equal
size. The compact evidence and exact identities are retained on
[PR #144](https://github.com/IPS-Stuttgart/Prob4D/pull/144).

This is measured backend behavior for one frozen calibration workload. It does
not establish the same RSS or runtime change on another host, provider, sequence,
resolution, allocator, compression setting, or model stack.

## Gauge-prior storage

The production causal gauge tree also has an `O(K)` square-root representation.
It removes the dense prior from runtime algebra after direct sparse construction
or strict conversion. Existing schema-v4 observation-factor bundles still carry
the dense covariance and retain their exact identities. See
[the sparse gauge-tree prior](sparse-gauge-tree-prior.md).

## Claim boundary

Storage accounting, exact parity, and measured RSS/runtime behavior are
engineering evidence. They do not establish reconstruction accuracy, calibrated
uncertainty, provider competence, BayesianPhysTwin benefit, Causal4D benefit,
deployment safety, or state of the art.
