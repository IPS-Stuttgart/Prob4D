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

## Frame-local ray and structured-covariance access

Use `PredictionWindow.rays_at(local_index)` when only one frame is needed.
`rays()` remains available for compatibility, but builds its output frame by
frame rather than creating another full-size normalization temporary.
Cross-fitted overlap disagreement retains only rays for valid overlap rows and
does not materialize complete ray fields for both windows.

Dense fusion preserves structured ray-parallel/lateral covariance until a
representative covariance-intersection sample or active spatial tile is needed.
The CI weights are still optimized once per complete frame and contributor-mask
pattern, so `fusion_tile_size` changes temporary memory rather than estimator
semantics.

## Memory-mapped prediction stores

`prob4d storage materialize` converts a verified portable prediction bundle into
a content-addressed directory of uncompressed NPY members. The store validates
source identity and every file hash before exposing read-only memory maps through
`MMapPredictionWindow`. Portable provider artifacts remain unchanged; the store
is an execution cache rather than a replacement provider schema.

```bash
prob4d storage materialize predictions.json outputs/prediction-store
prob4d storage validate outputs/prediction-store
prob4d storage benchmark predictions.json --backend mmap
```

The mmap route avoids eagerly retaining complete decoded input arrays, while the
ordinary loader remains available for compatibility and small jobs.

## Bounded export and evaluation

The bounded fused-prediction writer stages point and flow arrays in chunks and
packs symmetric covariance directly into six values per sample. It preserves the
existing fused-prediction field semantics while trading temporary disk space for
lower export allocations.

Engineering parity and ablation checks may use
`PredictionWindowTruthView` to reuse one already validated immutable prediction
window as an explicitly declared internal reference. External truth continues to
use the defensive `TruthSequence` contract.

Provider metrics and evaluation modes process active rows in bounded chunks.
The configured chunk size changes temporary memory and timing, not the declared
metric or estimator semantics.

## Completed production measurement

Issue #50 completed the preregistered full-resolution comparison on the
non-protected Deform360 calibration sequence `002-rope-silk-ep0008`, using three
`25 x 320 x 640` windows and matched fresh processes. Relative to eager NPZ
loading, the verified mmap store produced:

| Quantity | Eager NPZ | mmap NPY | Change |
|---|---:|---:|---:|
| Source-loading peak RSS | 1,766.43 MiB | 1,026.20 MiB | −740.24 MiB (−41.91%) |
| Maximum process RSS | 5,493.54 MiB | 5,352.92 MiB | −140.62 MiB (−2.56%) |
| Source-loading time | 6.415 s | 1.257 s | −5.158 s |
| Total arm wall time | 198.257 s | 188.344 s | −9.913 s (−5.00%) |
| Persistent input bytes | 855,636,705 | 1,027,712,841 | +172,076,136 bytes |

Every declared point, flow, covariance, gauge, seam, calibration, provider, and
export comparison passed with zero numerical difference. The locked decision was
`material_memory_benefit`: the absolute total-process reduction exceeded the
128 MiB gate, while the relative total-process reduction did not reach 10%; the
source-loading reduction passed both its absolute and relative gates.

This is measured backend behavior for one frozen calibration workload. It does
not establish the same RSS or runtime change on another host, provider, sequence,
resolution, allocator, compression setting, or model stack.

## Gauge-prior storage

The production causal gauge tree also has an `O(K)` square-root representation.
New prospective executions can persist it through the portable sparse artifact
without materializing a dense `7K x 7K` covariance. Historical provider-v2
schema-v4 bundles remain dense and retain their exact identities. See
[the sparse gauge-tree prior](sparse-gauge-tree-prior.md).

## Claim boundary

Storage accounting, exact parity, and measured RSS/runtime behavior are
engineering evidence. They do not establish reconstruction accuracy, calibrated
uncertainty, provider competence, BayesianPhysTwin benefit, Causal4D benefit,
deployment safety, or state of the art.
