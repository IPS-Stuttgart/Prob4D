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
count, window count, and retained fraction. This is storage accounting rather
than a process-RSS measurement; allocator, decompression, and NumPy temporaries
still depend on the host runtime.

## Numerical boundary

Dense point and flow payloads may remain `float32`, while low-dimensional
alignment, gauge, and fusion routines continue to promote their selected inputs
to `float64`. The mode changes retained storage rather than the declared gauge
or covariance model. Fused artifacts nevertheless record the mode so held-out
evaluation cannot silently combine benchmark runs produced under different
execution contracts.

## Frame-local ray access

Use `PredictionWindow.rays_at(local_index)` when only one frame is needed.
`rays()` remains available for compatibility, but builds its output frame by
frame rather than creating another full-size normalization temporary.
Cross-fitted overlap disagreement retains only rays for valid overlap rows and
does not materialize complete ray fields for both windows.

## Remaining measurement work

These changes halve retained point/flow storage in the benchmark and bound ray
temporaries in cross-fitted disagreement. Follow-up work under issue #50 should
preserve structured covariance until selected slices are required, add optional
memory-mapped loading, and record peak RSS, loading time, disagreement time,
gauge estimation time, fusion time, and export time for the production
`25 x 320 x 640` setting before making a runtime or process-memory claim.
