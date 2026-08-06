# Bounded-memory fused-prediction export

`prob4d.io.save_fused_prediction` writes the canonical fused-prediction NPZ
schema directly through NumPy. That path is simple and remains the default. For
a large `FusedSequence`, however, it can transiently materialize complete
float32 point/flow copies and complete six-component packed covariance arrays
before the archive is written.

`prob4d.bounded_export.save_fused_prediction_bounded` is an additive writer for
large engineering and provider-export workloads:

```python
from prob4d.bounded_export import save_fused_prediction_bounded

save_fused_prediction_bounded(
    "fused.npz",
    fused,
    method_id="prob4d_ci_smoothed_uncalibrated",
    fusion_method="covariance_intersection",
    include_covariance=True,
    metadata={"calibration": "uncalibrated-fixed-model"},
    chunk_rows=262_144,
)
```

The writer preserves the existing archive contract:

- identical field names;
- identical array dtypes and shapes;
- identical numeric values after the established float32 export conversion;
- the same `FusedPredictionMetadata` validation and covariance semantics; and
- compatibility with `load_fused_prediction`,
  `load_fused_prediction_metadata`, and
  `load_fused_prediction_artifact`.

Large fields are first written as NPY members in the destination directory.
Point and flow conversion is chunked. Symmetric 3×3 covariance matrices are
packed directly into six float32 values per sample without creating a complete
float64 packed array. The NPY members are then streamed into a temporary ZIP
archive, fsynced, and atomically installed at the destination.

The NPZ container bytes are not required to equal those produced by
`numpy.savez` because ZIP timestamps and compression implementation details are
container metadata. Tests require exact equality of every decoded field and
complete metadata equality.

## Resource trade-off

The peak array temporary is bounded by `chunk_rows`, but temporary disk usage can
approach:

```text
staged NPY members + final NPZ archive
```

For uncompressed export this can be roughly twice the final artifact size until
the atomic replacement completes. The staging directory is removed on success
or failure, and an existing destination remains unchanged if archive assembly
fails.

This is useful when RAM, rather than disk capacity, is the limiting resource.
The ordinary writer remains appropriate for small artifacts or environments
where temporary disk usage is more constrained.

## Claim boundary

The bounded writer changes the export implementation, not estimator or
covariance semantics. It does not establish reconstruction accuracy,
uncertainty calibration, provider competence, BayesianPhysTwin benefit,
Causal4D benefit, deployment safety, or state of the art.
