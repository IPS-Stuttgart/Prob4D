# Prediction-window storage precision

`PredictionWindow` archives use the versioned
`prob4d.prediction-window-npz` schema. New archives record three scalar fields:

- `schema_name`;
- `schema_version`; and
- `dense_storage_dtype`, either `float32` or `float64`.

The recorded dtype must agree exactly with every stored dense vector field:
`point_map`, `scene_flow`, and `ray_directions`. Loading fails closed when the
metadata is partial, the schema is unknown, or an array dtype disagrees with the
declaration.

## Writing

By default, `PredictionWindow.to_npz()` preserves the validated in-memory
`dense_storage_dtype`:

```python
window.to_npz("window.npz")
```

A compact archive therefore requires an explicit choice:

```python
window.to_npz("window-f32.npz", storage_dtype="float32")
```

This prevents a float64 prediction from being silently rounded merely because it
was serialized. MotionCrafter's native float32 output path selects float32
explicitly when constructing its `PredictionWindow`, so the change does not turn
normal GPU prediction bundles into float64 archives.

## Reading and legacy files

For a versioned archive, `PredictionWindow.from_npz()` adopts the stored dtype by
default. A caller may explicitly promote or compact the in-memory representation
through `dense_storage_dtype=`; the source precision remains available in the
archive metadata.

Historical archives without the three schema fields remain readable. They keep
the old loading default of float64 unless the caller supplies another
`dense_storage_dtype`. Files without absolute `frame_indices` still require an
explicit `start_frame`.

The versioned metadata is an execution/storage contract. It does not change the
scientific meaning of an already frozen provider artifact or rewrite historical
archive identities.
