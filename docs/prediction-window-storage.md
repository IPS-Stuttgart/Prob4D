# Prediction-window storage precision

`PredictionWindow` archives use the versioned
`prob4d.prediction-window-npz` schema. New archives record three scalar fields:

- `schema_name`;
- `schema_version`; and
- `dense_storage_dtype`, either `float32` or `float64`.

Versioned archives use a closed member set. They require `window_id`, absolute
`int64` `frame_indices`, `point_map`, and Boolean `valid_mask` in addition to the
schema fields. Optional members are `ray_directions` and the paired
`scene_flow`/Boolean `deform_mask` fields. Unknown members, missing required
members, a lone flow or deform mask, and coercion-dependent integer or mask
dtypes fail closed.

The recorded dense dtype must agree exactly with every stored vector field:
`point_map`, `scene_flow`, and `ray_directions`. Loading also fails closed when
the metadata is partial or the schema is unknown.

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
`dense_storage_dtype`, and retain the historical coercion of frame indices and
masks into the validated in-memory representation. Files without absolute
`frame_indices` still require an explicit `start_frame`.

The versioned metadata is an execution/storage contract. It does not change the
scientific meaning of an already frozen provider artifact or rewrite historical
archive identities.
