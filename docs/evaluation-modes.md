# Explicit evaluation modes

`prob4d.evaluate_sequence_modes` reports three distinct interpretations instead
of using one truth-derived alignment for every result.

## Metric

The prediction is evaluated in its declared coordinate system. No test truth is
used to alter its scale or translation. This is the primary interpretation for
an externally anchored Prob4D artifact and for Bayesian-PhysTwin integration.

## Causal prefix aligned

One isotropic scale and translation are fitted only from common valid points with

```text
frame_id < prefix_frame_stop_exclusive.
```

The frozen transform is then applied to all evaluated frames. Appending future
frames cannot change the fitted transform. This mode is appropriate when the
experimental protocol explicitly permits a preboundary registration/calibration
prefix.

## Oracle aligned

One scale and translation are fitted from all evaluated common frames. This
matches the usual reconstruction-style alignment but uses outcome information
from the evaluation interval. It is therefore a diagnostic control, not a causal
prediction result.

## Example

```python
from prob4d import evaluate_sequence_modes

results = evaluate_sequence_modes(
    prediction,
    truth,
    boundary_frames=window_boundary_frames,
    prefix_frame_stop_exclusive=134,
)

metric_rmse = results.metric.metrics.metric_point_rmse
prefix_rmse = results.prefix_aligned.metrics.metric_point_rmse
oracle_rmse = results.oracle_aligned.metrics.metric_point_rmse
```

The result records the fitted scale, translation, frame count, and point count
for each truth-derived alignment. `to_dict()` emits a JSON-ready nested payload.

Flow support is sanitized before evaluation: deformation entries are retained
only where geometry is valid and the corresponding flow vector is finite. This
prevents invalid or masked pixels from contributing to endpoint error.
