# Group-balanced point-uncertainty calibration

Dense calibration sets often contain very different numbers of valid rows per
sequence, object, camera, or acquisition session. Pooling all rows gives every
pixel equal mass, which can let one long or densely valid group determine nearly
the entire variance scale.

`DepthDisagreementModel.calibrate_group_balanced` provides an explicit
alternative:

```python
calibrated, report = model.calibrate_group_balanced(
    errors,
    covariance,
    sequence_ids,
    mask=valid,
    trim_quantile=0.99,
)
```

For every declared group, Prob4D independently computes the along-ray and
per-lateral-axis normalized squared errors, trims those ratios at the requested
within-group quantile, and obtains one parallel and one lateral scale update.
The final update is the arithmetic mean of the per-group updates. Thus each
sequence contributes one calibration unit regardless of how many sampled rows
it contains.

The returned `GroupBalancedCalibrationReport` records:

- the total valid row count and number of groups;
- the common trim quantile;
- equal-group aggregate scale updates and normalized MSE;
- canonical sorted group IDs;
- every group's row count, scale update, and untrimmed normalized MSE.

`report.to_dict()` emits the explicit semantics identifier
`equal-group-mean-of-within-group-trimmed-ratios-v1`.

## Choosing the statistical unit

The grouping must match the intended transfer claim. Typical choices are:

- scene sequence for sequence-family-held-out reconstruction;
- physical object or acquisition session for deformable-object transfer;
- camera only when cameras are the independent deployment units;
- simulation run for controlled Monte Carlo evidence.

Neighboring pixels, overlapping windows, or repeated frames from one sequence
are not independent calibration units merely because they are separate rows.

## Compatibility and claim boundary

`DepthDisagreementModel.calibrate` remains unchanged and retains pooled-point
weighting for frozen reproduction. Group-balanced calibration is additive and
must be selected explicitly.

Changing the calibration aggregation changes the fitted covariance model.
Claim-bearing artifacts must therefore record the grouping definition and exact
calibration method, use disjoint calibration and target groups, and regenerate
content-addressed calibration artifacts rather than relabeling pooled results.
