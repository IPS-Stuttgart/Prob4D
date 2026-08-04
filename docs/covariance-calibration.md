# Covariance calibration contracts

Prob4D exposes two content-addressed calibration artifacts:

- `GaugeCovarianceCalibrationV1` inflates the scale, rotation, and translation
  blocks of dense-overlap Sim(3) covariance.
- `PointUncertaintyCalibrationV1` carries the calibrated
  `DepthDisagreementModel` used for conditional point covariance.

Both artifacts bind their fitted parameters to calibration case identifiers,
Prob4D and MotionCrafter revisions, model and covariance settings, source
artifact digests, and finite JSON metadata. The loaders recompute and verify the
artifact identifier, so edited calibration files fail validation.

## Robust scale aggregation semantics

Point and gauge scale fitting cap normalized residual ratios at an upper empirical
quantile and then average all rows. This is **upper winsorization**: tail rows are
not removed. The explicit public semantics identifier is
`upper-quantile-winsorized-mean-v1`.

The frozen version-1 point and gauge artifact schemas retain the serialized field
name `trim_quantile` for compatibility. Code should read the `winsor_quantile`
alias when describing the statistical operation. Existing artifact descriptors and
IDs remain valid and unchanged.

New group-balanced point calibration reports use
`equal-group-mean-of-within-group-upper-winsorized-ratios-v2`. The loader also
accepts the historical identifier
`equal-group-mean-of-within-group-trimmed-ratios-v1`, because those legacy bytes
already implemented winsorization despite the name. New artifacts are never
silently relabelled as legacy artifacts.

## Claim-bearing export

New claim-bearing experiments should use `prob4d.provider_v2`:

```python
from prob4d.provider_v2 import (
    export_calibrated_observation_belief,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_point_uncertainty_calibration,
)

anchor = load_metric_gauge_anchor("metric-anchor.json")
gauge_calibration = load_gauge_covariance_calibration("gauge-calibration.json")
point_calibration = load_point_uncertainty_calibration("point-calibration.json")

artifact = export_calibrated_observation_belief(
    "predictions.json",
    case_id="held-out-case",
    causal_frame_stop=134,
    metric_anchor=anchor,
    gauge_covariance_calibration=gauge_calibration,
    point_uncertainty_calibration=point_calibration,
    source_revision="<exact Prob4D commit>",
)
```

Version 2 reads only prediction-manifest metadata first and fails closed unless
both artifacts match the MotionCrafter revision, canonical model identifier,
resolution, window geometry, covariance cluster size, and their respective
covariance methods. The decoded prediction arrays remain unopened until these
checks pass. See [Provider API version 2](provider-v2.md) for the exact contract.

`prob4d.provider_v1.export_calibrated_observation_belief` remains frozen for
existing run manifests. Its more general `export_observation_belief` remains
usable for explicitly exploratory runs and records `uncalibrated_exploratory` or
`partially_calibrated` in observation metadata.

## Dense-alignment fallback policy

The provider fails closed when a requested frame-by-spatial-tile covariance has
fewer than eight active clusters. A pointwise-cluster fallback can be enabled
only for an exploratory version-1 or version-2 export. The observation metadata
records both whether the fallback was allowed and how many alignments actually
used each fallback mode.

Direct low-level calls to `align_windows` retain the historical pointwise
fallback for source compatibility. Provider exports use a task-local covariance
policy, so concurrent exports cannot leak calibration or fallback settings into
one another.

## Interpretation

A structurally compatible calibration artifact is not, by itself, evidence of
prospective coverage or downstream physical-twin improvement. Those claims remain
gated on held-out calibration and end-to-end Prob4D-to-Bayesian-PhysTwin
evaluation.
