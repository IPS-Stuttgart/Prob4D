# Provider API version 2

`prob4d.provider_v2` is the safe-by-default Python surface for new claim-bearing
experiments. It does not change `ObservationBeliefV1`, the causal observation
stream contract, or the frozen `prob4d.provider_v1` behavior.

## Explicit export modes

Version 2 removes the ambiguous general export entry point. Callers must choose
one of two functions:

- `export_exploratory_observation_belief` permits uncalibrated or partially
  calibrated covariance and can explicitly enable the pointwise covariance
  fallback.
- `export_calibrated_observation_belief` requires both content-addressed
  calibration artifacts, an exact Prob4D source revision, sequential gauge
  covariance propagation, and the fail-closed spatial-cluster covariance path.

The calibrated function validates compatibility before any decoded prediction
payload is opened.

## Canonical covariance-root basis

Provider version 1 retains the frozen eigenvector/sign convention used by existing
artifacts. Version 2 selects a context-local canonical basis for numerically
repeated covariance eigenspaces. The basis is derived from each eigenspace
projector rather than from an arbitrary orthonormal basis returned by the linear
algebra backend.

Version 2 also fails closed when an eigenvalue floor or `max_gauge_rank` boundary
would split a numerically repeated eigenspace. Such a split would make the retained
subspace depend on an arbitrary eigensolver basis. Exploratory callers can request
`gauge_root_mode="legacy_eigenvectors"` when reproducing a version-1 factor basis;
the claim-bearing entry point always uses `canonical_eigenspaces`.

The mode is context-local, so concurrent version-1 and version-2 exports retain
their declared semantics without process-global mode leakage. The low-rank factor
bytes remain covered by the observation artifact ID.

## Canonical MotionCrafter model identifier

Calibration artifacts used through version 2 must set `model_identifier` to the
value returned by:

```python
import json

from prob4d.provider_v2 import motioncrafter_model_identifier

manifest = json.loads(open("predictions.json", encoding="utf-8").read())
model_identifier = motioncrafter_model_identifier(manifest)
```

The identifier hashes the model type, UNet and VAE identifiers, inference-step
configuration, guidance scale, decode chunk size, low-memory mode, random seed,
and temporal frame stride. Image resolution and window geometry remain separate
compatibility fields so mismatch diagnostics identify them directly. The exact
MotionCrafter source commit is checked separately.

## Compatibility fields

For a claim-bearing export, each calibration must match the prediction manifest
and runtime settings in all of the following fields:

- source repository;
- MotionCrafter revision;
- canonical model identifier, including seed and temporal stride;
- image resolution;
- window size and overlap;
- covariance cluster size; and
- the expected gauge or point covariance method.

The default methods are
`frame_spatial_cluster_robust_v1` for gauge covariance and
`depth_disagreement_anisotropic_v1` for conditional point covariance. A mismatch
raises `CalibrationCompatibilityError` with field-level diagnostics.

The calibration case identifiers and input digests are deliberately not compared
to the target sequence. They identify the independent calibration data and should
normally differ from the target artifact.

## Example

```python
from prob4d.provider_v2 import (
    export_calibrated_observation_belief,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_point_uncertainty_calibration,
)

artifact = export_calibrated_observation_belief(
    "predictions.json",
    case_id="held-out-case",
    causal_frame_stop=134,
    metric_anchor=load_metric_gauge_anchor("metric-anchor.json"),
    gauge_covariance_calibration=load_gauge_covariance_calibration(
        "gauge-calibration.json"
    ),
    point_uncertainty_calibration=load_point_uncertainty_calibration(
        "point-calibration.json"
    ),
    source_revision="<exact Prob4D commit>",
)
```

Version 1 remains available for frozen runs. New experiments should use version 2
unless they intentionally reproduce the earlier API semantics.
