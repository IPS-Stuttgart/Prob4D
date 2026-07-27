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

The calibrated function verifies the executing Prob4D revision and validates
calibration compatibility before any decoded prediction payload is opened.

The grouped CLI exposes the same distinction:

```bash
prob4d observation export-calibrated \
  outputs/test/predictions.json \
  outputs/test/observation_belief.npz \
  --case-id held-out-case \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/test/metric-anchor.json \
  --gauge-covariance-calibration outputs/calibration/gauge.json \
  --point-uncertainty-calibration outputs/calibration/point.json \
  --source-revision "$(git rev-parse HEAD)" \
  --summary-json outputs/test/observation_belief_summary.json
```

Use `prob4d observation export-exploratory` for labelled reconstruction controls.
The older `prob4d observation export` and
`prob4d-export-observation-belief` commands remain frozen provider-v1
compatibility surfaces.

## Provider and runtime attestation

Every provider-v2 artifact contains `metadata.prob4d_provider_attestation`. The
record binds:

- provider API version 2 and the content-addressed provider-v2 manifest;
- the artifact's exact Prob4D source revision;
- calibrated versus exploratory export mode;
- whether prediction/calibration compatibility was validated;
- gauge and point calibration artifact identifiers;
- covariance-root and composition-Jacobian modes; and
- the observed runtime revision, its evidence source, checkout cleanliness, and
  whether the observation was independently verified from VCS metadata.

Claim-bearing export fails closed when runtime provenance is unavailable,
mismatched, dirty, or not independently verified. It accepts only a VCS-installed
package whose PEP 610 metadata identifies the commit or a clean source checkout at
the declared revision.

`PROB4D_RUNTIME_REVISION` may annotate a packaged exploratory deployment. It is
recorded as `deployment_environment`, but an unauthenticated environment variable
cannot prove which code bytes are executing and therefore never satisfies the
claim-bearing entry point.

CI emits both provider manifests with:

```bash
prob4d provider manifest --api-version 1 --provider-revision "<commit>"
prob4d provider manifest --api-version 2 --provider-revision "<commit>"
```

## Analytic Sim(3) composition Jacobians

Sequential gauge covariance propagation composes a parent gauge with an uncertain
relative gauge. Provider v2 now differentiates that composition analytically in
the repository's seven-coordinate convention:

```text
[log scale, axis-angle rotation (3), translation (3)].
```

For `G = G_parent compose G_relative`, the derivatives account for:

- additive log scale;
- the SO(3) right Jacobians of the parent, relative, and composed rotations;
- scale and rotation transport of the relative translation; and
- the direct parent and relative translation blocks.

The SO(3) logarithm is not differentiable at its pi branch cut. Provider-v2
sequential export fails closed at that numerically ambiguous boundary instead of
exporting a platform-dependent covariance. Random-transform, near-identity, and
right-Jacobian inverse tests compare the analytic result with the frozen
central-difference implementation.

A task-local dispatcher keeps the compatibility boundary explicit:

- provider-v2 sequential joint-gauge export uses `analytic`;
- provider v1 defaults to `legacy_finite_difference` even after provider v2 has
  been imported;
- the exploratory fixed-lag reconstruction path retains
  `legacy_finite_difference`, because its rolling smoother has separate nonlinear
  derivatives; and
- nested or concurrent export contexts cannot leak the provider-v2 sequential
  choice into a frozen provider-v1 run.

The selected mode is recorded in the provider-v2 artifact attestation.

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

## Python example

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
