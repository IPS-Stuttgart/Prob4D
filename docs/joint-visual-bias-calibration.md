# Joint shared and camera-specific visual-bias calibration

Several cameras can agree while sharing the same reconstruction bias. Treating
one calibrated visual-bias sidecar per camera as independent would then count the
shared error several times. `prob4d.joint_visual_bias_calibration` provides a
prospective source/calibration-only wrapper that constructs one latent vector
across all declared cameras and delegates fitting, rank selection, covariance
estimation, persistence, and observation-sidecar construction to the existing
`VisualBiasCalibrationV1` implementation.

The frozen single-scope implementation is unchanged. New experiments select the
joint wrapper explicitly.

## Joint latent layout

For camera set `v = 1, ..., V`, the ordered latent vector is

```text
b = [b_shared, b_camera_mode_1_camera_1, ..., b_camera_mode_1_camera_V, ...].
```

The corresponding residual model for one complete calibration object or session
is

```text
r = G delta_xi + B_shared b_shared
    + sum_v B_camera,v b_camera,v + epsilon.
```

Shared basis columns are active for every camera row. A camera-specific column is
active only for rows from its named camera. The fitted selected covariance is the
complete covariance of the retained coefficients, so it may contain:

- covariance between shared and camera-specific modes;
- covariance between different cameras; and
- covariance between several retained camera-specific modes.

The latent is represented once in the resulting `VisualBiasNuisanceV1`. It must
not be instantiated independently for each camera or each recursive update.

## Canonical basis ordering

Candidate ranks in `VisualBiasCalibrationV1` are nested prefixes. The joint
layout therefore uses a fixed order:

1. shared modes in the supplied order;
2. for each camera-specific mode, one column for every camera in sorted camera
   order.

For example:

```python
from prob4d.joint_visual_bias_calibration import JointVisualBiasLayoutV1

layout = JointVisualBiasLayoutV1(
    camera_ids=("camera-0", "camera-1", "camera-2"),
    shared_basis_names=("shared-depth", "shared-scale-drift"),
    camera_basis_names=("camera-depth-bowl",),
)

assert layout.basis_names == (
    "shared::shared-depth",
    "shared::shared-scale-drift",
    "camera::camera-depth-bowl::camera-0",
    "camera::camera-depth-bowl::camera-1",
    "camera::camera-depth-bowl::camera-2",
)
```

By default, a selected rank that cuts through one camera-specific mode fails
closed. That prevents an apparently joint method from silently selecting the
mode for only a subset of cameras. `allow_partial_camera_mode=True` is available
only for explicitly labelled exploratory asymmetric studies.

## Calibration groups

Every `JointVisualBiasCalibrationGroupV1` represents one complete source or
calibration object/session. It binds:

- the exact joint layout identity;
- one camera index for every residual row;
- source residuals;
- shared and camera-specific candidate Jacobians;
- conditional point covariance;
- optional complete gauge design;
- row counts for every camera; and
- immutable finite metadata and array digests.

Every calibration group must contain rows from every declared camera. Frames,
pixels, views, points, and tracks are not independent calibration groups.

```python
import numpy as np

from prob4d.joint_visual_bias_calibration import (
    JointVisualBiasCalibrationGroupV1,
    fit_joint_visual_bias_calibration,
)

groups = tuple(
    JointVisualBiasCalibrationGroupV1(
        group_id=object_id,
        layout=layout,
        row_camera_indices=camera_index_per_row.astype(np.int64),
        residual=residual_xyz.astype(np.float64),
        shared_bias_jacobian=shared_basis.astype(np.float64),
        camera_bias_jacobian=camera_basis.astype(np.float64),
        conditional_covariance=conditional_covariance.astype(np.float64),
        gauge_design=complete_gauge_design.astype(np.float64),
        metadata={"episode_ids": episode_ids},
    )
    for object_id, camera_index_per_row, residual_xyz, shared_basis,
        camera_basis, conditional_covariance, complete_gauge_design, episode_ids
    in calibration_inputs
)

calibration = fit_joint_visual_bias_calibration(
    groups,
    provider_manifest_id=provider_manifest_id,
    calibration_source_id=calibration_source_id,
    group_definition="complete-physical-object-v1",
    residual_definition="source-metric-minus-provider-point-v1",
    uses_truth=True,
    covariance_shrinkage=0.25,
    minimum_nll_improvement=1e-4,
)
```

The wrapper embeds the canonical layout, exact group artifact IDs, selection
policy, covariance semantics, and information boundary into the ordinary
content-addressed calibration metadata. Existing strict calibration persistence
and loading therefore remain authoritative.

## Observation sidecar

A promoted calibration can instantiate one cross-camera latent for a new causal
observation:

```python
from prob4d.joint_visual_bias_calibration import (
    build_joint_visual_bias_nuisance_from_calibration,
)

sidecar = build_joint_visual_bias_nuisance_from_calibration(
    calibration,
    observation_artifact_id=observation_artifact_id,
    observation_identity_sha256=ordered_row_identity_sha256,
    row_camera_indices=camera_index_per_row,
    shared_bias_jacobian=shared_basis_for_observation,
    camera_bias_jacobian=camera_basis_for_observation,
    conditional_covariance=conditional_covariance,
    gauge_design=complete_gauge_design,
    metadata={"case_id": case_id},
)
```

The sidecar uses one bias ID, `joint-visual-cameras`, and the complete selected
coefficient covariance. Recursive BayesianPhysTwin use must retain that latent
once across time. Recreating an independent copy at each update would remove the
cross-time covariance and overcount visual evidence.

## Independent anchors

Tactile, depth, LiDAR, force, robot-state, or other independent evidence must
remain separate factors with no visual-bias Jacobian. Their scientific value is
that they can break an ambiguity shared by all visual cameras. Absorbing them
into the visual covariance would obscure that information boundary.

## Claim boundary

A valid joint calibration establishes only that the named shared and
camera-specific candidate modes were fitted and source-selected on complete
calibration groups under the retained rules. It does not prove that the basis is
complete, that camera errors are stationary, that target coverage is calibrated,
that a BayesianPhysTwin update is beneficial, that a Causal4D intervention is
valid, or that the method is state of the art. Those remain separate fresh-object
held-out gates.
