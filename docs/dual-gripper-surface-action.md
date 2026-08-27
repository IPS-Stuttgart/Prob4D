# Dual-gripper surface action bridge

`prob4d.dual_gripper_surface_action` converts two rigid end-effector pose
trajectories into the sampled robot surface-point trajectories consumed by
PointWorld.

It is an additive source-qualification artifact for the proposed
PointWorld–Flat'n'Fold study. It does not modify PointWorld, the production
Prob4D provider, or BayesianPhysTwin.

## Motivation

The released PointWorld input pipeline obtains `robot_flows` by sampling robot
surface points from a compatible kinematic model over the full action horizon.
Its built-in conversion supports DROID/Panda and BEHAVIOR embodiments.

Flat'n'Fold instead exposes two Baxter end-effector tracker streams containing
position, quaternion orientation, and gripper state, plus one electric-gripper
STL mesh. The inspected public source does not establish complete Baxter joint
trajectories or a PointWorld-compatible Baxter URDF sampler.

The narrow source-only bridge is therefore:

1. load the exact gripper STL bytes outside Prob4D;
2. deterministically sample a fixed set of points and normals;
3. bind the sampled template by SHA-256;
4. express the right and left samples in their respective tracker frames using a
   separately bound tracker-to-template calibration;
5. transform the same material surface points with each timestamped tracker pose;
   and
6. retain binary gripper state as the released PointWorld bimanual feature.

This is a **dual-gripper-surface** action representation. It is not claimed to be
equivalent to PointWorld's released DROID or BEHAVIOR full-robot inputs.

## Construction

```python
from prob4d.dual_gripper_surface_action import (
    dual_gripper_surface_action_from_tracker_poses,
)

action = dual_gripper_surface_action_from_tracker_poses(
    action_id="garment-07-demo-03-action-000",
    frame_indices=absolute_frames,
    surface_points_tracker=points_right_then_left,
    surface_normals_tracker=normals_right_then_left,
    positions_world_from_tracker=positions_right_then_left,
    quaternions_world_from_tracker_wxyz=quaternions_right_then_left,
    gripper_open=open_right_then_left,
    template_id=sampled_template_sha256,
    tracker_calibration_id=tracker_to_template_calibration_sha256,
    pose_stream_id=pose_stream_manifest_sha256,
    timestamp_association_id=timestamp_association_sha256,
)

action.to_npz("action-000.dual-gripper.npz")
pointworld_fields = action.pointworld_sample()
```

The public Flat'n'Fold parser uses quaternion order `[w, x, y, z]` and rounds
components to three decimal places. The bridge deterministically normalizes each
finite nonzero quaternion before constructing the rotation matrix.

## Artifact contract

The NPZ schema is `prob4d.dual-gripper-surface-action-window-npz` version 1.
It stores:

- absolute frame indices;
- stable surface-point identities;
- right/left arm and template-point indices;
- `T × N × 3` robot positions, normals, and magenta colors;
- complete rigid-template support;
- `T × 2` right/left gripper state;
- template, tracker-calibration, pose-stream, and timestamp-association IDs; and
- explicit action, identity, and coordinate semantics.

Arm order is fixed as right then left, matching PointWorld's bimanual feature
allocation. Point identity is derived from template ID, tracker-calibration ID,
arm, and source template index. It is therefore stable across action windows
only when the exact hardware template and calibration match.

## Remaining information gap

The bridge code can transform a frozen template and pose stream, but the public
repository alone does not provide enough evidence to instantiate a claim-bearing
Flat'n'Fold action artifact. Source qualification still has to bind:

- the exact STL bytes and deterministic surface-sampling report;
- tracker-to-template transforms for both arms;
- timestamped right/left pose and gripper-state bytes;
- one-to-one camera/action associations with a frozen maximum skew;
- metric frame orientation and units;
- technical-failure and missing-frame handling; and
- the exact PointWorld checkpoint/runtime using the resulting representation.

The gripper-only representation also creates embodiment shift relative to the
released PointWorld training domains. Its source competence must be measured
before a held-out target garment is opened.

## Claim boundary

A valid action archive proves deterministic rigid transformation, identity,
shape, dtype, and provenance semantics for its exact inputs. It does not prove
correct mesh-to-tracker calibration, timestamp synchronization, PointWorld
compatibility, provider accuracy, uncertainty calibration, Prob4D fusion
benefit, BayesianPhysTwin benefit, Causal4D benefit, or deployment safety.
