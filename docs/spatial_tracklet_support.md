# Spatially stratified causal tracklets and camera-panel support

Prob4D can build causal-prefix scene-flow tracklets with spatially balanced seed
selection and retain the seed-cell lineage of every surviving material track.
This addresses a specific real-provider failure mode: a stream can contain many
valid points while those points are concentrated in too few independent image
regions.

The implementation is additive. The historical regular-grid builder in
`prob4d.causal_tracklets` remains unchanged. New work can opt into
`prob4d.spatial_tracklets`.

## Spatial seed selection

`select_spatial_tracklet_seeds` partitions the first retained frame into a fixed
image-cell grid. In `spatial-stratified` mode it selects at least one valid,
deforming seed from every occupied cell. Within a cell, deterministic anchors
are matched to the nearest unused valid pixel, with row/column tie-breaking and
an optional per-cell quota.

The selector uses only the first causal-prefix frame's validity and deform masks.
It does not use prediction residuals, metric truth, future frames, target
outcomes, or downstream BayesianPhysTwin decisions.

```python
from prob4d.spatial_tracklets import (
    build_spatially_stratified_scene_flow_tracklets,
)

tracklets, report = build_spatially_stratified_scene_flow_tracklets(
    window,
    causal_frame_stop=134,
    seed_stride=8,
    cell_grid_rows=4,
    cell_grid_columns=4,
    maximum_seeds_per_cell=4,
    search_radius_pixels=4,
    maximum_step_error_local=0.05,
    minimum_link_probability=0.05,
    minimum_track_length=2,
)
```

The returned `CausalTrackletSet` retains these metadata fields:

- `seed_selection_policy`;
- requested/effective cell-grid shape and selection summary;
- one `seed_cell_ids_by_track` entry for every retained track;
- selected and retained seed-cell counts; and
- the source-only scientific claim boundary.

`SpatialTrackletReport` keeps the ordinary association/termination report and
separately reports selected versus retained spatial support. A cell that loses
all of its tracks is therefore visible rather than being hidden by the total
track count.

## Spatial correlation groups

Dense neighboring points must not be treated as independent observations merely
because they have distinct point IDs. Use
`spatial_tracklets_to_observation_factors` with the default
`frame-seed-cell` mode to emit one factor per frame and retained seed cell:

```python
from prob4d.spatial_tracklets import (
    spatial_tracklets_to_observation_factors,
)

factors = spatial_tracklets_to_observation_factors(
    tracklets,
    covariance,
    view_id="camera-0",
    correlation_group_mode="frame-seed-cell",
    effective_samples_per_frame=8.0,
)
```

This preserves the original persistent point IDs and gauge ID while retaining a
separate correlation-group identity for every represented seed cell. All cell
factors from one frame share one frozen generalized-Bayes budget:

```text
frame weight = min(1, effective_samples_per_frame / retained_rows_in_frame)
```

Consequently, splitting one frame into cell factors cannot increase its total
weighted row mass relative to `correlation_group_mode="frame"`. The two modes
have the same frame-level likelihood-power cap; the cell mode adds spatially
resolved support and robust-group identities. It remains more conservative than
treating every row as an independent full-weight observation.

The cell split is not a claim that cells are statistically independent. Shared
gauge, provider, camera, clock, and capture bias must still remain explicit
nuisance variables downstream. Increasing total likelihood power based on a
stronger cell-independence assumption requires a separately declared and
validated model rather than silently multiplying the frame budget.

## Camera-panel support audit

`evaluate_camera_panel_tracklet_support` combines several spatially annotated
tracklet sets without averaging their point estimates. The complete camera
roster is a separate required input; it is not inferred from whichever mappings
happen to be supplied at evaluation time. A declared camera missing from
`tracklets_by_view` contributes zero support on every required frame, and an
undeclared camera is rejected.

For every predeclared causal-prefix frame the report records:

- the complete frozen camera roster;
- each contributing camera view;
- the number of represented view-local seed cells in each contributing camera;
- the cameras that independently meet the frozen per-view cell threshold; and
- exact failure reasons for insufficient views, spatial support, or missing
  declared cameras.

Image-cell IDs are view-local. The audit deliberately does not treat the same
numeric image-cell ID in two cameras as the same physical object region.

```python
from prob4d.camera_panel_support import (
    CameraPanelSupportPolicyV1,
    evaluate_camera_panel_tracklet_support,
)

policy = CameraPanelSupportPolicyV1(
    minimum_view_count=2,
    minimum_seed_cell_count_per_view=8,
    minimum_supported_frame_fraction=1.0,
    require_all_declared_views=True,
)
report = evaluate_camera_panel_tracklet_support(
    {
        "camera-0": camera_0_tracklets,
        "camera-1": camera_1_tracklets,
        "camera-2": camera_2_tracklets,
    },
    declared_view_ids=("camera-0", "camera-1", "camera-2"),
    panel_id="session-17-prefix-panel",
    required_frame_indices=tuple(range(109, 134)),
    policy=policy,
)
```

The report is content-addressed and invariant to the input mapping order. Its
identity includes the complete declared roster, so dropping a camera changes the
request and cannot masquerade as a replay of the frozen panel. The report is a
source-only support diagnostic, not a provider-accuracy result. A passing panel
still requires separate source/calibration fitting, held-out provider competence,
and the guarded BayesianPhysTwin promotion gate.

## Recommended protocol use

1. Freeze the camera roster, cell grid, seed stride, per-cell quota, required
   frames, frame-level effective-sample budget, and panel thresholds before
   source residuals or target outcomes are opened.
2. Pass the frozen roster through `declared_view_ids`; retain absent or
   support-negative cameras instead of deleting them after seeing provider
   support or errors.
3. Fit covariance, association, reliability, camera-bias, and timing priors only
   on development/calibration objects or sessions.
4. Keep view-specific and shared visual nuisance variables separate from local
   point covariance.
5. Run the existing held-out provider and BayesianPhysTwin promotion workflow
   once on an unopened object/session cohort.

## Claim boundary

Spatial coverage and camera-panel support establish only that the frozen
causal-prefix provider has distributed support under the declared policy. They
do not establish calibrated uncertainty, provider competence, physical-state
identifiability, BayesianPhysTwin benefit, Causal4D intervention benefit,
deployment safety, or state of the art.
