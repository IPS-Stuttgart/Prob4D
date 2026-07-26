# Data Format

## Prediction Manifest

`prob4d-motioncrafter` writes a versioned `predictions.json`:

```json
{
  "format_version": 1,
  "motioncrafter_commit": "<git sha>",
  "config": {
    "window_size": 25,
    "overlap": 8
  },
  "temporal_lineage": {
    "schema_version": 1,
    "model": "motioncrafter_sliding_window_v1",
    "frame_index_source": "prediction archive frame_indices",
    "source_bounds": "inclusive source-video frame identifiers",
    "products": {
      "disjoint_baseline": {"window_size": 25, "overlap": 0},
      "latent_linear_baseline": {"window_size": 25, "overlap": 8},
      "overlap_windows": {
        "window_size_source": "prediction archive frame count",
        "overlap": 0
      }
    }
  },
  "overlap_windows": [
    {
      "window_id": "window_0000",
      "path": "windows/window_0000.npz",
      "start_frame": 0,
      "stop_frame": 25
    }
  ],
  "disjoint_baseline": "baseline_disjoint.npz",
  "latent_linear_baseline": "baseline_latent_linear.npz"
}
```

The temporal-lineage section identifies the exact MotionCrafter sliding-window
rule used by each product. Together with an archive's absolute `frame_indices`,
it determines, for every output frame, the inclusive minimum and maximum source
frame and the internal windows that contributed to the output. Causal consumers
must require `source_frame_max < cutoff_frame`. Unknown lineage schemas or
manifests without enough legacy configuration fail closed. Version-1 manifests
created before this field was added can be audited without rerunning the GPU
model when their `config.window_size` and `config.overlap` fields are present.

Every prediction archive contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `frame_indices` | `T` | Absolute source-video frame IDs |
| `point_map` | `T x H x W x 3` | Decoded points in the local window gauge |
| `valid_mask` | `T x H x W` | Valid geometry samples |
| `scene_flow` | `T x H x W x 3` | Optional forward flow vectors |
| `deform_mask` | `T x H x W` | Optional valid-flow mask |
| `ray_directions` | `T x H x W x 3` | Optional calibrated viewing rays |

Files without `frame_indices` are accepted only when the caller supplies an
explicit start frame. This prevents local clip indices from being mistaken for
global video indices.

`prob4d-motioncrafter --frame-start/--frame-stop/--frame-stride` writes source
video frame IDs directly into every baseline and overlap window. PhysTwin
evaluation requires these absolute IDs because its calibration, depth frames,
manual tracks, and physical trajectories all use the original sequence index.

## Phys4D Observation Belief

`prob4d-export-observation` writes a non-pickled NPZ implementing schema
`phys4d.observation_belief`, version 1. The descriptor and every array name,
dtype, shape, and byte payload contribute to one content address. The required
arrays are:

| Field | Shape | Meaning |
| --- | --- | --- |
| `declared_frame_ids` | `F` | Strictly increasing admissible source frames |
| `mean_xyz_m` | `M x 3` | Observation means; metric only when explicitly anchored |
| `frame_ids` | `M` | Absolute source frame per row |
| `entity_ids` | `M` | Persistent pixel or material identity per row |
| `view_indices` | `M` | Index into `view_names` |
| `window_indices` | `M` | Index into `window_names` |
| `correlation_group_ids` | `M` | Effective robust-likelihood group |
| `factor_group_ids` | `M` | Shared low-rank latent identity |
| `prior_reliability` | `M` | Residual-independent feeder reliability |
| `association_probability` | `M` | Association support, kept separate from reliability |
| `local_covariance_m2` | `M x 3 x 3` | Conditional point covariance excluding gauge uncertainty |
| `low_rank_factor_m` | `M x 3 x R` | Coherent joint-gauge covariance factor |
| `group_ids` | `G` | Sorted unique effective groups |
| `group_prior_nominal_probability` | `G` | Frozen nominal-component prior |
| `group_composite_weight` | `G` | Evidence-power cap in `(0, 1]` |

The causal cutoff is exclusive: all declared and observed frames must satisfy
`frame_id < causal_frame_stop`. Before loading payloads, the exporter selects
only independently decoded overlap windows with
`stop_frame <= causal_frame_stop`; all gauge estimation, disagreement
accumulation, uncertainty construction, and row export are recomputed from that
prefix. Excluded future payloads are neither opened nor hashed.

The descriptor records `coordinate_mode`. `gauge_relative` artifacts use an
arbitrary first-window reference and do not authorize metric state updates.
`metric_anchored` artifacts require an independent `Sim(3)` anchor and include
its digest, mean, and covariance in the content address. See
[the observation-export contract](observation-belief-export.md) for the exact
causal, gauge, and reliability semantics.

## Ground Truth

Real evaluation expects a compressed NumPy archive with:

```text
frame_indices: T
point_map:     T x H x W x 3
valid_mask:    T x H x W
scene_flow:    T x H x W x 3  # optional
deform_mask:   T x H x W      # required with scene_flow
```

`point_map` and `scene_flow` must use one metric world coordinate system for
the entire sequence. Camera-centric depth maps must be transformed before
evaluation. Prediction and truth arrays must use the same crop, resolution,
and source-frame sampling.

## Results

Each ablation writes:

- `ablation.json`: metadata, calibration report, and complete rows;
- `ablation.csv`: flat machine-readable table; and
- `ablation.md`: compact paper-note table.

The JSON metadata distinguishes exact upstream baselines from synthetic
proxies and records when metric anchors are simulated from benchmark truth.

`prob4d-phystwin` writes one `experiment_zero.json` with a separately fitted
gauge for each MotionCrafter baseline, same-view geometry summaries, manual
track flow methods and frozen fusion weights, held-out-view surface coverage,
input and output hashes, and explicit claim boundaries.

`prob4d-phystwin-state` writes a `causal_source_lineage` audit containing the
endpoint output frame, inclusive source-frame bounds, contributing internal
window IDs, the causal cutoff, and the fail-closed admission decision. The
state experiment does not open metric or manual-track evaluation data when the
endpoint depends on a source frame at or after the cutoff.
