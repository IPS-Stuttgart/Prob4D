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
global video indices. Negative absolute frame IDs, non-finite valid points,
active non-finite flow, deformation outside valid geometry, and invalid active
ray directions are rejected.

`prob4d-motioncrafter --frame-start/--frame-stop/--frame-stride` writes source
video frame IDs directly into every baseline and overlap window. PhysTwin
evaluation requires these absolute IDs because its calibration, depth frames,
manual tracks, and physical trajectories all use the original sequence index.

## Portable Observation Belief

`prob4d-export-observation-belief` writes the provider-neutral
`phys4d.observation_belief` version-1 NPZ consumed by Bayesian-PhysTwin and
validated independently by Causal4D. Its descriptor and every array name, dtype,
shape, and byte payload are covered by one artifact ID.

The exporter admits only independently decoded overlap windows whose complete
source interval lies before the exclusive `causal_frame_stop`. It decides
admissibility from the manifest before opening prediction payloads and then
recomputes alignment, gauge estimation, overlap disagreement, uncertainty, and
prior reliability on the admitted prefix. A content-addressed metric gauge
anchor for the first retained window is mandatory before coordinates can be
labelled in metres.

The arrays are:

| Field | Shape | Meaning |
| --- | --- | --- |
| `declared_frame_ids` | `F` | Sorted absolute frames authorized by the artifact |
| `mean_xyz_m` | `N x 3` | Metric observation means |
| `frame_ids` | `N` | Absolute frame identity per row |
| `entity_ids` | `N` | Dense pixel/entity identity per row |
| `view_indices` | `N` | Index into descriptor `view_names` |
| `window_indices` | `N` | Index into descriptor `window_names` |
| `correlation_group_ids` | `N` | Effective composite-likelihood group |
| `factor_group_ids` | `N` | Shared gauge-latent group |
| `prior_reliability` | `N` | Residual-independent nominal-source support |
| `association_probability` | `N` | Association support, kept separate from reliability |
| `local_covariance_m2` | `N x 3 x 3` | Conditional point covariance in square metres |
| `low_rank_factor_m` | `N x 3 x R` | Shared coherent joint-gauge covariance factor |
| `group_ids` | `G` | Sorted effective group IDs |
| `group_prior_nominal_probability` | `G` | Prior nominal probability per group |
| `group_composite_weight` | `G` | Information-cap weight in `(0, 1]` |

For `K` retained windows, the production exporter propagates one joint gauge
covariance `Sigma_g` of shape `7K x 7K` and obtains a deterministic root
`Sigma_g = L L^T`. For a row from window `k`,

```text
U_i = J_i L_k,
L_k = rows 7k : 7(k + 1) of L.
```

All rows therefore share one factor group. This represents per-window gauge
uncertainty and cross-window covariance. A downstream estimator that keeps
these factors as explicit nuisance variables must use `local_covariance_m2`
without adding the gauge marginal again. Rank reduction is allowed only when the
exported metadata records a retained covariance-trace fraction above the
explicit threshold.

See [the causal export contract](observation-belief-export.md) for the metric
anchor schema, append-invariant source digest, joint posterior construction,
fixed-lag limitation, command, and claim boundary.

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
