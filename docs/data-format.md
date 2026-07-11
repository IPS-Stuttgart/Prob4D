# Data Format

## Prediction Manifest

`prob4d-motioncrafter` writes a versioned `predictions.json`:

```json
{
  "format_version": 1,
  "motioncrafter_commit": "<git sha>",
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
