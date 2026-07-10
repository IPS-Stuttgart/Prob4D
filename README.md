# Prob4D

Probabilistic recursive fusion for long-horizon 4D reconstruction from
overlapping [MotionCrafter](https://github.com/TencentARC/MotionCrafter)
windows.

Prob4D treats every window's global gauge as an uncertain `Sim(3)` state. It
uses duplicate predictions for the same frame and pixel to estimate relative
gauges, calibrates a compact along-ray/lateral uncertainty model, and fuses
shared-backbone estimates using covariance intersection.

## Ablation

The experiment runner evaluates seven variants:

1. disjoint 25-frame MotionCrafter windows;
2. MotionCrafter's latent-space linear overlap blend;
3. decoded-space `Sim(3)` alignment and uniform averaging;
4. naive precision-weighted fusion;
5. covariance intersection;
6. covariance intersection with fixed-lag gauge smoothing; and
7. the smoothed estimator with sparse metric anchors.

Synthetic runs use a decoded crossfade only as a contract-test proxy for row 2.
Real runs consume the exact latent-space baseline emitted by MotionCrafter's
existing `_process_windows` implementation.

## Quick Start

Install the lightweight NumPy estimator:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

Run the correlated synthetic benchmark. Calibration uses a separate seed and
the output contains JSON, CSV, and Markdown tables:

```bash
prob4d-ablate synthetic --output-dir outputs/synthetic --seed 7
```

## MotionCrafter Inference

Create a GPU environment without requiring system package installation:

```bash
scripts/bootstrap_motioncrafter_env.sh ../MotionCrafter ../prob4d-motioncrafter-venv
```

Generate all three prediction products in one model-loading session:

```bash
CUDA_VISIBLE_DEVICES=0 ../prob4d-motioncrafter-venv/bin/prob4d-motioncrafter \
  input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence_name \
  --model-type determ \
  --height 320 --width 640 \
  --window-size 25 --overlap 8 \
  --cache-dir /path/to/shared/huggingface-cache
```

The output manifest references:

- the current disjoint-window baseline;
- MotionCrafter's exact latent-space overlap blend; and
- independently decoded overlapping windows with absolute frame indices.

Prepare a separate calibration sequence and world-coordinate truth files, then
run the real ablation:

```bash
prob4d-ablate real \
  --predictions outputs/test/predictions.json \
  --truth data/test_truth.npz \
  --calibration-predictions outputs/calibration/predictions.json \
  --calibration-truth data/calibration_truth.npz \
  --metric-anchor-every 2 \
  --output-dir outputs/ablation
```

See [the experiment protocol](docs/experiment-protocol.md) for estimator and
evaluation details, [the theoretical benefit and its assumptions](docs/theoretical-benefit.md),
and [the data format](docs/data-format.md) for artifact schemas.

Run the sequence-family-held-out Sintel uncertainty analysis on cached
prediction bundles:

```bash
prob4d-sintel-uncertainty \
  --dataset-dir /path/to/processed/Sintel_video \
  --results-dir /path/to/benchmark/results \
  --output-dir outputs/sintel-uncertainty
```

Calibrate dense-overlap gauge covariance on the predeclared Sintel family
split, then export the simulated sparse-sensor ablation:

```bash
python scripts/calibrate_sintel_gauge_covariance.py \
  --dataset-dir /path/to/processed/Sintel_video \
  --results-dir /path/to/benchmark/results \
  --output outputs/gauge-calibration.json

python scripts/export_heldout_sparse_gauge_anchors.py \
  --dataset-dir /path/to/processed/Sintel_video \
  --results-dir /path/to/benchmark/results \
  --gauge-calibration outputs/gauge-calibration.json \
  --output-dir outputs/sparse-gauge-anchors
```

The sparse-anchor exporter samples ground-truth 3D points to simulate an
associated depth/LiDAR sensor. It is an explicitly sensor-assisted ablation,
not a monocular input protocol.

## Repository Boundary

This repository owns code, tests, and run definitions. Videos, model weights,
decoded predictions, and generated results are gitignored. Paper-facing result
tables and figures belong in `FlorianPfaff/2026-07-Prob4D-Paper`, together with
the run manifest and exact Prob4D/MotionCrafter commit identifiers.

## Development

The core package intentionally depends only on NumPy. Torch, Diffusers, Decord,
and model-loading dependencies are imported lazily by `prob4d-motioncrafter`.
The tested GPU environment is pinned separately under `environments/`.

The implementation has been exercised at full `25 x 320 x 640` window size on
an RTX 6000 Ada host. Frame-level covariance intersection is the production
default; the more expensive per-pixel CI weight search remains available for
small diagnostic experiments.
