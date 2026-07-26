# Prob4D

Probabilistic recursive fusion for long-horizon 4D reconstruction from
overlapping [MotionCrafter](https://github.com/TencentARC/MotionCrafter)
windows.

Prob4D treats every window's global gauge as an uncertain `Sim(3)` state. It
uses duplicate predictions for the same frame and pixel to estimate relative
gauges, calibrates a compact along-ray/lateral uncertainty model, and fuses
shared-backbone estimates using covariance intersection. For Bayesian-PhysTwin,
it also exports causally sealed metric observations with explicit covariance,
reliability, dependence, and source-lineage semantics.

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
python -m pytest
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

It also records a versioned temporal-lineage contract for MotionCrafter's
internal sliding windows. Causal consumers can therefore determine the exact
source-frame bounds and contributing internal windows for any output frame.

### Portable Bayesian observation export

Export a provider-neutral observation artifact only after supplying an
independent metric `Sim(3)` prior for the first retained overlap window:

```bash
prob4d-export-observation-belief \
  outputs/sequence_name/predictions.json \
  outputs/sequence_name/observation_belief.npz \
  --case-id sequence_name \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/sequence_name/metric_gauge_anchor.json \
  --summary-json outputs/sequence_name/observation_belief_summary.json

prob4d-validate-observation \
  outputs/sequence_name/observation_belief.npz
```

The exporter opens only independently decoded windows wholly before the
exclusive cutoff. It recomputes alignment, gauge estimation, overlap
disagreement, uncertainty, and residual-independent reliability on that prefix.
Appending post-cutoff windows therefore cannot change the exported artifact ID.
Association probability remains separate from prior reliability, and gauge
uncertainty is represented as a shared low-rank factor rather than counted again
inside each local covariance.

The production default is a causal sequential spanning tree with the **full
joint cross-window gauge covariance** propagated from the metric anchor. A rank
cap is accepted only when the retained covariance-trace fraction satisfies the
explicit threshold. The legacy fixed-lag covariance is available solely as an
opt-in reconstruction ablation because its current boundary treatment fixes
marginalized gauges at their posterior means. See [the causal observation export
contract](docs/observation-belief-export.md).

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

## PhysTwin Experiment Zero

MotionCrafter can be tested on a real deformable interaction before any learned
physics integration is designed. Preserve absolute source-frame IDs while
running a monocular clip that crosses PhysTwin's locked test boundary:

```bash
prob4d-motioncrafter /path/to/double_stretch_sloth/color/0.mp4 \
  --upstream-root /path/to/MotionCrafter \
  --output-dir outputs/double_stretch_sloth_camera0 \
  --cache-dir /path/to/huggingface-cache \
  --frame-start 110 --frame-stop 160
```

Then fit one global `Sim(3)` on frames 110--133 and evaluate later frames
against same-view metric depth, sparse manual 3D tracks, and the two held-out
RGB-D cameras:

```bash
prob4d-phystwin \
  outputs/double_stretch_sloth_camera0/predictions.json \
  /path/to/double_stretch_sloth \
  outputs/double_stretch_sloth_camera0/experiment_zero.json \
  --fit-end-frame 134 \
  --physics-trajectory /path/to/released/inference.pkl \
  --corrected-trajectory /path/to/bayesian_anchor/trajectory.pkl
```

The evaluator reports raw visual flow, raw and discrepancy-corrected PhysTwin
flow, 50/50 fusion, and a scalar inverse-training-MSE fusion. The latter is a
contained train/test baseline, not calibrated per-pixel uncertainty. See the
experiment protocol for the exact claim boundary.

For independently seeded diffusion runs, estimate and calibrate the empirical
3D flow covariance before fusing it with the physical proposal:

```bash
prob4d-phystwin-uncertainty /path/to/double_stretch_sloth output.json \
  --manifest outputs/diff_seed101/predictions.json \
  --manifest outputs/diff_seed202/predictions.json \
  --fit-end-frame 134 \
  --physics-trajectory /path/to/released/inference.pkl \
  --corrected-trajectory /path/to/bayesian_anchor/trajectory.pkl
```

To test the state-space direction causally, create a separate prefix-aligned
bundle. With a 25-frame window and `--fit-end-frame 134`, the first selected
source frame must be 109 so that the endpoint at frame 133 depends only on
frames 109--133:

```bash
prob4d-motioncrafter /path/to/double_stretch_sloth/color/0.mp4 \
  --upstream-root /path/to/MotionCrafter \
  --output-dir outputs/double_stretch_sloth_camera0_causal \
  --cache-dir /path/to/huggingface-cache \
  --frame-start 109 --frame-stop 159

prob4d-phystwin-state \
  outputs/double_stretch_sloth_camera0_causal/predictions.json \
  /path/to/double_stretch_sloth state_forecast.json \
  --product disjoint \
  --fit-end-frame 134 \
  --physics-trajectory /path/to/released/inference.pkl \
  --corrected-trajectory /path/to/bayesian_anchor/trajectory.pkl
```

`prob4d-phystwin-state` audits the endpoint before reading metric or manual-track
evaluation data. It rejects the run whenever the endpoint's maximum source
frame is at or after `--fit-end-frame`. The latent overlap product is normally
a reconstruction-only control because an endpoint blended across windows can
depend on post-boundary RGB. Older manifests can be audited from their stored
window configuration without rerunning MotionCrafter.

## Matched VGGT Comparison

Export both official VGGT point constructions for the same dataset list. The
VGGT dependency and checkpoint remain external to this lightweight package:

```bash
prob4d-vggt-baseline \
  --dataset-root /path/to/processed/Sintel_video \
  --vggt-root /path/to/facebookresearch/vggt \
  --output-root outputs/vggt \
  --resume
```

For an exploratory model-combination ablation, align VGGT to the Prob4D gauge
using prediction-only correspondences and export a fixed blend grid:

```bash
python scripts/export_vggt_prob4d_blends.py \
  --prob4d-results-dir /path/to/prob4d/results \
  --vggt-prediction-dir outputs/vggt/depth_unprojected \
  --output-dir outputs/vggt-prob4d-blends
```

Blend weights are hyperparameters and require a scene-family-held-out
selection protocol. They are not uncertainty-calibrated fusion weights.

## Repository Boundary

This repository owns code, tests, run definitions, and portable observation
contracts. Videos, model weights, decoded predictions, and generated results are
gitignored. Paper-facing result tables, figures, frozen run manifests, and exact
Prob4D/MotionCrafter commit identifiers belong in the canonical
[`FlorianPfaff/BayesianPhysTwin-Paper`](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper)
project-notes repository. The earlier date-stamped proposal repository remains
an archival planning artifact.

## Development

The core package intentionally depends only on NumPy. Torch, Diffusers, Decord,
and model-loading dependencies are imported lazily by `prob4d-motioncrafter`.
The tested GPU environment is pinned separately under `environments/`.

Continuous integration tests Python 3.10, 3.12, and 3.14, builds the source and
wheel distributions, validates package metadata, installs the wheel in an
isolated environment, and smoke-tests every installed command.

The implementation has been exercised at full `25 x 320 x 640` window size on
an RTX 6000 Ada host. Frame-level covariance intersection is the production
default; the more expensive per-pixel CI weight search remains available for
small diagnostic experiments.
