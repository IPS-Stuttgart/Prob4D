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

Provider-specific prediction manifests can be normalized without payload loading
via the stable [`prob4d.source`](docs/source-provider-contract.md) boundary. The
current MotionCrafter adapter is additive; frozen manifests and provider-v1/v2
observation artifacts retain their existing identities. Portable downstream
contracts are collected under `prob4d.contracts`.

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

New claim-bearing experiments should use the explicit provider-v2 calibrated
command. It requires an independent metric `Sim(3)` prior for the first retained
window plus independently fitted gauge and point covariance calibrations:

```bash
prob4d observation export-calibrated \
  outputs/sequence_name/predictions.json \
  outputs/sequence_name/observation_belief.npz \
  --case-id sequence_name \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/sequence_name/metric_gauge_anchor.json \
  --gauge-covariance-calibration outputs/calibration/gauge.json \
  --point-uncertainty-calibration outputs/calibration/point.json \
  --source-revision "$(git rev-parse HEAD)" \
  --summary-json outputs/sequence_name/observation_belief_summary.json

prob4d-validate-observation \
  outputs/sequence_name/observation_belief.npz
```

Claim-bearing consumers should use the strict loader from the same provider-v2
namespace:

```python
from prob4d.provider_v2 import load_claim_bearing_observation_belief

validated = load_claim_bearing_observation_belief(
    "outputs/sequence_name/observation_belief.npz"
)
```

Provider v2 verifies the executing Prob4D revision and prediction/calibration
compatibility before decoded prediction payloads are opened. The final artifact
binds the provider-v2 manifest identity, export mode, covariance-root mode,
calibration artifact IDs, and runtime-revision evidence into its content address.
Claim-bearing export accepts only installed VCS metadata or a clean source
checkout. `PROB4D_RUNTIME_REVISION` can annotate an exploratory packaged
deployment, but an environment assertion is not accepted as independent evidence.

Use `prob4d observation export-exploratory` for labelled uncalibrated,
pointwise-fallback, legacy-root, or fixed-lag reconstruction controls. Use
`prob4d observation export-v1` for the frozen grouped provider-v1 route. The bare
`prob4d observation export` command prints migration guidance and runs no exporter;
the historical `prob4d-export-observation-belief` executable remains unchanged.

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
explicit threshold. Fixed-lag mode now carries a Schur-complement information
prior when a gauge leaves the active window, rather than fixing that boundary
with zero uncertainty. Its portable all-window covariance still contains only
historical marginal blocks, so it remains an opt-in reconstruction ablation. See
[the causal observation export contract](docs/observation-belief-export.md) and
[provider API version 2](docs/provider-v2.md).

For recursive experiments with several causal observation times, provider v2 also
exposes an append-only `ObservationFactorStreamV1`. Each update references one
schema-v4 unfused factor bundle, admits a new non-overlapping frame interval, and
binds bundle/payload hashes, observation identities, and the previous update ID.
See [append-only observation-factor streams](docs/observation-factor-stream.md)
and the [compatibility matrix](docs/compatibility.md).

### Held-out provider promotion and material identity

Freeze, run, and deterministically replay the independent Prob4D-to-BayesianPhysTwin
promotion gate through the grouped command surface:

```bash
prob4d experiment heldout-provider freeze protocol.json --output promotion-lock.json
prob4d experiment heldout-provider run promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results query-results.raw.json \
  --output-dir outputs/promotion
prob4d experiment heldout-provider verify promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results outputs/promotion/query_results.sealed.json \
  --report outputs/promotion/promotion_report.json
```

The target-free lock binds complete object/session splits, source and model
identities, comparison arms, calibration and guard artifacts, bootstrap settings,
and decision margins. The report composes the provider-competence decision with
separate guarded-query superiority, harmful-update, worst-group, coverage,
technical-failure, and exact-fallback gates. See
[held-out provider promotion](docs/heldout-provider-promotion.md).

Portable cross-window material-identity streams and source-calibrated mixtures can
be inspected without rewriting local observation IDs:

```bash
prob4d identity validate-stream material-identities.json
prob4d identity build-mixture mixture-config.json \
  --output material-identity-mixture.json
prob4d identity marginalize material-identity-mixture.json likelihoods.json
```

See [the material-identity command line](docs/material-identity-cli.md),
[append-only hypothesis streams](docs/material-identity-stream.md), and
[identity marginalization](docs/material-identity-marginalization.md).

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

Continuous integration tests Python 3.10 through 3.14, including the declared
NumPy 1.24.0 floor, checks provider manifests and clean-checkout runtime
attestation, builds source and wheel distributions, audits the extracted source
archive, installs each artifact in isolation, and smoke-tests every installed
command including the explicit provider-v2 surfaces.

The implementation has been exercised at full `25 x 320 x 640` window size on
an RTX 6000 Ada host. Frame-level covariance intersection is the production
default; the more expensive per-pixel CI weight search remains available for
small diagnostic experiments.

The pinned benchmark loads serialized MotionCrafter point and flow fields in
`float32` by default and records `dense_storage_dtype` in every fused artifact.
Low-dimensional gauge estimation and covariance calculations still use
`float64`. Stable provider loaders retain the historical `float64` default unless
the caller explicitly selects another storage mode. Frame-local ray access also
avoids full-window normalization temporaries in cross-fitted overlap diagnostics.
See [dense-memory execution](docs/dense-memory.md).

Dense fused outputs now use an immutable `FusedSequence` contract: public
construction defensively copies and normalizes arrays, active point/flow
covariances fail closed on asymmetry or indefiniteness, and every retained array
is read-only. Internal fusion and artifact-loading paths can transfer ownership of
private arrays to avoid a second dense copy while preserving the same validation.
See [the fused-sequence contract](docs/fused-sequence-contract.md).


Dense fusion now preserves structured ray-parallel/lateral covariance until a
representative CI sample or active spatial tile is needed. Covariance-intersection
weights are still optimized once per complete frame/mask pattern and reused for
all tiles, so `fusion_tile_size` changes temporary memory rather than estimator
semantics. A deterministic process-level benchmark records peak RSS, timing, and
an output digest. See [tiled dense fusion](docs/tiled-fusion.md).

Provider evaluation likewise processes active point, covariance, flow, and seam
rows in bounded spatial chunks. Prefix and oracle alignment modes apply their
fitted transforms during accumulation instead of materializing complete
transformed sequences. The provider report records the selected chunk size, and
a deterministic process benchmark measures peak RSS and numerical agreement. See
[bounded-memory provider evaluation](docs/streaming-evaluation.md).

A separate opt-in workflow runs the fusion and evaluation benchmarks at the full
`25 x 320 x 640` shape on the IPS self-hosted `nvidia-smi` runner. It captures
CPU, RAM, GPU, software, revision, timing, RSS, retained-storage, and output-digest
evidence, with optional eager-versus-mmap loading for runner-local prediction
bundles. See [production memory profiling](docs/production-memory-profile.md).
