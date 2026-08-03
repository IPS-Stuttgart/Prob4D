# Experiment Protocol

## Gauge Estimation

Window `k` has a local-to-global similarity transform

```text
G_k = (s_k, R_k, t_k) in Sim(3).
```

For each pair of overlapping windows, Prob4D collects decoded points for the
same absolute frame and pixel. Robust weighted Umeyama iterations estimate the
relative transform. The dense-overlap covariance uses a cluster-robust sandwich
estimator with frame-by-spatial-tile clusters (32 x 32 pixels by default), so
pixels produced by the same model window and local image region are not counted
as independent evidence. Tiny synthetic grids fall back to pointwise clusters,
while direct sparse registrations keep the IID Gauss--Newton covariance unless
cluster IDs are supplied explicitly. Rank-deficient seven-parameter geometry is
rejected instead of receiving spuriously small pseudoinverse variance.

Sequential initialization propagates transform uncertainty through composition
and uses covariance intersection when several previous windows constrain a new
one. Gauge-covariance calibration artifacts are tied to the covariance model
that produced them and must be regenerated after changing that model.

The fixed-lag smoother minimizes whitened nonlinear relative-gauge residuals. It
supports full gauge priors, log-scale observations, and associated sparse 3D
points. When a gauge leaves the active window, its local factors are linearized
and Schur-marginalized into a quadratic boundary prior rather than fixing the
expired state with zero uncertainty. Portable historical output still retains
only per-window marginal blocks, so fixed-lag covariance remains an explicitly
approximate reconstruction control rather than a strict causal-stream artifact.

## Dense Uncertainty

Per-point covariance is parameterized by the viewing ray `r`:

```text
Sigma = sigma_parallel^2 r r^T
      + sigma_lateral^2 (I - r r^T).
```

Depth supplies the initial heteroscedastic trend. Aligned overlap disagreement
adds online evidence. Global parallel and lateral scale factors are learned
from separate calibration scene families; test truth is never used to tune
uncertainty. The final covariance also includes first-order propagation of each
window's uncertain `Sim(3)` gauge.

## Correlated Fusion

Overlapping predictions share both the model and most input frames. The naive
precision baseline intentionally assumes independence and therefore tends to
underestimate covariance. Covariance intersection uses an unknown-correlation
fusion rule.

For one active contributor, the estimate is returned unchanged. Two contributors
retain the frozen scalar grid-search implementation. For three or more active
contributors, production fusion solves one generalized covariance-intersection
simplex problem over all contributors simultaneously. Uniform and independent
precision baselines likewise use one joint multi-input calculation rather than
repeated pairwise updates.

Pixels are grouped by their exact contributor mask. One coherent weight vector is
optimized per frame and mask pattern from a deterministic representative covariance
sample, then applied in chunks. Pointwise CI weights remain an explicit small-scale
diagnostic. Contributor and window order are canonicalized, and the public result
is permutation invariant. This estimator change does not reinterpret the unfused
observation-factor or `ObservationBeliefV1` contracts, which remain the preferred
downstream Bayesian representations when explicit nuisance structure is required.

The separate causal multi-edge gauge-graph diagnostic fuses complete augmented joint
gauge distributions with covariance intersection. Its optional source-only guard
compares direct edges with two-edge paths in representative displacement. A
source/calibration-frozen threshold and minimum per-multi-edge-child cycle count
either admit the unchanged graph or return the exact provider-v2 spanning tree for
the complete case.
The guard never uses target truth or a downstream physical innovation, and neither
graph mode is claim-bearing without a later frozen held-out promotion study.

## Seven Variants

1. Upstream disjoint 25-frame inference.
2. Upstream latent-space linear overlap blending.
3. Decoded `Sim(3)` alignment with uniform averaging.
4. Naive precision-weighted fusion.
5. Covariance intersection.
6. Covariance intersection with fixed-lag gauge smoothing.
7. The smoothed method with sparse metric scale anchors.

The synthetic runner labels row 2 as a decoded proxy. Only a manifest generated
by `prob4d-motioncrafter` provides the exact upstream latent-space baseline.

## Metrics

Accuracy and calibration are intentionally separated:

- `metric_point_rmse` evaluates the online result directly in meters;
- `point_rmse` uses one sequence-level scale/translation fit for comparison
  with MotionCrafter's existing evaluation protocol;
- `seam_rmse`, endpoint error, and drift slope expose long-horizon failures;
- `coverage_95`, Gaussian NLL, and mean marginal Mahalanobis distance evaluate
  dense uncertainty;
- 95% coverage shortfall and minimum per-sequence coverage measure the
  conservative-consistency objective separately from two-sided calibration;
- relative-uncertainty rank correlation and risk at 80% retention test whether
  covariance can identify likely failures; and
- gauge scale, rotation, translation, and normalized squared error are reported
  on benchmark datasets only after estimation is complete.

Dense point errors are not called NEES because pixels are mutually correlated
and the evaluation alignment consumes degrees of freedom. NIS/NEES terminology
is reserved for low-dimensional gauge or downstream object states.

## Metric Anchors

The current benchmark path simulates sparse scale observations from associated
ground-truth point pairs and records this explicitly in result metadata. These
rows demonstrate the estimator behavior but are not sensor-realistic evidence.
Real radar, sparse depth, or odometry observations should replace the simulator
before making a multimodal claim.

## Verified Runtime

The GPU adapter and estimator were smoke-tested on an RTX 6000 Ada with two
25-frame, `320 x 640` independently decoded windows. The decoded CI pass used
about 5 GB peak host memory and completed in roughly 23 seconds after switching
from pointwise to frame-level CI weights. Runtime measurements are diagnostics,
not paper results; benchmark runs must record hardware and commits in their
manifests.

For integration testing only, `scripts/create_smoke_fixture.py` creates a tiny
video and metric truth archive. Its image renderer is deliberately simple and
is not fully coupled to the 3D scene, so results from that fixture must never be
reported as reconstruction evidence.

## PhysTwin Experiment Zero

The first deformable-object test uses one RGB stream from an official
three-camera PhysTwin interaction. MotionCrafter receives camera 0 only. Its
point map is registered to metric PhysTwin world coordinates with one robust
global `Sim(3)` estimated exclusively before the official test boundary.

The later interval has three distinct evaluations:

- object-point error at the same camera pixels against calibrated depth;
- scene-flow and endpoint error at visible manual 3D tracks; and
- symmetric object point-set distance and held-out-only coverage in cameras 1
  and 2.

Released and discrepancy-corrected PhysTwin trajectories are evaluated as
physical flow proposals. Physics motion is transferred from the nearest
surface node to each transformed visual observation. Fixed 50/50 fusion and a
single inverse-training-MSE weight are baselines. The scalar weight is fitted
from preboundary manual tracks and is frozen before test evaluation.

This protocol does not yet supply `Sigma_vis`: deterministic MotionCrafter has
no per-pixel covariance, and one scalar training weight is not a substitute.
Held-out-camera point-set distance measures surface coverage, not dense
cross-view correspondence. A positive result would motivate sampled visual
covariance and recursive state-space fusion; a poor visual baseline motivates
deformable-domain adaptation first.

### Sampled visual uncertainty

The follow-up uncertainty protocol runs the diffusion MotionCrafter checkpoint
with at least two independent seeds. Every sample receives its own train-only
global `Sim(3)`, and manual observations are intersected by absolute frame and
persistent track identity. The empirical 3D flow covariance is shrunk toward
its isotropic component and given a 0.25 mm axis floor. One global scale makes
the preboundary mean normalized innovation squared equal three; no future
track label enters that scale.

The physical covariance is the regularized preboundary error second moment,
which deliberately retains systematic simulator bias. Gaussian products are
reported beside the ensemble mean, raw and corrected physics, fixed fusion,
and scalar inverse-MSE fusion. Coverage is sparse-track coverage, not a claim
of calibrated dense per-pixel uncertainty.

### Endpoint state update and occlusion forecast

The state-space diagnostic takes MotionCrafter's metric-aligned 3D positions
at the final preboundary frame, associates each observed track with the nearest
PhysTwin surface node, preserves the observed endpoint offset, and follows the
node under the known future action. It receives no future RGB, depth, or visual
flow. Per-frame MotionCrafter geometry and endpoint persistence are baselines.
For a causal endpoint, the disjoint window ends at the boundary; latent overlap
is not admissible because a blended window can contain future RGB.

Visible-track errors are paired by frame and track. A circular moving-block
bootstrap over frames uses five-frame blocks and 10,000 replicates. The
all-finite-track evaluation additionally includes camera-missing observations.
Manual track identities provide the sparse pixel association; manual 3D values
are used only for evaluation. Bias-corrected and truth-initialized forecasts
are explicitly label-dependent controls and cannot support the main claim.
An association-only control follows the same nearest simulator node without
preserving the MotionCrafter endpoint offset; its direct paired comparison
tests whether the visual state update contributes beyond node association.
