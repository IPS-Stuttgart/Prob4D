# Experiment Protocol

## Gauge Estimation

Window `k` has a local-to-global similarity transform

```text
G_k = (s_k, R_k, t_k) in Sim(3).
```

For each pair of overlapping windows, Prob4D collects decoded points for the
same absolute frame and pixel. Robust weighted Umeyama iterations estimate the
relative transform and a seven-parameter covariance approximation. Sequential
initialization propagates transform uncertainty through composition and uses
covariance intersection when several previous windows constrain a new one.

The fixed-lag smoother minimizes whitened nonlinear relative-gauge residuals.
It supports full gauge priors, log-scale observations, and associated sparse 3D
points. Old states leave the optimization window and remain fixed, preserving
the online-capable interpretation.

## Dense Uncertainty

Per-point covariance is parameterized by the viewing ray `r`:

```text
Sigma = sigma_parallel^2 r r^T
      + sigma_lateral^2 (I - r r^T).
```

Depth supplies the initial heteroscedastic trend. Aligned overlap disagreement
adds online evidence. Global parallel and lateral scale factors are learned
from an entirely separate calibration sequence; test truth is never used to
tune uncertainty.

## Correlated Fusion

Overlapping predictions share both the model and most input frames. The naive
precision baseline intentionally assumes independence and therefore tends to
underestimate covariance. Covariance intersection uses an unknown-correlation
fusion rule. Production runs optimize one weight per overlapping frame from a
representative covariance sample. Pointwise CI weights remain available as an
explicit small-scale diagnostic.

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
  dense uncertainty; and
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

