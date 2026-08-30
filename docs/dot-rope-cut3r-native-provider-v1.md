# DOT rope marker-free CUT3R source experiment v1

## Question

Can the nonlinear shared-gauge uncertainty methods that passed controlled and
real-geometry tests remain useful when the uncertain gauges are produced by a
real image-only 4-D provider rather than imposed analytically?

This protocol uses the public DOT rope dataset and the retained CUT3R checkout.
It is a source-development experiment, not a held-out benchmark.

## Frozen data boundary

The only opened source sequences are `R01`, `R02`, and `R03` in
`R01-10.zip`. The provider job reads exactly the normal-view JPEGs for
`cam001`, frames 1 through 7. It is structurally unable to read marker files.
The later evaluation job downloads and verifies the sealed prediction bundle
before it opens the matching 2-D and 3-D coordinate files.

`R04` through `R70` remain unopened. The experiment does not read UV images,
templates, videos, BayesianPhysTwin outcomes, or Causal4D outcomes.

## Runtime qualification

The historical CUT3R Python-RoPE source attempt is terminal and is not retried.
A first self-hosted job copies the pinned checkout into an isolated workspace,
builds the native CUDA RoPE extension, emits the existing content-bound runtime
receipt, and runs a three-frame synthetic forward pass. The compiled extension
and receipt are transferred as immutable Actions artifacts. The provider job
must reproduce the same runtime artifact ID before opening a DOT image.

## Provider comparison

For every source sequence, CUT3R runs on three fixed frame sets:

- continuous: frames 1--7;
- restarted window A: frames 1--5;
- restarted window B: frames 3--7.

The provider outputs dense point maps, confidences, camera poses, and estimated
intrinsics. Those arrays are compressed and hash-bound. Raw input images are
not retained in the artifact.

Frames 3--5 provide provider-only cross-window correspondences. The 2-D marker
positions are used only after sealing to sample the already generated point
maps. A proper robust Sim(3) maps window B into window A. Frames 1--2 fit the
window-A and continuous metric gauges; frames 6--7 score continuous and stitched
predictions. An identity-stitch control and an oracle window fit are retained.

## Fixed-mean uncertainty comparison

A clustered frame/marker bootstrap estimates the seven-dimensional Sim(3)
parameter covariance. Eight off-axis probes are frozen around the provider
rope's principal axis. Every uncertainty method uses the same plug-in mean and
only changes the predictive covariance:

1. local first order;
2. axis spherical-radial propagation;
3. fourfold scalar inflation;
4. independent pointwise quadratic covariance;
5. shared quadratic curvature covariance;
6. finite quadrature along the dominant rotational gauge plus a linear residual;
7. full three-node tensor Gauss--Hermite propagation;
8. clustered-bootstrap fallback.

The primary proper score is joint Gaussian NLL per coordinate after normalizing
by provider rope span and adding the frozen 2%-of-span observation floor. The
secondary uncertainty endpoint is joint 95% coverage. Reconstruction endpoints
are RMSE divided by ground-truth rope span.

## Interpretation boundary

A completed run establishes at most source-development evidence that a real
provider exhibits a cross-window gauge distribution on three already opened
sequences. It does not establish held-out transfer, empirical calibration,
physical-twin decision value, intervention validity, safety, or state of the
art. Any method or threshold selected from this run must be frozen before a
separately registered request opens any part of `R04`--`R70`.
