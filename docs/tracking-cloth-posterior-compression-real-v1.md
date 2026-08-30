# Tracking Cloth real-trajectory posterior-compression study

This experiment evaluates the posterior-preserving shared-noise compression
kernel on the complete public **Tracking cloth deformation** dataset
(DOI `10.5281/zenodo.14644526`). It is a recording-disjoint real-motion-capture
mechanism study, not a learned-provider promotion.

## Why this dataset

The release contains 120 high-quality motion-capture recordings spanning four
fabrics, A2 and A3 sizes, and dynamic shaking, twisting, table collision,
hitting, and self-collision scenarios. Each CSV stores a frame identifier,
timestamp, and marker coordinates. The parser deliberately accepts only the
12-marker A3 and 20-marker A2 cloth-only layouts. Recordings with two additional
rod/stick markers are excluded before model fitting, so no outcome-dependent
selection is introduced.

## Registered query and observations

Within each accepted recording, a causal window uses all cloth-marker
displacements over the preceding six frames, rescaled to the twelve-frame query
horizon. The three-dimensional query is the future displacement of the
cloth-marker centroid. Windows with any missing required coordinate are omitted.
At most 128 evenly spaced complete windows are retained per recording, giving
each recording a bounded contribution.

A2 and A3 recordings are modeled separately because they carry different marker
counts. Within each size, file identities are ordered by SHA-256 and assigned
round-robin to five folds. Every test recording is therefore excluded from the
covariance fit used to evaluate it.

## Local Gaussian model

For each training fold, the joint query/observation covariance is estimated from
real windows, with a fixed 10% diagonal shrinkage and a small scale-relative
ridge. The observation covariance is decomposed exactly as

\[
S = A + U U^\top ,
\]

where `A` is a positive-definite fraction of the 3-D marker-block diagonal and
`U` is the eigensystem factor of the remaining dependence. The fraction is
bounded by the smallest generalized eigenvalue, so the remainder remains
positive semidefinite.

The proposed method retains

\[
\operatorname{range}(U^\top S^{-1} C^\top),
\]

where `C` is the query/observation cross covariance. Since the registered query
is three-dimensional, the exact retained rank is at most three unless numerical
validation requires the exact fallback.

## Comparators and endpoints

The study evaluates:

1. full shared covariance;
2. posterior-preserving compression;
3. equal-rank observation-covariance PCA;
4. conditional block covariance only;
5. prior-only prediction; and
6. a cached full-query gain/covariance message.

Primary endpoints are full/compressed gain error, posterior-covariance error,
realized posterior-mean difference, retained rank, and shared-factor payload.
Recording-disjoint RMSE, Gaussian NLL, normalized NEES, and nominal 90% joint
coverage are diagnostics of the fitted local Gaussian model. A comparator whose
altered covariance makes the query posterior indefinite is retained as an
explicit invalid-posterior result rather than repaired with an outcome-dependent
ridge.

## Execution boundary

The workflow is triggered by a change to the exact execution-request file. It
uses the `gpuserver4090` self-hosted runner and the verified dataset root

```text
/home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526
```

The self-hosted job has read-only repository permissions, no persisted checkout
credentials, and no repository secrets. It uploads only compact JSON/Markdown
results, source-file hashes, and aggregate metrics. Raw CSV trajectories are
never copied into the artifact.

## Claim boundary

A positive result establishes numerical posterior preservation and
query-sufficient shared rank for a specified local Gaussian model estimated from
real cloth trajectories. It does not establish:

- a learned 4-D observation provider;
- deployment-grade covariance calibration;
- superiority of BayesianPhysTwin over its physical fallback;
- Causal4D counterfactual benefit;
- generalization to arbitrary cloths or perception systems; or
- state of the art.

The physical-performance question still requires a prospective provider,
calibration split, and protected target protocol. This experiment isolates the
real-data validity of the compression mechanism itself.
