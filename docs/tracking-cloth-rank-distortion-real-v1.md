# Tracking Cloth posterior rank--distortion study v1

## Purpose

This study tests the posterior rank--distortion theorem on public real cloth
trajectories. It reuses the previously verified Tracking Cloth parser and
recording-disjoint Gaussian fitting protocol, but replaces the single exact
rank-3 endpoint with a preregistered frontier at retained ranks 0, 1, 2, and 3.

The primary mechanism question is:

> At a fixed retained rank, does the generalized-eigen factor attain the
> globally minimum posterior-normalized covariance contraction predicted by
> the theorem, and is that advantage visible relative to the previous
> Euclidean response-SVD ordering on data-fitted cloth models?

Held-out predictive metrics are reported as evidence, not implied by the
training-model theorem.

## Dataset and admissible recordings

The source is *Tracking Cloth Deformation Using a Single RGB-D Camera*, version
1, DOI `10.5281/zenodo.14644526`. The complete extracted release contains 120
CSV recordings on `gpuserver4090` at

`/home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526`.

Only publisher-style cloth marker tables with the registered marker counts are
admitted:

- A2 cloth: 20 markers;
- A3 cloth: 12 markers.

Rod/stick tables and malformed recordings are rejected before model fitting.
The earlier checksum-verified execution admitted 80 cloth recordings—48 A2 and
32 A3—and evaluated 8,914 held-out windows. This study freezes the same parser,
unit detection, causal window construction, and recording-level split rule.

Raw trajectories are never copied into workflow artifacts. The artifact records
per-file hashes and derived summaries only.

## Registered query and observation

For marker positions `x_t^(i)`, the three-dimensional query is future centroid
displacement

`q_t = mean_i x_(t+h)^(i) - mean_i x_t^(i)`.

The observation is the stacked marker-wise constant-velocity extrapolation

`y_t^(i) = (x_t^(i)-x_(t-lag)^(i)) * (dt_horizon/dt_lag)`.

The frozen protocol uses `lag=3`, `horizon=6`, `stride=6`, five folds, and at
most 2,048 windows per recording.

## Recording-disjoint fitting

Recordings are deterministically ordered by SHA-256 of their relative path and
assigned round-robin to five folds. For each cloth size and fold, only training
recordings are used to estimate the joint Gaussian covariance of `(q,y)`.
Held-out recordings never influence means, covariances, shared-factor
construction, rank selection, or subspaces.

The training covariance is stabilized by preregistered diagonal shrinkage and a
small scale-relative ridge. Its observation block is decomposed as

`S = A + U U^T`,

where `A` is a strictly positive block-diagonal conditional term and `U` is the
remaining supplied shared factor. The theorem is evaluated on this fitted
factor family; the decomposition is a registered mechanism study, not a unique
physical noise decomposition.

## Methods

For each fold, the full data-fitted posterior is the reference. Three equal-rank
factor methods are evaluated at retained ranks 0, 1, 2, and 3:

1. **Optimal generalized eigen.** The new theorem minimizes
   `D = trace(P_full^-1 (P_full-P_reduced))` within `U -> U V`.
2. **Response SVD.** The old Euclidean SVD of the posterior-whitened latent
   response, truncated to the same rank. It identifies the exact nullspace but
   ignores the remainder metric `M=I-U^T S^-1 U` at inexact ranks.
3. **Covariance PCA.** The leading right-singular directions of `U`, retaining
   shared covariance energy without query conditioning.

Rank 3 is expected to be the exact endpoint because the registered query is
three-dimensional and the previous real-data study found numerical exact rank
3 in every fold. This is checked rather than assumed silently.

## Primary quantities

For every fold, rank, and method the artifact records:

- posterior-normalized covariance trace contraction `D`;
- maximum normalized covariance contraction and posterior validity;
- relative gain and posterior-covariance differences from the full model;
- held-out posterior-mean displacement from the full model;
- held-out RMSE, Gaussian NLL, normalized NEES, and 90% coverage;
- held-out normalized mean-shift risk;
- factor payload bytes;
- generalized eigengap and whether the optimum subspace is unique.

The workflow enforces only identities and fail-closed validity conditions that
follow from the registered model. It does **not** require a favorable scientific
outcome. In particular, strict response-SVD improvement counts are reported but
are not a pass criterion.

## Expected interpretation

A positive result requires more than exact rank-3 parity. The meaningful new
evidence is a strict reduction of `D` at rank 1 and/or rank 2 on data-fitted
folds, with no theorem-optimality violation. Held-out RMSE/NLL improvements would
strengthen the empirical case but are not guaranteed by the covariance
objective.

Repeated generalized eigenvalues require care. At a rank boundary inside a
repeated block, the optimum value is unique but the factor subspace is not.
Therefore the artifact compares distortion at every rank and factor covariance
only where the boundary eigengap is strict.

## Claim boundary

This study may support these claims:

- globally optimal posterior trace contraction within each fitted `U -> U V`
  family;
- exact rank-3 posterior parity on the registered local Gaussian query, if
  observed;
- recording-disjoint held-out behavior on public real cloth trajectories;
- superiority or equality to response SVD and covariance PCA for the registered
  theoretical distortion.

It does not establish:

- a learned 4-D reconstruction provider;
- calibrated deployment uncertainty;
- likelihood-preserving observation compression;
- arbitrary task-loss or full posterior-KL optimality;
- recursive or infinite-horizon exactness;
- BayesianPhysTwin or Causal4D physical-performance benefit.

The immutable protocol is
`protocols/tracking-cloth-rank-distortion-real-v1.json`; the self-hosted request
is `protocols/execution_requests/tracking_cloth_rank_distortion_real_v1.json`.
