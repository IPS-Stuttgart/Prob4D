# Tracking Cloth posterior rank--distortion frontier v1

## Question

The exact shared-factor compressor proves that a fixed Gaussian query needs at
most the query dimension when zero posterior distortion is required.  The new
rank--distortion theorem in the stacked kernel PR gives a globally optimal
projection at every smaller retained rank.  This study asks whether that
stronger ordering matters on recording-disjoint real cloth models, rather than
only on controlled dense matrices.

## Frozen real-data model

The study reuses the complete public Tracking Cloth Deformation release and the
model construction already validated by the query-portfolio study:

- 120 CSV recordings in the official release;
- 80 compatible cloth recordings after the frozen parser excludes rod/stick
  recordings;
- A2 and A3 cloth sizes treated separately;
- five deterministic recording-disjoint folds per size;
- causal marker motion over six frames as the observation;
- future marker displacement over twelve frames as the query source;
- at most 96 windows per recording with stride four;
- the same covariance shrinkage, ridge, structured conditional covariance, and
  shared-factor decomposition as the prior study; and
- deterministic portfolios of 1, 2, 4, 8, 12, and, for A2, 20 markers.

The base model script and protocol are bound by Git blob SHA-1.  Any change to
that source model construction causes the experiment to fail closed.

## Methods at equal rank and equal representation size

For each of the 55 fold/query cases, the experiment evaluates all retained
ranks from zero through the original shared-factor rank.

1. **Generalized-eigen optimum.**  The discarded subspace minimizes
   `trace(P_full^-1 (P_full - P_reduced))` globally within the orthogonal
   `U -> U V` factor-projection family.
2. **Posterior-response SVD.**  Directions are ordered by the Euclidean SVD of
   the posterior-whitened latent response used by the exact compressor.
3. **Shared-factor PCA.**  Directions are ordered by covariance energy in the
   supplied observation factor.

At a fixed rank, all methods store the same number of floating-point scalars:
an observation-by-rank factor and an original-rank-by-rank projection.  The
comparison is therefore simultaneously a same-rank and same-raw-byte-budget
comparison under the resident-model assumption.

## Registered endpoints

The complete training-model frontier reports, at every rank:

- normalized posterior covariance trace contraction and its value per query
  dimension;
- maximum normalized covariance contraction;
- expected posterior-normalized mean-shift risk;
- positive-definiteness of the reduced posterior;
- factor, projection, and combined raw byte counts; and
- for the optimum, the generalized eigengap and whether the optimizing
  subspace is unique.

For average trace-contraction budgets of 0.01, 0.05, 0.10, and 0.25 per query
dimension, each method selects the minimum numerically valid rank.  The primary
budget is 0.05.  Rank savings against both baselines are primary descriptive
outputs.  The theorem enforces that the optimum cannot use more rank than an
equal-rank baseline for the registered objective; the frequency and magnitude
of strict savings are empirical results, not pass criteria.

Held-out recording windows are evaluated only after all projections and ranks
have been selected from the training-fold Gaussian model.  RMSE, normalized
NEES, joint 90% coverage, and Gaussian NLL per query dimension are reported as
model-misspecification diagnostics.  They are not used for rank selection and
are not required to favor the theorem-derived projection.

## Numerical controls

The implementation independently reconstructs every projected posterior through
the structured block-diagonal-plus-low-rank solver.  It audits the closed-form
optimal distortion and mean-shift risk against those dense query-posterior
quantities.  It also verifies that neither equal-rank baseline beats the
registered optimum beyond numerical tolerance.

Rank cuts inside repeated generalized-eigenvalue blocks have a unique optimum
value but a non-unique optimizing subspace.  The result records this distinction
rather than treating one LAPACK basis as canonical.

## Claim boundary

A positive run establishes a real-data rate--distortion result for frozen local
Gaussian query models built from recording-disjoint motion-capture folds.  It
does not preserve observation likelihood, prove recursive exactness, validate a
learned 4D provider, demonstrate BayesianPhysTwin or Causal4D control benefit,
establish deployment calibration, or claim state of the art.  Raw trajectories
are not copied into the evidence artifact.
