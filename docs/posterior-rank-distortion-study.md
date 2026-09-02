# Posterior rank--distortion baseline-regret study

This document freezes the synthetic evidence protocol accompanying the exact
rank--distortion theorem in `posterior_rank_distortion.py`.  It is deliberately
separate from real-data evidence: its purpose is to measure how often and by
how much natural truncation rules miss the registered global optimum.

## Frozen strata

The workflow evaluates 128 deterministic seeds in each of four strata:

| Shared-factor rank | Query dimension | Purpose |
| ---: | ---: | --- |
| 7 | 1 | scalar-query limit |
| 7 | 3 | direct continuation of the dense theorem control |
| 14 | 3 | shared rank substantially larger than query rank |
| 28 | 5 | aggressive high-rank compression regime |

Each model has 12 three-dimensional observations and an eight-dimensional
latent physical state.  The non-shared observation covariance is generated as

```text
A = D + F F^T,
```

where `D` is strictly positive diagonal.  The complete innovation covariance is

```text
S = A + U U^T.
```

A query map `H` gives prior query covariance `H H^T` and query/observation
cross-covariance `H F^T`.  This construction makes every covariance block part
of one valid jointly Gaussian model rather than independently sampled matrices.

## Compared same-rank projections

For every retained rank, the study compares:

1. **Generalized-eigen optimum.** The proposed nested frontier minimizing
   normalized posterior-covariance trace contraction.
2. **Posterior-response SVD.** Euclidean singular-vector ordering of
   `U^T S^-1 C^T` after posterior whitening.  This is the natural extension of
   the zero-distortion exact compressor to ranks below the exact threshold, but
   it ignores the remainder metric `I - U^T S^-1 U`.
3. **Latent energy PCA.** Eigenvectors of `U^T U`, retaining directions with the
   largest shared-factor energy.  This represents covariance-energy
   truncation without query conditioning.

All methods receive the same complete innovation covariance, factor, prior,
cross-covariance, row order, and query.  There is no hyperparameter selection
or outcome-dependent projection.

## Registered evidence

For candidate distortion `D_candidate` and the theorem value `D_star`, the
study records

```text
additive regret = D_candidate - D_star,
relative regret = (D_candidate - D_star) / D_star
```

only where `D_star` is numerically positive.  The workflow fails closed if a
baseline beats the claimed optimum beyond numerical tolerance, if the
closed-form and downdate audits disagree, or if the frozen control is absent.

The seed-93, rank-7, query-dimension-3, retained-rank-1 control is fixed in
advance.  Its expected values are approximately

```text
D_star = 0.3703045855
D_SVD  = 0.4632161428
D_SVD / D_star > 1.24
```

This control demonstrates that the proposal is not a relabeling of the current
Euclidean SVD ordering.

## Reproduction

```bash
python scripts/research/posterior_rank_distortion_study.py \
  --seeds 128 \
  --configuration 7:1 \
  --configuration 7:3 \
  --configuration 14:3 \
  --configuration 28:5 \
  --output artifacts/posterior-rank-distortion-study
```

The command writes `frontier-points.csv` and `summary.json`.  The corresponding
GitHub Actions workflow uploads both as a 90-day artifact.

## Claim boundary

The study supports prevalence and effect-size claims for frozen jointly
Gaussian synthetic query models.  It does not by itself establish real-data,
closed-loop, nonlinear, robust-reweighting, or observation-likelihood
performance.  Those questions require a separately frozen physical-data
protocol using exported Prob4D covariance factors.
