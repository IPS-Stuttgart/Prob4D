# Tracking Cloth rank--distortion evidence v1

## Immutable execution

- Workflow run: `33625243868`
- Evaluated source revision: `96e01abf5d4c9b0d63f94e72c727f506fddc8ab7`
- Artifact: `tracking-cloth-rank-distortion-real-v1-96e01abf5d4c9b0d63f94e72c727f506fddc8ab7-1`
- Artifact ID: `9844515256`
- Artifact SHA-256: `4cc1d9a5521474969a8c7e9fa25d56ca177b538e0b555cd050f4db17ab32b49c`
- Result SHA-256: `1272f51f06e248675fe3f6643361df23ae19ceac879adac997dc1103923b1217`
- Protocol SHA-256: `3126627f8be90a590fc65ee739b9552362200e70bc3b50a338780a571b092407`
- Inventory SHA-256: `e60eb9a121e29524088e6bc6dae09c4c453661e4c1684766bc9282e657e6b44f`
- Dataset-manifest SHA-256: `e3f4bda9da258af6717ef9e5cbbd8c0e582ad2f80efe8419ca4c853d377a05c0`
- Runtime: Python `3.12.13`, NumPy `2.2.6`
- Raw trajectory files copied to artifact: **no**

The fail-closed workflow completed successfully on the trusted
`gpuserver4090` runner and passed its post-upload result audit.

## Data admitted

The complete verified release contained 120 CSV files. The frozen parser
admitted 80 cloth recordings—48 A2 and 32 A3—and rejected the registered
non-cloth/unsupported tables. Ten recording-disjoint evaluations produced
20,843 held-out causal windows.

The fitted supplied shared factors had ranks 36–60. Every fold had numerical
exact query rank 3.

## Primary result

`D` is the posterior-normalized covariance trace contraction

`D = trace(P_full^-1 (P_full-P_reduced))`.

| retained rank | optimal generalized-eigen mean D | response-SVD mean D | mean per-fold relative D reduction | strict wins over response SVD | posterior-valid optimal folds | optimal held-out RMSE [mm] | response-SVD held-out RMSE [mm] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 920.266 | 920.266 | 0.000% | 0 / 10 | 0 / 10 | 575.368 | 575.368 |
| 1 | 343.490 | 379.390 | 8.233% | 10 / 10 | 0 / 10 | 432.486 | 297.012 |
| 2 | 126.395 | 133.794 | 5.698% | 10 / 10 | 0 / 10 | 107.237 | 91.059 |
| 3 | 1.87e-32 | 1.03e-26 | numerically zero | 0 / 10 | 10 / 10 | 13.916 | 13.916 |

The maximum observed violation of the registered generalized-eigen optimality
inequality was zero at the workflow's reporting precision.

At rank 1 the per-fold relative `D` reduction ranged from 3.91% to 15.39%; at
rank 2 it ranged from 1.18% to 12.23%. Thus the distinction from the old
Euclidean response-SVD ordering is present in every real-data-fitted fold, not
only in the controlled dense counterexample.

## Exact endpoint

At retained rank 3, the generalized-eigen factor reproduced the full registered
query posterior in every fold:

- maximum relative gain error: `1.624e-14`;
- maximum relative posterior-covariance error: `1.174e-14`;
- maximum realized posterior-mean difference: `1.465e-14 m`;
- shared-factor payload reduction: `17.0083x`.

The full/exact held-out metrics were:

- query RMSE: `13.916 mm`;
- mean Gaussian NLL: `-8.0317 nats`;
- normalized NEES: `6.0962`;
- 90% coverage: `0.6790`.

The NEES and coverage values are reported as evidence that this local Gaussian
fit is not deployment-calibrated; no calibration claim is made.

## Interpretation

The experiment supports the mechanism claim that the generalized-eigen
construction is globally optimal for the registered posterior trace-contraction
objective within each fitted `U -> U V` family. It also supports the exact
rank-3 endpoint and its payload reduction on public, recording-disjoint real
cloth trajectories.

It does **not** support deploying ranks 0–2 as Gaussian posteriors: every such
fold failed positive-definiteness of the reduced query posterior. Nor does the
training-model `D` optimum imply superior held-out point prediction. The
response-SVD baseline had lower held-out RMSE in all rank-1 folds and in seven
of ten rank-2 folds. This separates three questions that must not be conflated:

1. optimality for the registered covariance-contraction objective;
2. Bayesian validity of the resulting reduced posterior;
3. out-of-sample task performance under model mismatch.

A useful inexact compressor therefore needs a validity-preserving residual or a
validity-constrained objective; simply selecting a lower factor rank is not
sufficient on these fitted real models.

## Claim boundary

This evidence does not establish observation-likelihood preservation, full
posterior-KL optimality, arbitrary task-loss optimality, recursive exactness, a
learned 4-D provider, deployment uncertainty calibration, or a physical benefit
for BayesianPhysTwin/Causal4D.
