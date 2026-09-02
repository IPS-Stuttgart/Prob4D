# Source-selected dependence witness on public DEFORM trajectories

**Classification:** retrospective public-data dependence diagnostic, with secondary post-hoc trajectory-level uncertainty.

This result tests whether a physical query chosen using only source trajectories can expose information lost by destroying cross-summary dependence while preserving the prediction mean and every coordinate marginal variance.

## Information order

The workflow uses the official public `roahmlab/DEFORM` release at commit `b73b8b8ecc033caefa693fab7898741d4e6dbeff`.

- 80 training trajectories, 40 each from DLO4 and DLO5, fit one source covariance per DLO.
- A disjoint set of 32 training trajectories, 16 per DLO, selects one query against a marginal-matched diagonal covariance.
- The query witness and source model are sealed before the held-data checkout.
- All 28 official DLO4/DLO5 evaluation trajectories are then opened once.
- The held evaluation contains 532 forecast windows; the complete trajectory is the independent group.
- No target-side query reselection occurs.

The DLO4/DLO5 evaluation trajectories had been opened by earlier studies. This result is therefore permanently retrospective and is not an independent prospective confirmation.

## Registered physical-query family

Each 25-frame, eight-internal-node forecast residual is mapped linearly to 12 metric summaries:

1. terminal internal-node centroid in x/y/z;
2. horizon-average internal-node centroid in x/y/z;
3. terminal right-half minus left-half centroid in x/y/z; and
4. terminal minus initial internal-node centroid in x/y/z.

The source auditor selects a unit linear combination within this frozen 12-dimensional family. The selected query is

```text
q = [
   0.186708, -0.039849,  0.365407,
   0.398947, -0.079183,  0.690941,
   0.052714, -0.027219,  0.111100,
   0.187706, -0.040197,  0.366727
]
```

with witness ID

```text
64146eefe8cf2dd5dce7229a3e8f1c32c7f506b37d47fe999add8d42fc717543
```

The equal-source-group normalized error ratio of the marginal-matched diagonal submission is `3.1293` on this frozen direction.

## Held real-data result

The full and diagonal submissions use exactly the same residuals and prediction means. Their 12 coordinate marginal variances are also identical; only off-diagonal dependence differs.

| Covariance contract | Query nNEES | 90% coverage | Query NLL |
| --- | ---: | ---: | ---: |
| **Full source-fitted dependence** | **1.0193** | **89.85%** | **-1.6180** |
| Marginal-matched diagonal | 3.3831 | 67.48% | -1.0318 |
| Explicit 0.05-scale full-covariance control | 20.3856 | 31.95% | 6.5673 |

Relative to the marginal-matched diagonal control, preserving source-fitted dependence improves held query NLL by `0.5862` nats per forecast query and reduces absolute log-calibration error by `1.1997`.

Thus point accuracy and every univariate marginal variance are tied by construction, yet the source-selected multivariate physical query distinguishes the information contracts strongly. This is the non-artificial dependence failure required by the benchmark design: the negative control is produced by deleting cross-summary covariance, not by shrinking all variances.

## Complete-trajectory uncertainty

A secondary paired analysis resamples complete trajectories, never windows. Full dependence has lower query NLL on `21/28` trajectories; the diagonal control wins on `7/28`, with no ties.

- Mean paired NLL gain, diagonal minus full: `0.5862` nats.
- 20,000-replicate trajectory bootstrap 95% interval: `[0.3298, 0.8709]`.
- Exact two-sided paired sign-test p-value: `0.01254`.
- Mean absolute-log-calibration-error gain: `1.1997`.
- Trajectory bootstrap 95% interval for that gain: `[0.7534, 1.2162]`.

This uncertainty analysis was specified after inspecting the retrospective aggregate result. It is therefore explicitly secondary, not preregistered confirmatory inference.

## Scientific interpretation

The result supports the benchmark thesis that coordinate error and marginal variances are insufficient interfaces for downstream physical tasks. Cross-time and cross-shape dependence can determine whether an apparently calibrated 4-D predictor is calibrated for a physically meaningful trajectory query.

The result does not establish learned-provider competence, unseen-object transfer, correctness of an inferred covariance model, deployment safety, state of the art, or a prospective provider ranking. The selected covariance is source-fitted from ground-truth residuals and the target cohort was historically open. The next claim-bearing milestone remains a prospectively sealed provider × dataset study with the same source-only witness protocol.

## Provenance

- Successful workflow run: `33638409974`
- Evaluated revision: `115846d315447ab87982e9e4a145d5604366eb9f`
- Complete result artifact: `9849780004`
- Complete artifact digest: `sha256:a0f53bdde663b68a2c5493c4e8461cc2d381d0e7c415268823f567112c8d572d`
- Source witness artifact: `9849766555`
- Source artifact digest: `sha256:562c4eef00ed06745e77b47388bde0f0588dec9ebcdd4000db345d4c4fe03b2c`
- Full result SHA-256: `729498d10fe3a149da6519e5364a8cf0e951883f3ba5306e27b5d1c284ce4e1e`
- Comparison SHA-256: `1a8b54e534889636dddd973119a4e62c7b5a935ee40acf4010acbcee2d47fe38`
- Inference SHA-256: `cb2408a7611d1021cf755e762d8988ca963e5690633543b86204751d52a8695e`

The complete per-trajectory arrays and hash-bound source payload remain in the Actions artifact. This directory retains the compact claim-bound summary, inference record, and provenance.
