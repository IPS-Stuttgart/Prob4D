# DOT rope CUT3R pooled source result v1

## Status

**Source-positive for a dependence diagnostic only.** The current CUT3R/Sim(3) stitching estimate is not promoted, and the current shared-quadratic covariance is not promoted.

This record binds the completed source-development evaluation on public DOT rope sequences `R01`--`R03`. `R04`--`R70` remained outside this evaluation route.

- provider run: `33329701704`
- provider bundle ID: `952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7`
- marker-support audit run: `33338130219`
- marker-support audit ID: `2b045cf62d1a4e385987def7c36674490130216e6bceb4883b4d86e445ec6049`
- pooled evaluation run: `33338269873`
- execution revision: `585e1b16042e9653fdcd151a90a56033bf0e47a2`
- evaluation ID: `9f900046ba2700bbc7476a4a5cd9c0da87b7bbe124c3109572eeae989fe09acd`
- workflow artifact SHA-256: `67bae296584da8fdb30e87dddbdf8b4bd5f2dfa163fe81fa98a57f8c1992c72a`
- `result.json` SHA-256: `bdb2f825659cabe08cdf4377626904a51dae510e69340d85867fb20047c4567a`

## Registered uncertainty comparison

All covariance arms use the same provider mean. Lower normalized Gaussian NLL is better.

| Method | Mean NLL / dim | Mean Mahalanobis | 95% covered | Mean predictive SD / span |
|---|---:|---:|---:|---:|
| **pointwise quadratic** | **0.766506** | **7.553** | **3/3** | 1.056488 |
| cluster bootstrap fallback | 5.207030 | 375.188 | 1/3 | 0.065642 |
| dominant rotation orbit | 10.617586 | 616.739 | 1/3 | 1.198704 |
| shared quadratic curvature | 16.895744 | 919.978 | 1/3 | 1.056488 |
| tensor Gauss-Hermite | 17.003451 | 925.211 | 1/3 | 0.998307 |
| axis spherical-radial | 21.646245 | 1148.232 | 1/3 | 0.999371 |
| scalar inflation | 28.678447 | 1480.743 | 1/3 | 1.989423 |
| local first order | 38.001730 | 1936.620 | 1/3 | 0.994941 |

The pointwise-quadratic arm improves mean NLL by `37.235223` nats/dimension relative to local first order and by `16.129238` nats/dimension relative to shared quadratic curvature on these three development sequences.

## Dependence finding

`pointwise_quadratic` and `shared_quadratic_curvature` have the same mean marginal predictive standard deviation to numerical precision (`1.056488250615809` versus `1.0564882506158095` of provider span), yet their mean joint NLL differs by `16.129238` nats/dimension.

Therefore the source failure cannot be explained by marginal variance scale alone. It localizes to the **cross-query covariance / shared-dependence structure** of the current shared-quadratic approximation. This motivates a separately frozen dependence-strength calibration that leaves every marginal variance and every provider mean unchanged.

This is not a universal superiority result: on `R02`, shared quadratic curvature and local first order score better than the pointwise arm. With only three source sequences, no population confidence interval or held-out superiority claim is made.

## Reconstruction and stitching

| Sequence | continuous / span | identity stitch / span | estimated Sim(3) stitch / span | oracle window / span |
|---|---:|---:|---:|---:|
| R01 | 0.094973 | 0.112156 | 0.102379 | 0.055046 |
| R02 | 0.066431 | 0.037155 | 0.081340 | 0.029580 |
| R03 | 0.051136 | 0.057282 | 0.058047 | 0.033581 |

Estimated Sim(3) stitching improves over identity only on `R01` and is worse than the continuous-window prediction on all three sequences. The current stitching estimate is therefore **not promoted**. The oracle restarted window is better than the continuous run on all three sequences, which leaves headroom for a better alignment/fusion method without establishing that the current estimator achieves it.

## Next registered scientific step

Before opening any new DOT outcome, fit a single shared-dependence strength on `R01`--`R03` only:

\[
\Sigma_\alpha = \Sigma_{\mathrm{pointwise}} + \alpha\left(\Sigma_{\mathrm{shared}}-\Sigma_{\mathrm{pointwise}}\right),\qquad 0\leq\alpha\leq1.
\]

Because the endpoint covariances have matching diagonals, this family preserves the source-fitted marginal variances and varies only shared dependence. `alpha=0` reproduces the pointwise arm and `alpha=1` reproduces the current shared-quadratic arm. The source selection rule and a disjoint DOT confirmation cohort must be frozen before opening that cohort.

## Claim boundary

Source-development real-image/provider evidence on `R01`--`R03` only. It supports the bounded diagnosis that joint dependence modeling materially changes proper scores even when marginal uncertainty is unchanged. It does **not** establish held-out transfer, independent calibration, improved reconstruction from the current stitching estimator, BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the art.
