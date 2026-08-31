# DEFORM DLO4/DLO5 query-observability robustness audit

Status: **completed-posthoc-group-robustness-audit**

This post-hoc audit reanalyzes the immutable 28-file held-out result at the declared trajectory-file level. It changes no method, threshold, support, prior, covariance, or evaluation outcome.

## Positive mechanism robustness

| Endpoint | Mean | Worst file | 20% trimmed mean | Family-blocked 95% CI | Wins | Exact sign p |
|---|---:|---:|---:|---:|---:|---:|
| Centroid RMSE improvement [mm] | 14.499 | 12.977 | 14.503 | [14.242, 14.761] | 28/28 | 3.725e-09 |
| Centroid Gaussian-NLL improvement | 7.259 | 6.167 | 7.283 | [7.047, 7.473] | 28/28 | 3.725e-09 |

Leave-one-file-out RMSE means remain in `[14.450, 14.556]` mm. DLO4 and DLO5 each retain `14/14` RMSE and NLL wins.

## Negative controls and calibration boundary

| Finding | Result |
|---|---:|
| Minimum query-aware off-axis exact fallback | 1.000 |
| Unconditional off-axis use harmful | 28/28 files |
| Invalid full-rank completion harmful | 28/28 files |
| Accepted centroid coverage below 90% | 28/28 files |
| Accepted centroid normalized NEES above 1 | 28/28 files |

The positive mean/proper-score effect is not driven by one trajectory or one DLO family. The covariance-calibration failure is equally systematic and remains a limitation; the descriptive scale mismatch is not an evaluation-side repair.

## Claim boundary

- This is a post-hoc robustness audit of one immutable held-out DLO4/DLO5 result, not a new independent confirmation cohort.
- All inference remains at the official trajectory-file level; frames and cases are nested observations.
- The audit changes no query gate, support, covariance, prior, comparator, threshold, seed, provider, or evaluation outcome.
- The descriptive scale mismatch is not an evaluation-side recalibration and is not promoted as a method.
- The audit does not establish learned-provider competence, arbitrary DLO transfer, BayesianPhysTwin or Causal4D benefit, deployment calibration, safety, or state of the art.
