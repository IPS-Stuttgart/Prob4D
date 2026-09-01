# Source-calibrated DLO4/DLO5 query covariance

Decision: **source-calibrated-strong-positive**

- Calibration ID: `c1efc8d6fe2ec27d63083ee29b4a677faf1c37b26cd4239b54e4cd7a90a34fcd`
- Result ID: `383b78a1a66a02f0f54dffa11e68303cd2ded8c6cf05877a9c7c213fcd92aca4`
- No new data were collected.
- Means, factors, admission, and exact rejected-query fallback are unchanged.

| Equal-trajectory result | Fallback | Raw query-aware | Source-calibrated |
|---|---:|---:|---:|
| Centroid RMSE [mm] | 15.479331 | 0.979855 | 0.979855 |
| Centroid Gaussian NLL | -8.294309 | -15.552911 | -16.026448 |
| Centroid 90% coverage | 0.902530 | 0.741491 | 0.897825 |
| Centroid normalized NEES | 1.024574 | 2.019998 | 0.951604 |
| Centroid marginal SD [mm] | 15.172393 | 0.861870 | 1.255710 |

Paired calibrated-versus-raw NLL improvement: `0.473537` [`0.354693`, `0.594658`].

## Registered checks

- PASS — `raw_target_reproduces_immutable_result`
- PASS — `source_calibration_is_target_closed`
- PASS — `centroid_rmse_is_unchanged`
- PASS — `off_axis_rejections_remain_exact_fallback`
- PASS — `centroid_nll_improves`
- PASS — `paired_centroid_nll_lower_95_is_positive`
- PASS — `centroid_nll_still_beats_fallback`
- PASS — `centroid_coverage_moves_closer_to_90pct`
- PASS — `centroid_nees_moves_closer_to_one`
- PASS — `centroid_coverage_in_registered_band`
- PASS — `centroid_nees_in_registered_band`

This is a post-hoc source-only calibration repair on the already-opened evaluation split, not a fresh confirmation cohort.
