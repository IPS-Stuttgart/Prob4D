# Source-calibrated query covariance on existing DEFORM data

Status: **source-calibrated strong positive on the existing evaluation split**.

This experiment addresses the one systematic limitation of the existing
DLO4/DLO5 query-observability result: accepted segment-centroid posteriors were
underdispersed despite large RMSE and NLL gains.

No new data were collected. The experiment used only the already available
official DEFORM DLO4 and DLO5 trajectories:

- 112 training trajectories selected one scalar query-covariance inflation;
- the existing 28 evaluation trajectories assessed the frozen inflation;
- each complete trajectory file had equal weight;
- means, observable information factors, query-admission decisions, and exact
  fallback for rejected queries were unchanged.

The source rule selected

\[
\gamma = \max\left(1,\ \overline{\mathrm{nNEES}}_{\mathrm{source}},\
\frac{Q^{\mathrm{equal-group}}_{0.9}(d^2)}{\chi^2_{3,0.9}}\right)
= 2.1227300133784226.
\]

On the 112 training trajectories, this moved equal-trajectory 90% coverage from
`72.68%` to `90.00%` and normalized NEES from `2.043` to `0.962`.

## Existing 28-file evaluation result

The implementation first reproduced the immutable raw result
`1ac8cd083b39877888ea0eb2f4b9400ca89eda09436f25f5f0a6f43b154b1007`
within absolute tolerance `1e-10`. It then changed only the accepted centroid
covariance.

| Equal-trajectory endpoint | Physical fallback | Raw query-aware | Source-calibrated |
|---|---:|---:|---:|
| Centroid RMSE [mm] | 15.479331 | 0.979855 | **0.979855** |
| Centroid Gaussian NLL | -8.294309 | -15.552911 | **-16.026448** |
| Centroid 90% coverage | 90.25% | 74.15% | **89.78%** |
| Centroid normalized NEES | 1.025 | 2.020 | **0.952** |
| Centroid marginal SD [mm] | 15.172393 | 0.861870 | 1.255710 |

The paired calibrated-versus-raw NLL improvement was `0.473537`, with a
trajectory-bootstrap 95% interval of `[0.354693, 0.594658]`. RMSE, accepted
means, and the `100%` centroid admission rate were exactly unchanged.
Gauge-sensitive off-axis queries remained rejected and reproduced the complete
physical fallback exactly.

All 11 registered checks passed. The source calibration ID is
`c1efc8d6fe2ec27d63083ee29b4a677faf1c37b26cd4239b54e4cd7a90a34fcd`;
the executed result ID is
`383b78a1a66a02f0f54dffa11e68303cd2ded8c6cf05877a9c7c213fcd92aca4`.
The retained artifact is bound to workflow run `33532006652`, job
`99937185679`, and artifact `9810128613`.

## Interpretation and boundary

The result repairs the known covariance-underdispersion limitation without
sacrificing the large query-mean gain or the exact safety fallback. It supports
source-calibrated uncertainty for this controlled-gauge DLO4/DLO5 experiment.

This is transparently a post-hoc source-only calibration repair because the 28
evaluation trajectories had already been opened by the earlier experiment. It
is not a fresh confirmation cohort. No target-side tuning was performed, and it
does not establish learned-provider, BayesianPhysTwin, Causal4D,
deployment-safety, or state-of-the-art claims.
