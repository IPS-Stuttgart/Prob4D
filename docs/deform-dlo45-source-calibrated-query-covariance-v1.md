# Source-calibrated query covariance on existing DEFORM data

This experiment addresses the one systematic limitation of the existing
DLO4/DLO5 query-observability result: accepted segment-centroid posteriors were
underdispersed despite large RMSE and NLL gains.

No new data are collected. The experiment uses only the already available
official DEFORM DLO4 and DLO5 trajectories:

- 112 training trajectories select one scalar query-covariance inflation;
- the existing 28 evaluation trajectories assess the frozen inflation;
- each complete trajectory file has equal weight;
- means, observable information factors, query-admission decisions, and exact
  fallback for rejected queries are unchanged.

The source rule selects

\[
\gamma = \max\left(1,\ \overline{\mathrm{nNEES}}_{\mathrm{source}},\
\frac{Q^{\mathrm{equal-group}}_{0.9}(d^2)}{\chi^2_{3,0.9}}\right).
\]

The target evaluation must first reproduce the immutable raw metrics from result
`1ac8cd083b39877888ea0eb2f4b9400ca89eda09436f25f5f0a6f43b154b1007`
within the frozen numerical tolerance. It then reports RMSE, Gaussian NLL,
coverage, normalized NEES, covariance width, exact fallback, and paired
trajectory-level intervals for the source-calibrated covariance.

This is transparently a post-hoc source-only calibration repair because the 28
evaluation trajectories were opened by the earlier experiment. It is not a new
confirmation cohort. No target-side tuning is allowed, and a directional or
negative result is retained without changing the source rule.

The GitHub workflow is started by a reviewed file change and runs the source
calibration before the existing evaluation reanalysis. Temporary repair helpers
are removed before the scientific execution.
