# Review boundary

This branch adds an in-memory execution representation for the existing
production causal gauge tree. It does not change provider-v1/provider-v2
artifacts, factor-bundle schema v4, estimator defaults, calibration semantics,
or frozen evidence identities.

The reviewable claims are limited to exact dense parity for admitted tree
covariances, fail-closed rejection of non-tree dependence, exact covariance and
information products, and linear retained prior storage.
