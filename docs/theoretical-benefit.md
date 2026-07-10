# What Uncertainty Can Improve

Prob4D's uncertainty has two different theoretical roles. Their assumptions and
guarantees should not be mixed.

## Expected Error Under Independence

Consider two unbiased scalar estimates with independent errors and variances
`v1` and `v2`. Precision fusion has variance

```text
v_precision = v1 * v2 / (v1 + v2).
```

Uniform averaging has variance `(v1 + v2) / 4`, and therefore

```text
v_uniform - v_precision = (v1 - v2)^2 / (4 * (v1 + v2)) >= 0.
```

Precision weighting is strictly better when the variances differ. The matrix
generalization is the best linear unbiased estimator:

```text
P = (P1^-1 + P2^-1)^-1,
x = P * (P1^-1 * x1 + P2^-1 * x2).
```

Its covariance is no larger than either input covariance in the positive
semidefinite order. This is an expected squared-error result, not a guarantee
for every realized prediction.

## Consistency Under Unknown Correlation

MotionCrafter overlap estimates are not independent: they share a backbone and
most input frames. Covariance intersection uses

```text
P_CI^-1 = w * P1^-1 + (1 - w) * P2^-1,  0 <= w <= 1.
```

If each input covariance is a consistent marginal bound, CI preserves
consistency without knowing their cross-correlation. In contrast, independent
precision fusion can become overconfident. CI's benefit is therefore measured
by coverage shortfall and worst-case coverage, not only by symmetric distance
from nominal coverage. CI does not guarantee a more accurate fused mean.

## Risk-Aware Decisions

A calibrated covariance supports prediction ellipsoids and rejection at fixed
coverage. At a fixed retention rate, selecting samples with the lowest true
conditional expected loss minimizes retained risk. A predictive relative-risk
score approximates this decision rule:

```text
relative uncertainty = trace(P) / max(||x||, epsilon)^2.
```

This score must be evaluated on held-out scene families. Positive rank
correlation or selective gain is empirical evidence; it is not implied by the
fusion equations.

## Claim Boundary

There is no unconditional theorem that adding estimated uncertainty improves
MotionCrafter's point prediction. The expected-MSE result requires correct
covariances and independence. The CI consistency result requires conservative
input covariances. Shared model bias can be both wrong and low-disagreement, so
overlap-derived uncertainty may fail to rank it. Prob4D must report point
accuracy, two-sided calibration, one-sided coverage shortfall, and selective
risk separately.
