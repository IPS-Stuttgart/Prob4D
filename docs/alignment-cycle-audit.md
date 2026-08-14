# Gauge alignment cycle audit

The strict Prob4D observation provider uses one causal spanning tree to propagate
the joint window-gauge covariance. This avoids multiplying dense overlap edges
as though they were independent. The remaining non-tree alignments are still
valuable as diagnostics.

`prob4d.alignment_cycles` compares a direct edge

```text
A <- C
```

with every available two-edge path

```text
A <- B <- C.
```

```python
from prob4d.alignment_cycles import audit_alignment_cycles

audit = audit_alignment_cycles(
    alignments,
    representative_radius=0.25,
    maximum_representative_displacement=0.01,
)
print(audit.to_dict())
```

## Reported quantities

For each directed triangle, Prob4D composes `A <- B` and `B <- C`, compares the
result with `A <- C`, and reports:

- absolute log-scale disagreement;
- shortest-axis-angle rotation disagreement in radians;
- translation disagreement in the moving window's coordinates;
- root-mean-square displacement of the origin and the six signed coordinate-axis
  points at the declared representative radius;
- direct and path-edge residual RMS diagnostics;
- the minimum correspondence count across the three alignments.

The representative-point displacement combines scale, rotation, and translation
in the observation geometry rather than comparing mixed-unit `Sim(3)` parameters
with one arbitrary norm. Its units are those of the local alignment coordinates.

An optional displacement threshold creates a deterministic pass/fail result. The
threshold is a registered engineering diagnostic, not a probability statement.

## Statistical boundary

Overlap alignments share frames, pixels, a model backbone, and often upstream
window uncertainty. Their cross-covariance is not available. The audit therefore
does **not** whiten the cycle residual by the three marginal edge covariances and
does not label its result NIS, NEES, chi-square, or a calibrated p-value.

A large cycle residual can indicate:

- a bad direct or path alignment;
- local deformation that violates the rigid/similarity overlap approximation;
- shared or spatially varying model bias;
- insufficient geometry or occlusion;
- a threshold or correspondence-selection problem.

It does not by itself identify which edge is wrong.

## Intended use

The audit is additive and does not change provider-v1/v2 gauge estimation. It can
be used to:

1. stratify held-out gauge-calibration results by cycle consistency;
2. reduce source-side prior reliability for visibly inconsistent alignment
   groups after an independently frozen calibration rule is established;
3. compare alternative causal spanning-tree edge scores;
4. diagnose whether a prospective Prob4D-to-Bayesian-PhysTwin failure originates
   in the observation gauge graph.

Using cycle residuals to remove target examples, tune a threshold after opening
target outcomes, or multiply redundant edges as independent likelihood factors
would violate the intended evidence boundary.
