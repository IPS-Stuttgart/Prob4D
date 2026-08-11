# Correlation-group robust likelihood

Prob4D factors can contain many rows produced by one shared visual event: a
tracklet, a camera/frame cell, or another declared correlation group. A local
failure in that event should not create hundreds of apparently independent
outlier decisions. `prob4d.correlation_group_robust_likelihood` therefore uses
one latent contamination state for the complete group.

The module is experimental source-side infrastructure. It does not change the
provider-v2 factor contract, association probabilities, prior reliability,
BayesianPhysTwin admission, or exact physical fallback.

## Group-level likelihood

For one complete group, Prob4D already represents the covariance as

```text
C = blockdiag(D_1, ..., D_N) + U U^T.
```

The robust candidate is

```text
p(r) = (1 - rho) N(r; 0, C) + rho N(r; 0, lambda C),
```

with `0 < rho < 1` and `lambda > 1`. The Gaussian fallback is represented
exactly by `rho = 0` and `lambda = 1`.

Scaling the complete covariance gives

```text
log det(lambda C) = log det(C) + d log(lambda)
q_lambda          = q / lambda,
```

where `d = 3N` and `q = r^T C^-1 r`. The implementation reuses Prob4D's
Woodbury joint-covariance evaluator and never materializes the dense `3N x 3N`
covariance.

```python
import numpy as np

from prob4d.correlation_group_robust_likelihood import (
    CorrelationGroupContaminationSpecV1,
    CorrelationGroupResidualV1,
    evaluate_correlation_group_mixture,
)

group = CorrelationGroupResidualV1(
    group_id="camera-0:frame-12:cell-3",
    residual_xyz_m=np.array([[7.0, 0.0, 0.0]]),
    local_covariance_m2=np.eye(3)[None, ...],
    low_rank_factor_m=np.empty((1, 3, 0)),
)
spec = CorrelationGroupContaminationSpecV1(
    contamination_probability=0.1,
    inflation_factor=25.0,
)
evaluation = evaluate_correlation_group_mixture(group, spec)
```

The result reports one posterior contamination probability for the complete
group and the posterior expected inverse-scale multiplier

```text
(1 - gamma) + gamma / lambda.
```

That multiplier is a likelihood diagnostic. It is deliberately not named or
stored as association probability, prior reliability, or deployment acceptance.

## Source-only finite-grid selection

`select_source_correlation_group_mixture` evaluates a finite candidate grid on
complete independent source groups. One exact Gaussian fallback is mandatory.
Candidate and group input order are canonicalized, and every source bundle is
bound by a digest over the normalized arrays.

Selection uses nested leave-one-group-out evaluation:

1. hold out one complete source object/session/group;
2. choose a candidate from the remaining groups by equal-group mean Gaussian
   mixture NLL per dimension;
3. score that candidate on the held-out group against the Gaussian fallback;
4. repeat for every group; and
5. select the full-source candidate only when the frozen support gates pass.

The gates are explicit:

- minimum independent source-group count;
- minimum mean held-out NLL advantage per dimension;
- maximum tolerated held-out NLL harm per dimension before a group is classified
  as harmful;
- maximum harmful-group fraction; and
- minimum fraction of folds selecting the final full-source candidate.

A failed gate returns the exact Gaussian likelihood specification rather than a
partially robust configuration. Insufficient group count is retained as a valid
negative source result.

```python
from prob4d.correlation_group_robust_likelihood import (
    GAUSSIAN_GROUP_LIKELIHOOD_V1,
    select_source_correlation_group_mixture,
)

selection = select_source_correlation_group_mixture(
    source_groups,
    (
        GAUSSIAN_GROUP_LIKELIHOOD_V1,
        CorrelationGroupContaminationSpecV1(0.05, 10.0),
        CorrelationGroupContaminationSpecV1(0.10, 25.0),
    ),
    minimum_group_count=8,
    minimum_mean_heldout_advantage_per_dimension=0.0,
    maximum_heldout_nll_harm_per_dimension=0.1,
    maximum_harmful_group_fraction=0.0,
    minimum_final_candidate_fold_fraction=0.5,
)
```

The selection object retains the complete candidate-by-group proper-score
matrix, every held-out decision, source identities, thresholds, selected and
unconstrained specifications, reasons for Gaussian fallback, and a deterministic
selection identity. It does not retain or inspect target outcomes.

## Statistical unit and grouping

The group must correspond to the shared failure mechanism represented by one
latent scale. Suitable examples include a complete correlation group already
retained by provider-v2 factors or a complete independent calibration unit when
all rows share one acquisition-level contamination event.

Do not split one coherent factor across several IDs merely to create more
replicates. Do not join unrelated objects or sessions into one group merely to
obtain a stronger outlier decision. Source model selection should use the object
or acquisition session as the independent outer statistical unit.

## Boundaries

Association probability answers whether a row corresponds to the intended
physical point or identity. Prior reliability is a source-calibrated probability
that an admitted observation is nominal before seeing a downstream residual.
Posterior contamination responsibility is the likelihood's response to the
complete matched residual group. These quantities are not interchangeable.

BayesianPhysTwin remains responsible for deciding whether a physical update is
identifiable and beneficial and for returning the exact physical fallback on
rejection. Causal4D should consume only the accepted physical belief. A robust
source likelihood does not establish provider competence, fresh-object
calibration, physical-query benefit, intervention benefit, deployment safety, or
state of the art.
