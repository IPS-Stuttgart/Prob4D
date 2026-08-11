# Correlation-group robust likelihood

Prob4D factors can contain many rows produced by one coherent visual event: a
tracklet, a camera/frame cell, or another declared correlation group. One local
failure should not create hundreds of apparently independent outlier decisions.
`prob4d.correlation_group_robust_likelihood` therefore assigns one latent
contamination state to the complete correlation group.

Correlation groups and statistical units are deliberately separate:

- a `CorrelationGroupResidualV1` is one inner likelihood group that shares a
  contamination state; and
- a `SourceCorrelationGroupUnitV1` is one independent outer source object or
  acquisition session used for selection and leave-one-unit-out evaluation.

The module is experimental source-side infrastructure. It does not change the
provider-v2 factor contract, association probabilities, prior reliability,
BayesianPhysTwin admission, or exact physical fallback.

## Group-level likelihood

For one complete correlation group, Prob4D represents the covariance as

```text
C = blockdiag(D_1, ..., D_N) + U U^T.
```

The robust candidate is

```text
p(r) = (1 - rho) N(r; 0, C) + rho N(r; 0, lambda C),
```

with `0 < rho < 1` and `lambda > 1`. The Gaussian fallback is represented
exactly by `rho = 0` and `lambda = 1`. Here, `lambda` multiplies covariance, not
standard deviation.

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

correlation_group = CorrelationGroupResidualV1(
    group_id="camera-0:frame-12:cell-3",
    residual_xyz_m=np.array([[7.0, 0.0, 0.0]]),
    local_covariance_m2=np.eye(3)[None, ...],
    low_rank_factor_m=np.empty((1, 3, 0)),
)
specification = CorrelationGroupContaminationSpecV1(
    contamination_probability=0.1,
    inflation_factor=25.0,
)
evaluation = evaluate_correlation_group_mixture(
    correlation_group,
    specification,
)
```

The result reports one posterior contamination probability for the complete
correlation group and the posterior expected inverse-scale multiplier

```text
(1 - gamma) + gamma / lambda.
```

That multiplier is a likelihood diagnostic. It is deliberately not named or
stored as association probability, prior reliability, or deployment acceptance.

## Independent source units

A source object or acquisition session can contain several correlation groups.
They are nested inside one `SourceCorrelationGroupUnitV1`:

```python
from prob4d.correlation_group_robust_likelihood import (
    SourceCorrelationGroupUnitV1,
)

source_unit = SourceCorrelationGroupUnitV1(
    source_unit_id="calibration-object-01",
    correlation_groups=(
        camera_0_frame_12_cell_3,
        camera_0_frame_12_cell_4,
        camera_1_frame_12_cell_2,
    ),
)
```

Within one source unit, candidate log likelihoods are summed over its complete
correlation groups and divided by the total retained residual dimension. Across
source units, the resulting scores receive equal weight. Thus, a dense object or
session cannot dominate method selection solely because it contributes more
rows, while the proper likelihood still accounts for all declared inner groups.

Correlation-group input order is canonicalized inside each source unit. The
source-unit identity binds its ID and every inner group ID and content identity.

## Source-only finite-grid selection

`select_source_correlation_group_mixture` evaluates a finite candidate grid on
complete independent source units. One exact Gaussian fallback is mandatory.
Candidate and source-unit input order are canonicalized.

Selection uses nested leave-one-source-unit-out evaluation:

1. hold out one complete source object or acquisition session;
2. choose a candidate from the remaining units by equal-source-unit mean mixture
   NLL per residual dimension;
3. score that candidate on the held-out unit against the Gaussian fallback;
4. repeat for every source unit; and
5. retain the full-source candidate only when all frozen support gates pass.

The gates are explicit:

- minimum independent source-unit count;
- minimum mean held-out NLL advantage per dimension;
- maximum tolerated held-out NLL harm per dimension before a source unit is
  classified as harmful;
- maximum harmful-source-unit fraction; and
- minimum fraction of folds selecting the final full-source candidate.

A failed gate returns the exact Gaussian likelihood specification rather than a
partially robust configuration. Insufficient independent source units are
retained as valid negative source evidence.

```python
from prob4d.correlation_group_robust_likelihood import (
    GAUSSIAN_GROUP_LIKELIHOOD_V1,
    select_source_correlation_group_mixture,
)

selection = select_source_correlation_group_mixture(
    source_units,
    (
        GAUSSIAN_GROUP_LIKELIHOOD_V1,
        CorrelationGroupContaminationSpecV1(0.05, 10.0),
        CorrelationGroupContaminationSpecV1(0.10, 25.0),
    ),
    minimum_source_unit_count=8,
    minimum_mean_heldout_advantage_per_dimension=0.0,
    maximum_heldout_nll_harm_per_dimension=0.1,
    maximum_harmful_source_unit_fraction=0.0,
    minimum_final_candidate_fold_fraction=0.5,
)
```

The selection object retains:

- the complete candidate-by-source-unit proper-score matrix;
- source-unit content identities;
- every held-out selection and score comparison;
- all thresholds;
- the unconstrained and deployed specifications;
- explicit reasons for Gaussian fallback; and
- a deterministic selection identity that binds the claim boundary.

It does not retain or inspect target outcomes.

## Grouping rules

The inner correlation group must correspond to the shared failure mechanism
represented by one latent scale. Suitable examples include an existing
provider-v2 correlation group, one complete causal tracklet group, or one
camera/frame cell whose rows share one visual failure mode.

The outer source unit must be the independent replication unit used for the
scientific source study, normally a complete physical object or acquisition
session.

Do not:

- split one coherent factor across several correlation-group IDs merely to
  create more apparent outlier decisions;
- treat several correlation groups from one object as independent outer
  replicates;
- join unrelated objects or sessions into one source unit merely to obtain a
  stronger likelihood result; or
- select `rho`, `lambda`, or support thresholds from target outcomes.

## Boundaries

Association probability answers whether a row corresponds to the intended
physical point or identity. Prior reliability is a source-calibrated probability
that an admitted observation is nominal before seeing a downstream residual.
Posterior contamination responsibility is the likelihood's response to one
complete matched residual correlation group. These quantities are not
interchangeable.

BayesianPhysTwin remains responsible for deciding whether a physical update is
identifiable and beneficial and for returning the exact physical fallback on
rejection. Causal4D should consume only the accepted physical belief. A robust
source likelihood does not establish provider competence, fresh-object
calibration, physical-query benefit, intervention benefit, deployment safety, or
state of the art.
