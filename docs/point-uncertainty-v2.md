# Point uncertainty calibration v2

`PointUncertaintyCalibrationV2` is an **experimental, source/calibration-only**
conditional point-covariance model. It is deliberately downstream of
`SourceCovarianceLocalizationV1` and refuses to fit unless that artifact classifies
the remaining source-side failure as `point-covariance-localized`.

This preserves the provider stop rule:

1. support failure redirects the provider;
2. observation-mean failure redirects the provider;
3. identity/association failure redirects the provider;
4. gauge/dependence failure redirects the gauge or dependence model; and
5. only a conditional point-covariance failure authorizes this model.

The model is not enabled in provider v2 exports by this implementation.

## Motivation

The production point model uses one variance along the viewing ray and one
shared lateral variance. That is intentionally compact, but a localized
conditional failure may indicate that the two lateral directions have
systematically different uncertainty.

Version 2 uses the orthonormal basis

\[
(r,t_1,t_2),
\]

where:

- `r` is the viewing ray;
- `t1` is the prediction-only reference direction projected into the tangent
  plane; and
- `t2 = r x t1`.

If the reference direction is degenerate, a deterministic camera-coordinate
axis is projected into the tangent plane. Residuals or target outcomes must not
be used to choose the reference direction.

The conditional covariance is

\[
\Sigma_{\mathrm{point}} =
\sigma_r^2 rr^\top +
\sigma_{t_1}^2 t_1t_1^\top +
\sigma_{t_2}^2 t_2t_2^\top.
\]

The **shared Sim(3) gauge covariance remains separate**. Version 2 must not absorb
that low-rank nuisance into local point variances.

## Heteroscedastic variance model

For each of the three local axes, the log variance is a linear function of frozen
source-side features:

\[
\log \sigma_k^2 = \beta_{k,0} + x^\top \beta_k.
\]

Features are standardized using equal-group weighted source statistics. Candidate
features should be computable without residual or target access, for example:

- predicted depth or inverse depth;
- overlap disagreement computed out of fold;
- temporal position inside the source window;
- predicted flow magnitude and direction;
- prediction-only occlusion or visibility scores;
- association entropy or retained-track support; and
- geometry-conditioning diagnostics.

Do not use target residuals, future frames beyond the causal cutoff, manually
selected error regions, or downstream physical innovations as features.

## Fit objective

Rows inside one physical object/session are not treated as independent
calibration units. Each independent group receives equal total weight, and rows
inside a group divide that weight equally.

For projected residual `e` and predicted variance `v`, the data term is the
Gaussian variance negative log likelihood

\[
\frac{1}{2}(\log v + e^2/v).
\]

This is optimized directly with damped Newton steps. It avoids the systematic
variance bias that results from least-squares regression on `log(e^2)`.

A ridge penalty regularizes feature coefficients. A separate coupling penalty

\[
\lambda_{\mathrm{lat}}\|\beta_{t_1}-\beta_{t_2}\|^2
\]

shrinks the two lateral models toward the existing isotropic-lateral model.
Anisotropy therefore has to be supported by the source/calibration evidence
rather than appearing merely because two unconstrained regressions were fitted.

The artifact records the in-sample normalized residual energies only as a fit
diagnostic. They are **not** calibration evidence and cannot replace a disjoint
source-validation or fresh-target evaluation.

## Authorization

The Python API requires an actual `SourceCovarianceLocalizationV1` instance and
checks both:

```text
classification == "point-covariance-localized"
authorize_point_uncertainty_development == true
```

Any other source classification raises an error before training rows are used.

The training group roster must exactly equal the localization's independent group
roster. This prevents a richer covariance model from silently being fitted on a
different or partial source cohort.

## Training data

The module can be used from Python or with the research-only module CLI.

The CLI accepts one NPZ with exactly these arrays:

- `residual_xyz`: shape `(N, 3)`, source/calibration point residuals;
- `ray_directions`: shape `(N, 3)`;
- `tangent_reference`: shape `(N, 3)`, prediction-only reference vectors;
- `features`: shape `(N, F)`, prediction-only covariates;
- `group_ids`: shape `(N,)`, physical object/session IDs; and
- `feature_names`: shape `(F,)`.

The complete NPZ bytes are SHA-256 bound into the output artifact.

Example:

```bash
python -m prob4d.point_uncertainty_v2 fit \
  --localization outputs/source-covariance-localization.json \
  --training outputs/point-uncertainty-source-v2.npz \
  --policy protocols/point-uncertainty-v2-policy.json \
  --output outputs/point-uncertainty-v2.json

python -m prob4d.point_uncertainty_v2 verify \
  --artifact outputs/point-uncertainty-v2.json
```

The optional policy JSON freezes:

- ridge strength;
- lateral coupling strength;
- minimum independent-group count;
- minimum rows per group;
- variance floor;
- log-variance clipping bounds;
- Newton tolerance; and
- maximum iteration count.

## Consumer API

```python
from prob4d.point_uncertainty_v2 import (
    load_point_uncertainty_calibration_v2,
)

calibration = load_point_uncertainty_calibration_v2(
    "outputs/point-uncertainty-v2.json"
)

variances = calibration.predict_variances(features)
conditional_covariance = calibration.covariance_matrices(
    ray_directions,
    tangent_reference,
    features,
)
```

`conditional_covariance` contains only the local conditional point covariance.
A downstream gauge-aware consumer must continue to carry the shared gauge
Jacobian and gauge prior separately.

## Promotion requirements

Implementing and successfully fitting this model does not justify replacing the
production uncertainty model. Promotion requires a separately frozen experiment
that demonstrates, on disjoint source-validation or fresh object/session units:

- improved proper score such as Gaussian NLL;
- improved or non-worse object/session-level coverage;
- acceptable interval/covariance width;
- no degradation of point-mean or identity competence;
- no evidence that the remaining error is actually shared gauge/common-mode
  error; and
- downstream BayesianPhysTwin benefit under the same regret guard and exact
  fallback rule.

A valid negative result leaves the production point model unchanged.
