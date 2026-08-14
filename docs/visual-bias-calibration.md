# Source-only calibration of coherent visual bias

A visual provider can be internally consistent and still be coherently wrong.
Overlap agreement, repeated diffusion seeds, or two constructions from one model
run therefore cannot by themselves certify a physical-state observation. Prob4D
represents this failure mode with an explicit low-dimensional visual-bias latent,
separate from local point covariance and from the joint `Sim(3)` gauge nuisance.

`VisualBiasCalibrationV1` supplies the missing source-side calibration step. It
estimates a zero-mean prior over predeclared bias modes from complete calibration
objects or acquisition sessions and records rank zero as a valid decision when
those groups do not support a coherent-bias model.

## Statistical unit and information boundary

Each `VisualBiasCalibrationGroup` contains only:

- one source/calibration group ID;
- source residual rows with shape `(N, 3)`;
- the matching candidate bias Jacobian with shape `(N, 3, R)`;
- positive-definite conditional point covariance blocks; and
- optionally, the complete gauge design admitted for those rows.

A group should be a complete physical object or acquisition session. Frames,
pixels, points, views, and tracks are not independent calibration groups. Every
group receives equal total information weight, so repeating all rows of one group
does not change rank selection or the fitted covariance.

The artifact explicitly records whether source truth was used. Target outcomes
and downstream BayesianPhysTwin physical innovations are forbidden. The provider
manifest identity, exact calibration-source inventory identity, grouping rule,
residual definition, group IDs, and all numerical choices are content-bearing.

## Fit and rank selection

For group `g`, Prob4D uses

```text
r_g = B_g b_g + e_g,
e_g ~ N(0, D_g),
b_g ~ N(0, Sigma_b).
```

If a gauge design is supplied, every candidate basis is first projected out of
the complete conditional-whitened gauge span. This prevents translation,
rotation, or scale modes already represented by the explicit gauge state from
being counted again as visual bias.

For every prefix rank `0..R`, the fitter:

1. computes a group-normalized generalized least-squares coefficient and its
   finite-information covariance for each calibration group;
2. estimates `Sigma_b` from equal-group, noise-corrected second moments;
3. shrinks the estimate toward a nonnegative diagonal target and projects it to
   the positive-semidefinite cone; and
4. scores each held-out group under `D_g + B_g Sigma_b B_g'` using a
   leave-one-group-out Gaussian log score.

The group normalization makes the score invariant to exact row duplication. The
rank-independent Gaussian constant is omitted because the score is used only to
compare candidate ranks within the same held-out group. Ties select the smaller
rank. A nonzero rank is promoted only when it beats rank zero by the frozen
`minimum_nll_improvement`; otherwise the result remains rank zero.

## Python usage

```python
import numpy as np

from prob4d.visual_bias_calibration import (
    VisualBiasCalibrationGroup,
    build_visual_bias_nuisance_from_calibration,
    fit_visual_bias_calibration,
    write_visual_bias_calibration,
)

groups = tuple(
    VisualBiasCalibrationGroup(
        group_id=object_id,
        residual=residual_xyz.astype(np.float64),
        bias_jacobian=candidate_basis.astype(np.float64),
        conditional_covariance=conditional_covariance.astype(np.float64),
        gauge_design=complete_gauge_design.astype(np.float64),
        metadata={"episode_ids": episode_ids},
    )
    for object_id, residual_xyz, candidate_basis,
        conditional_covariance, complete_gauge_design, episode_ids
    in calibration_inputs
)

calibration = fit_visual_bias_calibration(
    groups,
    basis_names=("ray-depth", "depth-bowl", "slow-temporal-drift"),
    provider_manifest_id=provider_manifest_id,
    calibration_source_id=calibration_source_inventory_id,
    group_definition="complete-physical-object-v1",
    residual_definition="source-metric-minus-provider-point-v1",
    uses_truth=True,
    covariance_shrinkage=0.25,
    minimum_nll_improvement=1e-4,
    metadata={
        "provider_family": "example-provider",
        "uses_target_outcomes": False,
    },
)
write_visual_bias_calibration(
    calibration,
    "outputs/calibration/visual-bias.json",
)
```

The fit result contains all group-by-rank coefficients, coefficient covariances,
leave-one-group-out scores, row counts, and gauge-projection diagnostics. An
independent consumer can therefore replay the rank decision without rerunning
the dense provider.

Validate an installed artifact with:

```bash
prob4d diagnostic visual-bias-calibration \
  outputs/calibration/visual-bias.json
```

## Building an observation sidecar

A promoted calibration can instantiate one provider or camera scope for a new
causal observation:

```python
sidecar = build_visual_bias_nuisance_from_calibration(
    calibration,
    observation_artifact_id=observation_artifact_id,
    observation_identity_sha256=ordered_row_identity_sha256,
    bias_id="camera-0",
    bias_jacobian=candidate_basis_for_observation,
    conditional_covariance=conditional_covariance,
    gauge_design=complete_gauge_design,
    metadata={"case_id": case_id},
)
```

The builder keeps only the selected prefix basis, repeats the gauge projection
under the actual observation design, binds the exact calibration artifact ID,
and creates the existing `VisualBiasNuisanceV1`. Rank-zero calibration fails
closed instead of silently inventing a nuisance prior.

Version 1 deliberately supports one calibrated bias scope per sidecar. Several
views may use separate calibrated sidecars, but cross-view independence must not
be assumed unless a later calibration artifact estimates their complete joint
covariance.

## Recursive BayesianPhysTwin use

`VisualBiasNuisanceStreamV1` can bind several causal observation updates to the
same source-calibrated bias prior. BayesianPhysTwin must represent that latent
once across time. Instantiating a new independent copy at every update removes
the off-diagonal covariance and overcounts repeated visual evidence.

The calibration does not define physical discrepancy dynamics, accept a state
update, or authorize Causal4D execution. BayesianPhysTwin still owns
query-specific observability, the baseline-relative guard, interval calibration,
and exact fallback. Causal4D consumes only the accepted or fallback physical
belief. Independent tactile, depth, LiDAR, or force evidence should remain a
separate factor with no visual-bias Jacobian so it can genuinely break the camera
ambiguity.

## Claim boundary

A valid artifact proves that a declared low-dimensional, zero-mean visual-bias
prior was selected and estimated from the named source groups under the recorded
rules. It does not prove that the basis is complete, the provider is competent,
target coverage is calibrated, a physical update is beneficial, an intervention
conclusion is valid, or the method is state of the art. Those remain separate,
fresh-object held-out gates.
