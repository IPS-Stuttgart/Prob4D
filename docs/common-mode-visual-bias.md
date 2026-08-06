# Coherent visual bias and explicit nuisance factors

Overlapping windows can agree while every window is wrong in nearly the same
way. Independent diffusion seeds can fail similarly because they retain the same
model family, input video, and monocular geometry assumptions. Overlap or seed
dispersion therefore cannot by itself certify a visual observation.

Prob4D now provides two additive controls:

1. a calibration-separated synthetic common-mode stress benchmark; and
2. a content-addressed explicit visual-bias nuisance sidecar.

Neither changes the frozen provider-v1/provider-v2 observation contracts or the
schema-v4 observation-factor bundle.

## Observation model

For observation row `i`, the intended downstream model is

```text
z_i = h_i(x) + J_i^g delta_g + B_i b_s + epsilon_i,
```

where:

- `x` is the physical state owned by BayesianPhysTwin;
- `delta_g` is the existing ordered joint `Sim(3)` gauge nuisance;
- `b_s` is a low-dimensional visual-bias state scoped by sequence, view, model,
  or another declared source domain;
- `B_i` is the row-local bias Jacobian; and
- `epsilon_i` has the existing conditional local point covariance.

The downstream estimator must not add a gauge-marginal or bias-marginal point
covariance as another independent term while also retaining the corresponding
explicit latent state.

## Visual-bias sidecar

`VisualBiasNuisanceV1` binds:

- the exact observation artifact ID;
- the exact ordered observation-identity digest;
- ordered bias-scope IDs and basis names;
- one bias-scope index and one `3 x R` Jacobian per observation row;
- the complete joint covariance over all bias scopes and basis coefficients;
- gauge-orthogonalization semantics and the measured residual projection; and
- finite JSON metadata.

The JSON manifest and NPZ payload are checksum-bound and content-addressed.
Array names, dtypes, shapes, and raw bytes are independently described. Loading
uses `allow_pickle=False`, rejects unknown members and path/symlink escapes, and
returns immutable arrays.

```python
import numpy as np

from prob4d.visual_bias import (
    VisualBiasNuisanceV1,
    write_visual_bias_nuisance,
)

sidecar = VisualBiasNuisanceV1(
    observation_artifact_id=observation_artifact_id,
    observation_identity_sha256=ordered_row_identity_sha256,
    bias_ids=("camera-0",),
    basis_names=("ray-depth", "depth-bowl"),
    row_bias_indices=np.zeros(observation_count, dtype=np.int64),
    bias_jacobian=bias_jacobian.astype(np.float64),
    joint_bias_covariance=bias_prior.astype(np.float64),
    orthogonalization_semantics=(
        "conditional-whitened-global-gauge-projection-v1"
    ),
    maximum_gauge_projection=maximum_projection,
    gauge_projection_tolerance=1e-8,
    metadata={
        "calibration_object_ids": calibration_object_ids,
        "uses_truth": False,
        "uses_downstream_physical_innovation": False,
    },
)
write_visual_bias_nuisance(sidecar, "outputs/case-a/visual-bias.json")
```

Validate persisted bytes through the grouped command:

```bash
prob4d observation visual-bias validate \
  outputs/case-a/visual-bias.json
```

`low_rank_factor()` returns rowwise factors `U_i` whose collective dense product
preserves the complete cross-row covariance induced by the joint bias prior.
This permits scalable Woodbury-style downstream use without discarding
cross-scope dependence.

## Avoiding a duplicate gauge mode

A candidate bias basis can contain translation, rotation, or scale directions
already represented by the explicit `Sim(3)` gauges. Use
`orthogonalize_visual_bias_basis` with the complete block-sparse global gauge
design and conditional point covariance. It whitens both designs by the local
conditional covariance, removes the global gauge span, maps the residual basis
back into observation coordinates, and reports the maximum normalized gauge
projection before and after removal.

The operation must use the complete global gauge design, not only a row-local
`3 x 7` Jacobian. A single 3-D row may be locally full rank even when the stacked
sequence leaves useful coherent modes outside the global gauge span.

## Controlled stress benchmark

Run the deterministic calibration-separated mechanism test with:

```bash
prob4d diagnostic common-mode-stress \
  --output outputs/common-mode-stress/report.json
```

The benchmark uses three disjoint simulated panels:

1. clean calibration fixes disagreement and error thresholds;
2. bias calibration estimates the shared-bias variance; and
3. the target panel compares a naive independent visual update against the same
   update with the explicit shared-bias variance marginalized.

The report includes the low-disagreement/high-error rate, raw and deployed RMSE,
90% complete-policy and accepted-update coverage, interval width, NLL, accepted
and rejected counts, harmful accepted updates, worst-group regression, and exact
fallback reproduction. The default registered gates require that the coherent
failure is exposed, coverage improves, deployed RMSE is no worse than the naive
arm and beats the physical fallback, harmful admissions do not increase, and
every rejection reproduces the exact physical fallback.

This benchmark is controlled synthetic mechanism evidence. It must not be used
to claim real MotionCrafter competence or to tune the already opened 19-case
PhysTwin diagnostic cohort.

## BayesianPhysTwin and Causal4D boundary

Prob4D owns the source-calibrated bias basis, prior, source lineage, and explicit
factor sidecar. BayesianPhysTwin owns joint inference or marginalization with the
physical state, query-specific observability, regret guard, interval calibration,
and exact fallback. An independent contact, depth, LiDAR, or tactile factor
should remain separate and have no visual-bias Jacobian so it can genuinely
break a visual ambiguity.

Causal4D receives only the selected BayesianPhysTwin belief. A valid visual-bias
sidecar is infrastructure evidence, not evidence that a physical update is
identifiable, accepted, calibrated, or beneficial for an intervention query.
