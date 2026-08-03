# Source-only diagnostics for common-mode failures

Overlap disagreement is useful when independently decoded windows disagree, but
it cannot by itself identify a visual backbone that is consistently wrong in all
windows. Prob4D therefore provides opt-in source-side diagnostics that can augment
reliability features without reading Bayesian-PhysTwin innovations or target
truth.

## Flow/geometry consistency

`build_flow_point_consistency_diagnostic` compares the decoded displacement

```text
point_map[t + 1] - point_map[t]
```

with `scene_flow[t]` wherever both frames and the deforming-flow row are valid.
It exports an availability indicator, a scale-free relative residual, and a
bounded direction disagreement. The function records the assumed one-step flow
semantics in metadata; producers with a different flow convention must not reuse
it silently.

## Independent-seed dispersion

`build_common_gauge_seed_dispersion_diagnostic` estimates empirical point
spread across two or more independently seeded predictions. Inputs must already
share exactly the same frame grid and an explicitly named common gauge. The
diagnostic does not perform alignment itself, because target-informed or
sample-specific alignment would change the interpretation of dispersion.

The metadata binds the common-gauge identity, immutable model-set identity,
window IDs, frame IDs, and the fact that no alignment was performed inside the
diagnostic.

## Reliability augmentation

Use `augment_source_reliability_features` to append one or more diagnostic grids
to an existing `SourceReliabilityFeatures` object. It preserves the base valid-row
set and rejects feature-name or grid mismatches. Every diagnostic must explicitly
declare:

```text
uses_truth = false
uses_downstream_physical_innovation = false
uses_association_probability = false
```

Changing the feature set requires a newly fitted, content-addressed source
reliability calibration. Existing calibration artifacts remain unchanged.

## Common-mode evaluation audit

`audit_common_mode_failures` partitions opened evaluation rows into four cells:

| Disagreement | Error | Interpretation |
| --- | --- | --- |
| low | low | nominal agreement |
| high | low | conservative or benign disagreement |
| low | high | common-mode failure candidate |
| high | high | detected failure |

The disagreement and error thresholds are explicit inputs and should be frozen
from development or calibration units. The low-disagreement/high-error rate and
severity are evaluation outputs only; they must not be used to refit reliability
on the same target outcomes.

These diagnostics do not establish that Prob4D observations improve a physical
twin. Provider competence, BayesianPhysTwin acceptance, harmful accepted updates,
and future physical prediction remain separate prospective gates.
