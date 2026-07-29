# Source-only observation reliability calibration

Prob4D keeps four different concepts separate:

- association probability: support for a named entity or track identity;
- prior reliability: source-side evidence that an observation row is nominal;
- posterior nominal responsibility: a downstream robust-likelihood quantity;
- composite weight: a cap for correlated or repeated evidence.

`prob4d.source_reliability` adds an opt-in calibration path for the second
quantity. It does not alter the current provider-v1/v2 export defaults.

## Source-only features

```python
from prob4d import build_source_reliability_features

features = build_source_reliability_features(
    prediction_window,
    structured_covariance,
    overlap_disagreement,
)
```

The built-in feature contract contains:

1. whether overlap evidence is present;
2. log normalized overlap disagreement;
3. proximity to a temporal window edge;
4. log total predictive variance relative to the window median;
5. whether valid scene flow is present;
6. log scene-flow magnitude relative to the window median;
7. local validity-mask deficit in a 3 by 3 neighborhood.

These values use only the prediction source, its declared covariance, and overlap
consistency. They do not use target truth at inference time, a physical-twin
innovation, association probability, or a downstream posterior decision.

The explicit `has_overlap` feature lets calibration learn a finite reliability
for rows without duplicate-window evidence. Absence of disagreement is therefore
not hard-coded as reliability one.

## Equal-group logistic calibration

```python
from prob4d import fit_group_balanced_source_reliability

model = fit_group_balanced_source_reliability(
    calibration_features,
    source_nominal_labels,
    sequence_ids,
    feature_names=feature_names,
    label_definition="3-D source error below the frozen 20 mm threshold",
    group_definition="physical sequence",
    ridge=1e-3,
)
```

Every declared group receives equal total optimization weight. A sequence with
many valid pixels cannot dominate only because it contributes more rows. Within a
group, rows retain equal weight. Inputs are canonically ordered before fitting,
so row permutation does not change model bytes or the artifact ID.

The fitted artifact records:

- feature names, centering, scaling, intercept, and coefficients;
- probability clipping limits;
- exact label and group definitions;
- canonical calibration group IDs;
- group-balanced nominal fraction;
- weighted log loss and Brier score;
- optimization convergence and ridge strength;
- finite JSON metadata and a SHA-256 content address.

Save and load with:

```python
from prob4d import (
    load_source_reliability_model,
    save_source_reliability_model,
)

save_source_reliability_model(model, "source-reliability.json")
loaded = load_source_reliability_model("source-reliability.json")
```

## Label boundary

The label must be defined before opening target outcomes and generated from an
independent source/calibration unit. Examples include a predeclared 3-D error
threshold, an independently registered depth check, or a held-out source tracker
competence label.

Invalid labels include:

- whether Bayesian-PhysTwin accepted the row;
- the physical innovation magnitude;
- a future target metric;
- the robust likelihood's posterior nominal responsibility;
- the track association probability itself.

Using any of those would leak downstream evidence back into the prior.

## Promotion boundary

The model is additive and is not yet a claim-bearing provider-v2 default. Before
promotion:

1. freeze feature, label, group, clipping, and ridge definitions;
2. fit on source/calibration objects or sessions disjoint from targets;
3. bind prediction and calibration artifact digests in metadata;
4. compare pooled and equal-group calibration, worst-group coverage, selective
   risk, and harmful-update acceptance under the same Bayesian-PhysTwin guard;
5. retain the previous provider semantics for frozen reproduction;
6. require a new content-addressed calibration artifact rather than silently
   reusing the current point-uncertainty calibration.

A calibrated source nominality probability is still not proof that assimilating
the observation improves a physical twin. The downstream baseline-relative guard
and exact fallback remain necessary.
