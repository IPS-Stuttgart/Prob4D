# Source-only calibration-transport certificate

`prob4d.calibration_transport` is an additive experimental gate for checking
whether a target prefix lies inside the source-feature regime on which a Prob4D
reliability or covariance treatment was calibrated. It is evaluated before a
BayesianPhysTwin likelihood is admitted.

The certificate does **not** use target truth, a BayesianPhysTwin innovation,
posterior responsibility, accepted-update harm, or a Causal4D outcome. It does
not prove calibration transfer. It only detects unsupported extrapolation in a
frozen source-only feature representation and gives the downstream consumer an
explicit reason to use its exact physical fallback.

## Statistical unit and information order

Each `CalibrationTransportUnitV1` must be one complete independent calibration
object or acquisition session. A target unit may instead be a predeclared
camera, spatial cell, or horizon group inside the causal target prefix, provided
those groups are not treated as independent source replicates during fitting.

The required order is:

1. freeze the source-only feature definition and compute its feature-contract ID;
2. build complete source units without target access;
3. freeze the transport policy;
4. fit and persist the source model;
5. compute the same features on the target prefix without target residuals;
6. evaluate and persist the transport evidence; and
7. admit the visual likelihood only when both this certificate and the separate
   BayesianPhysTwin guard allow it.

An already-open target cohort cannot be used to change the quantiles, scale
floors, neighbor count, miscoverage rate, or tolerated unsupported mass.

## Method

For every source unit and feature, the model records the frozen empirical
quantiles declared in `CalibrationTransportPolicyV1`. Quantiles are flattened in
feature-major order to form one distributional embedding per complete unit.

Across source-unit embeddings, every coordinate receives a robust diagonal
scale. The scale is the maximum of:

- normal-consistent median absolute deviation;
- normal-consistent interquartile range; and
- the declared absolute plus relative scale floor.

The distance between two unit embeddings is the root mean square standardized
coordinate difference. A unit's source nonconformity score is the mean distance
to its `k` nearest *other* source units. Thus no source unit can select itself as
support.

For `M` source units and declared miscoverage rate `alpha`, the frozen threshold
uses rank

```text
min(M, ceil((M + 1) * (1 - alpha)))
```

of the sorted leave-one-unit-out source scores. The robust scale is source-fitted
once; each nonconformity score excludes the scored unit from its neighbor set.
This is a conservative source-only support diagnostic, not a claim of an exact
distribution-free conformal guarantee.

A target group is supported when its score does not exceed the frozen threshold.
The evidence also reports a source-support p-value, nearest source units,
feature-wise standardized distance, and feature-wise excursion beyond the source
embedding range.

The final decision is conjunctive over two frozen limits:

- maximum unsupported target-group fraction; and
- maximum unsupported target-row fraction.

This keeps one tiny unsupported group from necessarily vetoing a large target
prefix when the protocol explicitly permits it, while preventing many tiny
unsupported groups from being hidden by row weighting.

## Example

```python
from prob4d.calibration_transport import (
    CalibrationTransportPolicyV1,
    CalibrationTransportUnitV1,
    calibration_transport_feature_contract_id,
    evaluate_calibration_transport,
    fit_calibration_transport_model,
    save_calibration_transport_evidence,
    save_calibration_transport_model,
)
from prob4d.source_reliability import build_source_reliability_features

feature_names = (
    "has_overlap",
    "log1p_normalized_overlap_disagreement",
    "temporal_edge_proximity",
    "log_relative_total_variance",
    "has_scene_flow",
    "log1p_relative_scene_flow",
    "local_validity_deficit",
)
feature_contract_id = calibration_transport_feature_contract_id(
    feature_names,
    semantics="prob4d-source-only-reliability-features-v1",
    configuration={
        "window_geometry_id": "<frozen SHA-256>",
        "reliability_feature_builder_revision": "<exact revision>",
    },
)

source_units = []
for source_object in frozen_source_objects:
    features = build_source_reliability_features(
        source_object.window,
        source_object.covariance,
        source_object.disagreement,
    )
    source_units.append(
        CalibrationTransportUnitV1.from_feature_grid(
            source_object.object_id,
            features,
            feature_contract_id=feature_contract_id,
            metadata={
                "source_artifact_id": source_object.artifact_id,
                "statistical_unit": "complete-object",
            },
        )
    )

policy = CalibrationTransportPolicyV1(
    quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
    miscoverage_rate=0.1,
    minimum_source_units=8,
    neighbor_count=1,
    maximum_unsupported_group_fraction=0.0,
    maximum_unsupported_row_fraction=0.0,
    absolute_scale_floor=1e-8,
    relative_scale_floor=1e-6,
)
model = fit_calibration_transport_model(
    source_units,
    policy=policy,
    metadata={"split_lock_id": "<frozen split lock SHA-256>"},
)
save_calibration_transport_model(model, "calibration-transport-model.json")

# Build target units only from frames before the exclusive causal cutoff.
evidence = evaluate_calibration_transport(
    model,
    frozen_target_prefix_groups,
    metadata={"causal_cutoff": 134},
)
save_calibration_transport_evidence(
    evidence,
    "calibration-transport-evidence.json",
)

if not evidence.accepted:
    use_exact_physical_fallback()
```

`from_feature_grid` accepts `SourceReliabilityFeatures` directly through its
`feature_names` and `flattened()` interface. Per-unit metadata should bind the
source artifact, object/session identity, camera/horizon grouping, causal cutoff,
and exact feature-builder configuration.

## Replay and tamper resistance

The model and evidence are strict finite-JSON artifacts. They:

- reject duplicate object keys and non-finite numbers;
- defensively own immutable arrays and metadata;
- sort source and target unit IDs before fitting/evaluation;
- derive all thresholds, diagnostics, decisions, and content IDs from retained
  embeddings rather than trusting serialized derived fields;
- reject a changed feature-contract ID or feature ordering;
- reject target/source unit-ID overlap; and
- default to atomic no-clobber persistence.

Source-unit order, target-unit order, and feature-row order do not change the
result. Feature order remains part of the contract and may not be rewritten.

## Interpretation

A passing certificate means only that the target-prefix source-only feature
summaries are not more isolated than permitted by the source-unit calibration
cohort. A failing certificate localizes extrapolation by target group and
feature. Neither result measures point accuracy, covariance coverage, physical
query improvement, accepted-update harm, or intervention benefit. Those remain
separate held-out provider and BayesianPhysTwin/Causal4D endpoints.
