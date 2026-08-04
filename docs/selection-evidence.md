# Replayable selection evidence

`prob4d.selection_evidence` provides a portable evidence-v2 contract for
source-calibrated method selection and guarded deployment. It is intended for the
Prob4D → BayesianPhysTwin boundary, where a candidate visual update must be chosen
without using target outcomes and every rejected update must reproduce the physical
fallback exactly.

The artifact does not claim that the selected method improves physical prediction.
It makes the selection and fallback decisions independently replayable so that a
separate, frozen target analysis can establish or reject that claim.

## Retained evidence

A `SelectionEvidenceBundleV2` contains:

- every candidate method and threshold configuration;
- one immutable metric row for every calibration object/session and candidate;
- the primary objective, feasibility constraints, metric tie-breaks, and final
  complexity/identifier tie-break;
- the complete deterministic ordering of all candidates and the selected candidate;
- every target deployment guard decision;
- candidate, fallback, and deployed artifact identities for every target group; and
- separate SHA-256 identities for the full evidence bundle and its replay report.

Calibration rows must form a complete rectangular group-by-candidate matrix. Missing,
duplicate, unknown-candidate, non-finite, or noncanonical rows fail closed.

## Target-blind selection

Selection uses only `CalibrationMetricRowV1` values. Deployment decisions are stored
in a separate section and cannot affect candidate ordering.

```python
from prob4d.selection_evidence import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    DeploymentDecisionV1,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionRuleV1,
    build_selection_evidence_bundle,
)

candidates = (
    CandidateSpecV1(
        candidate_id="physical-fallback",
        method_id="physical-fallback",
        complexity_rank=0,
    ),
    CandidateSpecV1(
        candidate_id="persistent-joint-gauge",
        method_id="persistent-explicit-joint-gauge",
        complexity_rank=2,
        parameters={"minimum_track_length": 3, "guard_threshold": 0.25},
    ),
)

rows = (
    CalibrationMetricRowV1(
        group_id="calibration-object-01",
        candidate_id="physical-fallback",
        metrics={"rmse_mm": 5.1, "harmful_updates": 0, "coverage": 1.0},
    ),
    CalibrationMetricRowV1(
        group_id="calibration-object-01",
        candidate_id="persistent-joint-gauge",
        metrics={"rmse_mm": 2.2, "harmful_updates": 0, "coverage": 0.95},
    ),
)

rule = SelectionRuleV1(
    primary=MetricOrderV1("rmse_mm", "minimize"),
    tie_break_metrics=(MetricOrderV1("coverage", "maximize"),),
    constraints=(
        MetricConstraintV1("harmful_updates", "at_most", 0, "sum"),
        MetricConstraintV1("coverage", "at_least", 0.9),
    ),
)
```

A real experiment must provide all candidate rows for every calibration group. The
small example above has one group; held-out experiments should use the object or
acquisition session as the statistical unit.

## Exact fallback contract

Each `DeploymentDecisionV1` binds three content identities:

- `candidate_artifact_id`: the proposed Prob4D-informed update;
- `fallback_artifact_id`: the unchanged physical fallback; and
- `deployed_artifact_id`: the artifact actually delivered downstream.

For an accepted update, the deployed identity must equal the candidate identity. For
a rejected update, it must equal the fallback identity byte-for-byte. Construction
and loading reject any mismatch.

```python
decision = DeploymentDecisionV1(
    group_id="target-object-01",
    candidate_id="persistent-joint-gauge",
    accepted=False,
    guard_name="target-blind-residual-guard-v1",
    guard_value=0.81,
    candidate_artifact_id="a" * 64,
    fallback_artifact_id="b" * 64,
    deployed_artifact_id="b" * 64,
    reason="guard rejection; exact fallback",
)
```

## Write and independently replay

```python
from prob4d.selection_evidence import write_selection_evidence

bundle = build_selection_evidence_bundle(
    experiment_id="prob4d-bpt-real-provider-v1",
    source_repository="IPS-Stuttgart/Prob4D",
    source_revision="<40-character Git revision>",
    candidates=candidates,
    calibration_rows=rows,
    selection_rule=rule,
    deployment_decisions=(decision,),
    metadata={"split_registry_id": "<SHA-256>"},
)
write_selection_evidence(bundle, "selection-evidence.json")
```

The verifier loads the artifact with duplicate-key and unknown-field rejection,
reconstructs all aggregates and constraints, recomputes the complete candidate
ordering, checks the selected candidate, validates exact fallback decisions, and
prints the replay digest:

```bash
python -m prob4d.selection_evidence selection-evidence.json
```

The verifier does not import experiment code, model checkpoints, target outcomes, or
selection summaries produced by the original run.

## Required use in the real Prob4D → BayesianPhysTwin gate

For the prospective object/session-held-out experiment:

1. freeze candidate definitions, metric names, constraints, and split registry before
   opening target outcomes;
2. retain one calibration row per object/session and candidate, not per frame or point;
3. build the evidence bundle before target analysis;
4. retain one deployment decision for every target object/session;
5. report accepted-update error, harmful accepted updates, coverage and width,
   rejection rate, and exact-fallback rate in a separate frozen analysis; and
6. pass only the accepted BayesianPhysTwin belief and lineage to Causal4D.

A negative physical result leaves the evidence useful: it identifies the exact method
selected by source calibration and proves that rejected target updates preserved the
fallback.
