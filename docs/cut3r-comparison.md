# CUT3R native-versus-Prob4D comparison

`prob4d prediction cut3r-comparison` freezes the source-only experiment that
separates CUT3R's own recurrent-state value from the value of Prob4D fusion.
The lock is outcome-blind and content-addressed. It does not execute CUT3R or
open a target cohort.

The complete execution order, stop rules, and final reporting requirements are
in the [CUT3R source qualification runbook](cut3r-qualification-runbook.md).

## Frozen arms

The lock always declares four arms:

| Arm | Purpose | Causal | Claim eligible |
| --- | --- | --- | --- |
| `native-continuous` | One uninterrupted recurrent-online CUT3R pass | yes | yes |
| `restarted-newest` | Fresh CUT3R state for each overlapping causal window; retain the newest eligible prediction | yes | yes |
| `restarted-prob4d-fused` | The same restarted windows fused through Prob4D | yes | yes |
| `revisit-diagnostic` | Provider revisiting as a noncausal upper-bound diagnostic | no | never |

The primary contrast is `restarted-prob4d-fused` versus `restarted-newest`.
Because both arms use exactly the same restarted source windows, this contrast
isolates the contribution of Prob4D fusion. The contextual
`native-continuous` versus `restarted-newest` contrast measures how much value
comes from CUT3R's persistent recurrent state.

The revisiting arm remains present even when disabled. Its causal and claim
flags are immutable, so it cannot silently become promotion evidence.

## Independent evidence units

Frames, points, tracks, and views are nested observations. The lock accepts only
complete `physical-object-or-acquisition-session` groups and partitions every
group exactly once into:

- development;
- calibration; or
- source evaluation.

The three roles must be nonempty and disjoint. Every case binds the input-video
digest, byte count, complete source interval, and a common evaluation interval.
Across-group reporting must give complete groups equal weight.

## Build and verify

Start from the checked-in example and replace every placeholder revision,
distribution digest, video digest, and group roster with the exact source-only
experiment identities:

```bash
prob4d prediction cut3r-comparison build \
  docs/examples/cut3r-comparison-spec.json \
  --output outputs/cut3r/comparison-lock.json

prob4d prediction cut3r-comparison verify \
  outputs/cut3r/comparison-lock.json

prob4d prediction cut3r-comparison summarize \
  outputs/cut3r/comparison-lock.json \
  --json
```

Publication is atomic and no-clobber. Repeating an identical write is
idempotent; a different lock at the same path is rejected.

## Registered provider endpoints

The lock freezes support and technical-failure accounting, point or track error,
seam and drift error, 50/90/95% coverage, proper score, normalized NEES, full
covariance width, identity retention, selective risk, and worst-group coverage
shortfall. It intentionally contains no observed outcome values and authorizes
no target access.

After execution, use the
[CUT3R source competence evaluator](cut3r-source-competence-evaluation.md) to
turn the complete paired source records into the authoritative source-mean and
identity/reliability decision. That evaluator binds one registered causal
contrast, rejects incomplete frame/seed/arm support, retains predeclared
complete-group technical failures, and emits the existing
`SourceProviderCompetenceReportV1` artifact consumed by readiness logic.

A passing source comparison is not a held-out promotion result. A later target
experiment must use the independently frozen held-out provider and
BayesianPhysTwin gate, with provider competence and downstream physical-query
benefit reported separately.

## Frozen diagnostic strata

The aggregate source endpoints remain primary. Before source scores are opened,
freeze the optional [CUT3R diagnostic strata](cut3r-stratified-reporting.md) to
localize long-horizon, restart-boundary, occlusion, deformation, viewpoint, and
metric-anchor failures without creating additional method-selection endpoints.

The strata use one arm-neutral feature value for every paired observation. They
therefore cannot be derived separately from the native, restarted-newest, and
Prob4D-fused predictions. Reports first average nested records inside each frozen
seed, then give seeds equal mass inside each case, cases equal mass inside each
complete object/session, and finally complete groups equal mass. A sparsely
populated bin remains descriptive and cannot rescue or reverse the registered
aggregate source decision.
