# Held-out provider and BayesianPhysTwin promotion gate

`prob4d experiment heldout-provider` coordinates the decisive real-data gate
without introducing a new estimator. It composes an independently generated
Prob4D provider-competence report with complete guarded BayesianPhysTwin query
outcomes under one target-free, content-addressed protocol.

The statistical unit is a complete physical object or acquisition session. A
frame, point, track, or already-opened output directory is not an independent
held-out unit.

## Four-stage workflow

### 0. Bind the authoritative Stage-0 cohort

For the fresh-object Deform360 experiment, cohort ownership remains in
BayesianPhysTwin. Prob4D must consume the committed official-Hub Stage-0
selection rather than discover, rename, replace, or re-split objects locally.
The authoritative artifact is:

```text
IPS-Stuttgart/BayesianPhysTwin
protocols/locks/deform360_official_hub_visuotactile_v1_selection.json
```

It contains ten calibration objects and twelve confirmation objects, balanced
between the registered sheet and volumetric strata. The Stage-0 information
boundary permits official object names and the selected `metadata.json` files
only. Camera media, tactile and robot arrays, geometry annotations, and target
outcomes remain unopened at selection time.

Check out the exact BayesianPhysTwin revision that will be used by the experiment
and create a portable binding:

```bash
bpt_revision="$(git -C ../BayesianPhysTwin rev-parse HEAD)"

prob4d experiment heldout-provider cohort-bind \
  ../BayesianPhysTwin/protocols/locks/\
deform360_official_hub_visuotactile_v1_selection.json \
  --source-revision "${bpt_revision}" \
  --output deform360-cohort-binding.json
```

The command validates all three nested BayesianPhysTwin identities:

- `selection_sha256`, over the exact 10/12 object-and-episode selection;
- `content_selection_sha256`, over the complete Stage-0 content before the
  implementation revision is attached; and
- `selection_artifact_sha256`, over the committed artifact content.

It additionally enforces:

- exactly five sheet and five volumetric calibration objects;
- exactly six sheet and six volumetric confirmation objects;
- object-level disjointness and one selected episode per object;
- exact object-metadata paths and metadata SHA-256 values;
- the official dataset and resolved revision;
- the frozen official-processing repository and revision;
- the names/metadata-only information boundary;
- prohibition of replacement after payload access; and
- exact BayesianPhysTwin source repository, revision, and path provenance.

Replay the portable binding independently, optionally against the source
selection bytes:

```bash
prob4d experiment heldout-provider cohort-verify \
  deform360-cohort-binding.json \
  --selection ../BayesianPhysTwin/protocols/locks/\
deform360_official_hub_visuotactile_v1_selection.json
```

The binding is protocol evidence only. It does not open raw Deform360 payloads
or establish provider competence or physical benefit.

### 1. Freeze before target access

Start from the documented configuration skeleton and replace every placeholder
revision, digest, group, method, margin, and metadata value:

```bash
cp docs/examples/deform360-heldout-provider-promotion-config.json protocol.json
```

For the real Deform360 gate:

- copy `calibration_group_ids` and `target_group_ids` exactly from the cohort
  binding;
- set `bayesian_phystwin_repository` and `bayesian_phystwin_revision` to the
  binding source;
- set `minimum_target_group_count` to the complete confirmation count, twelve;
- keep development groups disjoint from both bound splits; and
- set `frozen_artifact_ids.cohort_binding` to the exact `cohort_binding_id`.

Then require agreement while freezing:

```bash
prob4d experiment heldout-provider freeze protocol.json \
  --cohort-binding deform360-cohort-binding.json \
  --output promotion-lock.json
```

The command fails if the BayesianPhysTwin source, calibration split, confirmation
split, complete-target requirement, development separation, or binding identity
differs. Omitting `--cohort-binding` preserves the historical interface for
existing controlled and synthetic studies; new Deform360 claim-bearing runs
should use the bound route.

The promotion lock binds the exact Prob4D, BayesianPhysTwin, and MotionCrafter
revisions; immutable model and run-spec identities; the provider-evaluation
manifest; all calibration, cohort, selection, and guard artifacts; the bootstrap
unit and seed; and every decision margin.

Exactly one arm is required for each registered role:

1. unchanged physical fallback;
2. simple visual baseline;
3. persistent identities with row-wise gauge marginalization;
4. framewise observations with the complete explicit joint gauge nuisance;
5. persistent observations with the complete explicit joint gauge nuisance;
6. cross-window identity marginalization with the complete joint gauge nuisance;
7. an independently anchored, visibly sensor-assisted arm.

Additional diagnostic arms are permitted, but the primary query arm must be a
non-sensor, non-diagnostic candidate. The physical fallback has no provider
method because it is not a visual observation source.

### 2. Run once on the frozen target

First run `prob4d evaluate provider` with a decision-bearing schema-v2 manifest.
The resulting provider report must be schema version 3, use common support,
reject legacy artifacts, use no oracle alignment, and match the frozen target
groups, methods, reference, bootstrap count, seed, and manifest digest exactly.

BayesianPhysTwin then writes one raw row for every target-group/arm pair:

```json
{
  "promotion_lock_id": "<promotion-lock SHA-256>",
  "rows": [
    {
      "group_id": "166-glove-green-cloth",
      "arm_id": "fallback",
      "query_rmse_mm": 5.1,
      "deployed_artifact_id": "<physical-fallback SHA-256>",
      "fallback_artifact_id": "<physical-fallback SHA-256>",
      "accepted": null,
      "exact_fallback_reproduced": null,
      "accepted_coverage": null,
      "accepted_width_mm": null,
      "technical_failure": false,
      "technical_failure_reason": null,
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

Reference fallback rows use `accepted=null`. Accepted visual updates use
`accepted=true` and may carry coverage and width. Rejected updates use
`accepted=false`, deploy the exact fallback artifact, and set
`exact_fallback_reproduced=true`. Technical failures are retained as rejected
rows with a reason; they are never silently excluded.

Seal and evaluate the complete matrix:

```bash
prob4d experiment heldout-provider run promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results query-results.raw.json \
  --output-dir outputs/promotion \
  --require-pass
```

The command refuses an output directory containing any retained output, so a
repeated invocation cannot silently replace an opened result. It writes:

- `query_results.sealed.json`, with canonical ordering and a content identity;
- `promotion_report.json`, with both provider and physical-query decisions;
- `promotion_report.md`, a compact gate table;
- `promotion_diagnosis.json`, a content-addressed candidate-boundary attribution;
- `promotion_diagnosis.md`, a human-readable evidence and next-action summary;
- `promotion_evidence_card.json`, a content-addressed paper-facing summary; and
- `promotion_evidence_card.md`, the corresponding compact evidence card.

The evidence card is derived only from the validated lock and deterministic
promotion report. It retains exact source revisions, frozen artifact IDs,
cohort counts, comparison arms, the paired query effect and interval, guard
outcomes, and explicit non-claims. The frozen `cohort_binding` ID links the
paper-facing result to the exact BPT Stage-0 selection without duplicating cohort
ownership in Prob4D.

Without `--require-pass`, a scientifically valid negative result still returns
exit code 0 after writing all evidence. With it, a valid failed gate returns exit
code 3.

### 3. Replay independently

```bash
prob4d experiment heldout-provider verify promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results outputs/promotion/query_results.sealed.json \
  --report outputs/promotion/promotion_report.json \
  --evidence-card outputs/promotion/promotion_evidence_card.json
```

Verification revalidates every claim-bearing artifact, recomputes the
deterministic report, and optionally requires the retained evidence card to match
its replay. Any changed row, method set, target group, fallback identity,
bootstrap setting, report field, or evidence-card field fails closed.

The diagnosis is derived only from the retained report and can be regenerated
independently for an older report:

```bash
prob4d experiment heldout-provider diagnose \
  outputs/promotion/promotion_report.json \
  --output outputs/promotion/promotion_diagnosis.replayed.json \
  --markdown outputs/promotion/promotion_diagnosis.replayed.md
```

Matching diagnosis IDs establish deterministic replay of the attribution. The
diagnosis never changes the promotion report or its decision.

## Deterministic failure attribution

A failed gate is mapped to one or more ordered candidate boundaries. Direct
query gates identify technical integrity, exact fallback, independent-group
support, guard calibration, worst-object/session transfer, and accepted-update
coverage failures. Failed provider rules are grouped by their frozen metric
family into observation quality, gauge consistency, identity persistence,
uncertainty calibration, or support/reliability.

When provider competence passes but query superiority fails, the diagnosis marks
`query_identifiability_or_physical_model_discrepancy`: the retained evidence does
not distinguish an uninformative physical query from a deficient physical model.
When both provider competence and query superiority fail, it instead marks
`downstream_query_superiority` and directs follow-up to the upstream provider
boundary first.

These labels are candidate boundaries, not causal proof. They are generated from
predeclared gates and metric names, include the exact observed and required
values, and explicitly forbid repairing a failed target result by post-hoc
retuning. Any changed estimator, calibration, cohort selection, guard, or
diagnostic policy requires a new unopened target cohort.

## Conjunctive physical-query gates

The primary candidate passes only when all of the following pass:

- the upper 95% paired target-group bootstrap bound clears the frozen superiority
  margin relative to physical fallback;
- harmful accepted updates do not exceed the frozen count;
- worst-group regression remains within its frozen limit;
- technical failures do not exceed their frozen limit;
- accepted-update mean coverage reaches the frozen threshold when one is set; and
- every rejected update reproduces the exact physical fallback.

Provider competence is a separate conjunctive gate. A good observation score
does not authorize a Bayesian update, and a guarded query result does not repair
a failed provider report.

## Claim boundary

A passing report supports only the exact frozen provider and guarded
BayesianPhysTwin query on the declared independent objects or sessions. It does
not establish general MotionCrafter competence, calibrated uncertainty outside
that cohort, Causal4D intervention benefit, or overall state of the art. A failed
well-powered gate is complete evidence and must not be retuned on the same opened
target cohort.
