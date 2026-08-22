# CUT3R source qualification runbook

This runbook turns the existing CUT3R adapter, comparison lock, readiness gates,
and held-out promotion protocol into one execution order. It is the repository
side of [issue #49](https://github.com/IPS-Stuttgart/Prob4D/issues/49).

The runbook does not authorize access to an unopened target cohort. Exact
provider, model, runtime, corpus, calibration, and cohort identities must be
filled from retained bytes; placeholders are never claim-bearing evidence.

## Registered question

The source experiment must distinguish CUT3R recurrent-state value from Prob4D
fusion value. Freeze these arms on complete object or acquisition-session groups:

| Arm | Purpose | Causal use |
| --- | --- | --- |
| `native-continuous` | one uninterrupted recurrent-online CUT3R pass | claim eligible |
| `restarted-newest` | restart CUT3R for each overlapping causal window and retain the newest eligible prediction | claim eligible |
| `restarted-prob4d-fused` | fuse the same restarted windows through Prob4D | claim eligible |
| `revisit-diagnostic` | noncausal provider upper bound | diagnostic only |

The primary Prob4D contrast is

```text
restarted-prob4d-fused versus restarted-newest
```

because both arms use exactly the same restarted source windows. The contextual

```text
native-continuous versus restarted-newest
```

contrast measures the value of CUT3R's persistent recurrent state.

## Information freeze

Before residuals or outcomes are opened, bind:

- disjoint development, calibration, source-evaluation, and target groups;
- complete required camera, robot, timestamp, mask, and identity streams;
- exact CUT3R revision, checkpoint, loader, argument vector, and runtime;
- exact Prob4D and BayesianPhysTwin distribution identities;
- a Causal4D identity only for a separately registered downstream query;
- causal frame stops, window size, overlap, resolution, and confidence rule;
- provider and BayesianPhysTwin baselines;
- source calibration policies and group-level decision margins;
- unsupported-case and technical-failure dispositions; and
- exact physical fallback for every unsupported, invalid, rejected, or
  unidentifiable update.

The group partition and policies must be frozen before any source score is read.

## Phase 0: freeze the source comparison

Copy the checked-in specification, replace every placeholder with exact retained
identities, and build the immutable comparison lock:

```bash
cp docs/examples/cut3r-comparison-spec.json \
  outputs/cut3r/source-comparison-spec.json

prob4d prediction cut3r-comparison build \
  outputs/cut3r/source-comparison-spec.json \
  --output outputs/cut3r/source-comparison-lock.json

prob4d prediction cut3r-comparison verify \
  outputs/cut3r/source-comparison-lock.json

prob4d prediction cut3r-comparison summarize \
  outputs/cut3r/source-comparison-lock.json \
  --json
```

Do not edit the lock after source outcomes are opened. A changed provider,
prefix, group roster, comparison arm, or support rule is a new method version.

## Phase 1: support feasibility before residuals

Build the exact support-feasibility request for every required stream. Derive
the contiguous support envelope without loading predictions or residuals:

```bash
prob4d diagnostic provider-support-envelope derive \
  --request outputs/cut3r/support-request.json \
  --output outputs/cut3r/support-envelope.json

prob4d diagnostic provider-support-envelope verify \
  --artifact outputs/cut3r/support-envelope.json
```

Apply the predeclared complete-stream rule.

- If support fails, retain the support-negative artifact and stop this provider
  version.
- Do not delete cameras, shorten the prefix, replace objects, fit only supported
  streams, or relax a threshold after seeing the negative.
- A prospectively different support design requires a new frozen request before
  later information is opened.

## Phase 2: execute and attest CUT3R

Run each arm through an external wrapper that snapshots the provider revision,
checkpoint, input, loader, runtime, argument vector, and causal declarations
before inference. The admissible CUT3R import is recurrent online, source ordered,
prefix only, with one revisit and no global alignment.

Import every retained recurrent-online output:

```bash
prob4d prediction import-cut3r-online \
  outputs/cut3r/raw/group-a/restarted-window-000 \
  outputs/cut3r/provider/group-a/window-000.json \
  --sequence-id group-a-window-000 \
  --cut3r-revision <exact-cut3r-revision> \
  --checkpoint-sha256 <checkpoint-sha256> \
  --input-video-sha256 <input-video-sha256> \
  --input-video-byte-count <input-video-bytes> \
  --frame-start 0 \
  --confidence-threshold <frozen-threshold>
```

Create and verify one execution attestation per immutable execution:

```bash
python -m prob4d.provider_execution_attestation create \
  outputs/cut3r/provider-execution-spec.json \
  --output outputs/cut3r/provider-execution-attestation.json

python -m prob4d.provider_execution_attestation verify \
  outputs/cut3r/provider-execution-attestation.json \
  --require-complete
```

The noncausal revisit arm must remain separately labelled and must never enter a
claim-bearing source or target decision.

## Phase 3: source mean and identity competence

Score the three causal arms on the frozen source-evaluation groups with equal
group mass. Build `SourceProviderCompetenceReportV1` from complete object or
session records and report at least:

- proper score relative to `restarted-newest`;
- point and endpoint error;
- seam and drift error;
- support and technical-failure accounting;
- association precision and identity retention; and
- worst-group error and pass status.

Stop on a source-mean negative. Do not relabel inaccurate means, drift, broken
identities, or poor support as a covariance problem.

### Independent support and proper-score audit

The common-support v2 envelope is necessary but not sufficient for a claim-bearing
source decision: both arms could still carry the same incorrectly supplied digest.
Before emitting readiness gates, freeze and execute the additive audit in
`docs/cut3r-source-competence-support-audit.md`. It must:

- bind the exact development/calibration-only proper-score reference artifact and
  exact file-byte SHA-256 before source scores are opened;
- retain the ordered canonical point, endpoint, proper-score, and seam row arrays;
- independently reconstruct every digest, count, and score dimension;
- compare the reconstruction separately with `restarted-newest` and
  `restarted-prob4d-fused`; and
- emit source-mean and identity gates whose evidence ID is the audit report ID.

Once the support-audit lock exists, the unaudited v2 gate output is diagnostic and
may not enter the claim-bearing readiness portfolio. A valid negative v2 decision
remains reportable after successful support/reference auditing. A support or
reference mismatch invalidates the metric construction and stops the execution.

## Phase 4: localize uncertainty failure

Only after source means and identities pass, run the joint covariance diagnostic
and bind it to the exact same source groups. Evaluate the frozen localization
policy:

```bash
prob4d diagnostic source-covariance-localization evaluate \
  --source-competence outputs/cut3r/source-provider-competence.json \
  --joint-diagnostic outputs/cut3r/joint-covariance.json \
  --policy outputs/cut3r/covariance-localization-policy.json \
  --output outputs/cut3r/covariance-localization.json

prob4d diagnostic source-covariance-localization verify \
  --artifact outputs/cut3r/covariance-localization.json
```

Use the joint `Sim(3)` closure diagnostic before assigning a remaining failure
to conditional point covariance:

```bash
python -m prob4d.gauge_linearization_closure build \
  outputs/cut3r/gauge-linearization-closure-input.json \
  --output outputs/cut3r/gauge-linearization-closure.json

python -m prob4d.gauge_linearization_closure verify \
  outputs/cut3r/gauge-linearization-closure.json
```

Interpret terminal results as follows:

- mean negative -> stop;
- identity negative -> stop;
- gauge or dependence negative -> repair or replace the gauge model under a new
  source protocol;
- linearization negative -> use nonlinear propagation or query projection under
  a new source protocol;
- `point-covariance-localized` -> and only then develop a richer conditional
  point model; or
- covariance adequate -> proceed to query relevance.

## Phase 5: freeze query, target protocol, and readiness

Bind the source-only BayesianPhysTwin query projection, its exact Jacobian
lineage, the target roster, and the exact physical fallback. After the single
source-selected candidate is fixed, freeze the held-out protocol before target
provider manifests or outcomes are opened:

```bash
prob4d experiment heldout-provider freeze \
  outputs/cut3r/heldout-protocol.json \
  --output outputs/cut3r/promotion-lock.json
```

When the study uses a separately owned cohort binding, supply the exact
`--cohort-binding` required by the frozen protocol.

The fresh-provider cohort lock must reference this exact promotion-lock identity.
Build the complete target-free readiness decision only after that identity is
available:

```bash
prob4d experiment fresh-provider-readiness evaluate \
  --request outputs/cut3r/readiness-request.json \
  --output outputs/cut3r/readiness-decision.json

prob4d experiment fresh-provider-readiness verify-decision \
  --artifact outputs/cut3r/readiness-decision.json
```

Only `ready-for-one-target-evaluation` authorizes a target run. Every other
classification is a complete negative or bounded source result for this provider
version. For a positive decision, seal and replay the one-shot target
authorization:

```bash
prob4d experiment fresh-provider-readiness authorize-target \
  --decision outputs/cut3r/readiness-decision.json \
  --output outputs/cut3r/target-authorization.json

prob4d experiment fresh-provider-readiness verify-authorization \
  --artifact outputs/cut3r/target-authorization.json
```

Run the target-free observation rehearsal from the exact reviewed Prob4D
revision:

```bash
prob4d diagnostic target-free-rehearsal run \
  outputs/cut3r/target-free-rehearsal \
  --source-revision "$(git rev-parse HEAD)"

prob4d diagnostic target-free-rehearsal verify \
  outputs/cut3r/target-free-rehearsal/target_free_rehearsal_receipt.json
```

Then quantify the resolution of the frozen independent-group count:

```bash
prob4d study preflight \
  outputs/cut3r/promotion-lock.json \
  --source-summary-id <source-summary-sha256> \
  --source-metric deployed_minus_physical_rmse_mm \
  --paired-sd source-estimate=<source-paired-sd> \
  --output-dir outputs/cut3r/study-preflight
```

The readiness decision, authorization, rehearsal, and preflight do not alter the
target roster, decision margin, fallback, provider, or analysis.

## Phase 6: metadata-only target admission

If and only if the target authorization passes, generate one provider-neutral
target manifest for every frozen target group without opening target outcomes.
Bind those manifests to the promotion lock and cohort:

```bash
prob4d provider target-admit \
  outputs/cut3r/promotion-lock.json \
  outputs/cut3r/cohort-binding.json \
  outputs/cut3r/target-provider-admission-config.json \
  --output outputs/cut3r/target-provider-admission.json

prob4d provider target-verify \
  outputs/cut3r/target-provider-admission.json \
  outputs/cut3r/promotion-lock.json \
  outputs/cut3r/cohort-binding.json \
  outputs/cut3r/target-provider-admission-config.json
```

This step validates manifest metadata and causal payload lineage only. It must
complete before truth, dense target predictions, or physical-query outcomes are
opened.

## Phase 7: one frozen target evaluation

If and only if the one-shot target authorization and metadata admission pass,
execute the provider evaluator once:

```bash
prob4d evaluate provider \
  outputs/cut3r/provider-evaluation-v2.json \
  --promotion-lock outputs/cut3r/promotion-lock.json \
  --target-provider-admission outputs/cut3r/target-provider-admission.json \
  --bootstrap-resamples 10000 \
  --seed <frozen-bootstrap-seed> \
  --output-dir outputs/cut3r/provider
```

The existing promotion schema may retain required diagnostic or
sensor-assisted roles. The primary scientific report must nevertheless
foreground only:

1. unchanged physical fallback;
2. the frozen simple visual or `last_residual` comparator; and
3. the single source-selected Prob4D candidate.

Every unsupported, invalid, rejected, technically failed, or unidentifiable
candidate must deploy the exact physical fallback. Diagnostic arms cannot rescue
a failed primary comparison.

Seal and independently replay the matched provider and BayesianPhysTwin result
streams:

```bash
prob4d experiment heldout-provider run \
  outputs/cut3r/promotion-lock.json \
  --target-provider-admission outputs/cut3r/target-provider-admission.json \
  --provider-report outputs/cut3r/provider/provider_evaluation.json \
  --query-results outputs/cut3r/query-results.raw.json \
  --output-dir outputs/cut3r/promotion

prob4d experiment heldout-provider verify \
  outputs/cut3r/promotion-lock.json \
  --target-provider-admission outputs/cut3r/target-provider-admission.json \
  --provider-report outputs/cut3r/provider/provider_evaluation.json \
  --query-results outputs/cut3r/promotion/query_results.sealed.json \
  --report outputs/cut3r/promotion/promotion_report.json \
  --evidence-card outputs/cut3r/promotion/promotion_evidence_card.json
```

Report provider competence and BayesianPhysTwin value separately. Neither result
may rescue the other. Omitting `--require-pass` keeps a scientifically valid
negative as a successful retained execution rather than treating it as a retry
condition.

## Required final report

For each causal source arm, report by independent group and horizon:

- support and technical failures;
- point or identity-valid track error;
- seam and drift error;
- 50%, 90%, and 95% coverage;
- joint proper score and normalized NEES;
- full covariance width;
- identity retention, selective risk, and worst-group shortfall.

For the deployed BayesianPhysTwin policy, report:

- physical-query proper score and RMSE;
- paired group interval versus both comparators;
- accepted, rejected, and exact-fallback counts;
- harmful accepted updates and worst-group regret; and
- accepted-update coverage and width.

A valid negative must identify the first failed boundary: support, means,
identity, gauge/dependence, linearization, conditional covariance, calibration
transport, or physical-query relevance. Do not retune on the opened group set.

## Development freeze

Until this execution localizes a failure, do not add another provider adapter,
point-covariance family, calibration score, fusion heuristic, or target-side
guard. A new method is justified only by a retained source result that names the
missing capability.
