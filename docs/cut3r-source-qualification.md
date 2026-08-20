# CUT3R source qualification runbook

This runbook executes the already-frozen source-only comparison for the
recurrent-online CUT3R provider and prepares the real-provider decision tracked
in [issue #49](https://github.com/IPS-Stuttgart/Prob4D/issues/49). It stops before
opening any held-out target outcome.

The decisive source contrast is:

```text
restarted-prob4d-fused versus restarted-newest.
```

Both arms must use the exact same restarted causal window outputs. Their
difference therefore isolates Prob4D fusion from CUT3R's internal recurrent
memory. The contextual contrast

```text
native-continuous versus restarted-newest
```

measures the value of CUT3R's persistent recurrent state. A revisit arm is a
noncausal diagnostic and can never become claim-bearing evidence.

## 1. Freeze before any residual or outcome is opened

Create one immutable source design containing:

- exact CUT3R repository revision and checkpoint SHA-256;
- exact Prob4D revision and installed-distribution SHA-256;
- complete physical-object or acquisition-session groups;
- disjoint development, calibration, and source-evaluation roles;
- every required stream and camera for every group;
- input-video SHA-256 and byte count for every case;
- source and evaluation frame intervals;
- window size, overlap, confidence threshold, storage dtype, and seeds;
- technical-failure and unsupported-case policy; and
- whether the noncausal revisit diagnostic is disabled.

Frames, points, tracks, views, and cameras are nested observations. They must not
be entered as independent groups.

Start from the checked-in example:

```bash
OUT=outputs/cut3r-source-qualification-v1
mkdir -p "$OUT"
cp docs/examples/cut3r-comparison-spec.json "$OUT/comparison-spec.json"
"${EDITOR:-vi}" "$OUT/comparison-spec.json"

prob4d prediction cut3r-comparison build \
  "$OUT/comparison-spec.json" \
  --output "$OUT/comparison-lock.json"

prob4d prediction cut3r-comparison verify \
  "$OUT/comparison-lock.json"

prob4d prediction cut3r-comparison summarize \
  "$OUT/comparison-lock.json" \
  --json > "$OUT/comparison-lock-summary.json"
```

Do not build the lock with placeholder revisions, digests, group names, or byte
counts. Rebuilding a different lock after source outcomes are visible is a new
protocol, not a repair.

## 2. Produce the frozen CUT3R arms

CUT3R inference remains external to Prob4D. Retain the exact command line,
environment, model bytes, input bytes, and output members for every run.

For each case:

1. run one uninterrupted recurrent-online pass for `native-continuous`;
2. run fresh recurrent state for every frozen overlapping causal window;
3. retain the newest eligible prediction from those restarted windows for
   `restarted-newest`;
4. fuse those same restarted window bytes through Prob4D for
   `restarted-prob4d-fused`; and
5. run revisiting only when the lock enables it, label it noncausal, and exclude
   it from every promotion decision.

Import each admissible recurrent-online CUT3R output with its exact identities:

```bash
prob4d prediction import-cut3r-online \
  "$CUT3R_OUTPUT_ROOT" \
  "$PROVIDER_MANIFEST" \
  --sequence-id "$CASE_ID" \
  --cut3r-revision "$CUT3R_REVISION" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --input-video-sha256 "$INPUT_VIDEO_SHA256" \
  --input-video-byte-count "$INPUT_VIDEO_BYTE_COUNT" \
  --frame-start "$FRAME_START" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
```

The importer must report recurrent online execution, prefix-only dependence,
`revisit_count=1`, and no global alignment. The canonical coordinate declaration
remains `sequence-local-sim3` until an independent metric anchor is admitted.

The `restarted-newest` and `restarted-prob4d-fused` arms must reference identical
provider-window identities and payload digests. A second provider rerun for only
one arm invalidates the isolating contrast.

## 3. Complete-stream support gate before residuals

Freeze a `ProviderSupportFeasibilityRequestV1` for the exact provider, object or
session roster, streams, causal spans, camera geometry, robot geometry, metric
anchor, and physical-query mapping. Derive and verify the contiguous support
envelope without opening prediction residuals:

```bash
prob4d diagnostic provider-support-envelope derive \
  --request "$OUT/support-request.json" \
  --output "$OUT/support-envelope.json"

prob4d diagnostic provider-support-envelope verify \
  --artifact "$OUT/support-envelope.json"
```

Stop immediately when any required stream is support-negative under the frozen
complete-stream rule. Retain that negative result. Do not delete cameras, shorten
the prefix, substitute partial streams, fit only supported rows, or inspect
downstream outcomes under the same provider identity.

A technical failure follows only the predeclared technical-failure policy. It
must not be relabelled as a scientific support negative, and a support negative
must not be relabelled as a software failure.

## 4. Evaluate source means and identities

Only a support-positive provider version may open the registered source residuals.

Build one provider-evaluation manifest with:

- `reference_method` equal to `restarted-newest`;
- the same method set in every case;
- `group_id` equal to the complete object or acquisition session;
- common-support primary evaluation;
- native-support retention as a secondary diagnostic; and
- one frozen primary alignment mode.

Run:

```bash
prob4d evaluate provider \
  "$OUT/provider-evaluation-manifest.json" \
  --output-dir "$OUT/provider-evaluation" \
  --bootstrap-resamples 2000 \
  --evaluation-chunk-size 65536 \
  --seed 7
```

The primary source comparison is the equal-group paired difference between
`restarted-prob4d-fused` and `restarted-newest`. Report at least:

- support and technical-failure accounting;
- point and endpoint error;
- seam and drift error;
- identity retention and association precision;
- common/native support retention;
- 50%, 90%, and 95% coverage;
- Gaussian proper score and normalized NEES;
- covariance width and worst-group coverage shortfall; and
- selective risk.

Construct and retain `SourceProviderCompetenceReportV1` from the exact evaluation
groups and a source-frozen policy. Mean quality and identity/reliability are
separate decisions. A failed mean gate is terminal for that provider version even
when its covariance looks conservative.

## 5. Localize gauge, dependence, and covariance only after useful means

When source means and identities pass, create matched source residuals with the
strict joint-covariance schema and evaluate:

```bash
prob4d diagnostic joint-covariance \
  "$OUT/matched-source-residuals.npz" \
  --output "$OUT/joint-covariance.json"

python -m prob4d.joint_covariance_ablation \
  "$OUT/matched-source-residuals.npz" \
  --output "$OUT/joint-covariance-ablation.json" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 7 \
  --confidence-level 0.95
```

Run the joint `Sim(3)` linearization-closure diagnostic before assigning any
remaining failure to conditional point covariance:

```bash
python -m prob4d.gauge_linearization_closure --help
```

Then bind source competence and joint diagnostics through the frozen localization
policy:

```bash
prob4d diagnostic source-covariance-localization evaluate \
  --source-competence "$OUT/source-provider-competence.json" \
  --joint-diagnostic "$OUT/joint-covariance.json" \
  --policy "$OUT/covariance-localization-policy.json" \
  --output "$OUT/covariance-localization.json"

prob4d diagnostic source-covariance-localization verify \
  --artifact "$OUT/covariance-localization.json"
```

Interpret the terminal result as follows:

| Result | Required action |
| --- | --- |
| source-mean negative | Stop; do not change covariance |
| identity/reliability negative | Stop; repair identities under a new source protocol |
| gauge-or-dependence negative | Redirect gauge/dependence modelling |
| linearization-closure negative | Redirect propagation or query linearization |
| point-covariance-localized | Point-uncertainty development is authorized on source data only |
| covariance-adequate | Continue to physical-query relevance |

A joint-NEES failure that is not localized to the conditional subspace is not
authorization for a richer point model.

## 6. Query relevance and complete readiness

BayesianPhysTwin owns the physical query and its target-blind Jacobian. Bind the
exact query projection, row order, source observation, provider manifest,
conditional covariance, and shared factor before evaluating relevance.

Compose all target-free gates:

```bash
prob4d experiment fresh-provider-readiness evaluate \
  --request "$OUT/readiness-request.json" \
  --output "$OUT/readiness-decision.json"
```

Only the exact classification

```text
ready-for-one-target-evaluation
```

permits one evaluation of the bound unopened target roster. Before that one-shot
execution, run the target-free observation rehearsal and independently verify the
positive and adversarial controls:

```bash
prob4d diagnostic target-free-rehearsal --help
```

This runbook does not execute the target evaluation.

## 7. Frozen target design after a positive source decision

The later issue-#49 target run must score these arms on the same complete target
objects or sessions:

1. unchanged physical fallback;
2. one simple direct visual or `last_residual` comparator;
3. one source-selected Prob4D candidate with complete joint gauge uncertainty;
4. exact physical fallback for every unsupported, invalid, rejected, or
   unidentifiable candidate.

Provider competence and downstream BayesianPhysTwin value remain separate
conjunctive endpoints. A downstream improvement cannot rescue failed support,
means, identities, or calibration. Causal4D may consume only the selected
BayesianPhysTwin belief under a separately registered downstream query.

## 8. Retained evidence

Retain at minimum:

- comparison spec and content-addressed lock;
- exact provider, model, environment, and distribution identities;
- every input and generated source-member digest;
- imported provider manifests and execution attestations;
- support request, feasibility result, and support envelope;
- provider-evaluation manifest and complete report;
- source-competence report and policy;
- joint covariance and dependence-ablation reports;
- linearization-closure artifact;
- covariance-localization decision;
- query-projection binding and relevance result;
- complete readiness request and terminal decision; and
- an explicit statement that zero target outcomes were opened.

A valid negative at any gate is a complete result for that provider version.
Do not retune it on the same opened source or target cohort.

Related documentation:

- [CUT3R recurrent-online adapter](cut3r-online-provider.md)
- [Frozen native-versus-Prob4D comparison](cut3r-comparison.md)
- [Provider evaluation](provider-evaluation.md)
- [Source-only provider competence](source-provider-competence.md)
- [Joint covariance diagnostics](joint-covariance-diagnostics.md)
- [Provider readiness localization](provider-readiness-localization.md)
- [Scientific kernel](scientific-kernel.md)
