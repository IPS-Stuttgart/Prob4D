# Held-out target provider admission

The target-provider admission contract closes the information-order gap between a
frozen held-out promotion protocol and the later provider manifests generated for
its target objects. It validates manifest metadata only. Dense prediction arrays,
truth, physical-query outcomes, and target metrics remain unopened.

## Purpose

A promotion lock freezes the provider revision, model-set identity, prediction
run-spec identity, cohort, comparison arms, and decision rules. The later target
execution still produces one provider-neutral manifest per physical target object.
Without an explicit admission step, those target manifests could silently use a
different provider revision, model set, loader, coordinate interpretation, causal
cutoff, or target subset.

`HeldoutTargetProviderAdmissionV1` binds all of the following before outcome
evaluation:

- the exact promotion-lock and Deform360 cohort-binding identities;
- the exact Prob4D source revision and frozen prediction run-spec identity;
- one provider-neutral manifest for every frozen target object;
- each manifest's exact byte SHA-256 and semantic artifact identity;
- provider family, repository, revision, model set, loader, coordinate, point,
  flow, ray, and causal-source semantics;
- the object, selected episode, registered stratum, and expected sequence ID;
- one explicit causal frame cutoff per target object; and
- every payload identity admitted by that cutoff, including its window, output
  frames, source-frame interval, and dependence groups.

The builder requires all target manifests to share one provider contract. It
requires the provider revision and model set to match the promotion lock and
requires exact coverage of the frozen target object IDs. A valid negative state,
such as no payload admitted before a cutoff, fails closed rather than being
silently excluded.

## Request configuration

Start from:

```bash
cp docs/examples/deform360-target-provider-admission-config.json \
  target-provider-admission-config.json
```

Manifest paths are safe POSIX paths relative to the request configuration's
directory. Absolute paths, parent traversal, backslashes, and symbolic-link
traversal are rejected. The output artifact does not retain those paths; paths
are retrieval metadata used only for deterministic replay.

Each request entry contains:

```json
{
  "group_id": "166-glove-green-cloth",
  "expected_sequence_id": "166-glove-green-cloth-episode-0",
  "manifest_path": "target-provider-manifests/166-glove-green-cloth.json",
  "causal_frame_stop": 134
}
```

The exact sequence ID is explicit because provider sequence naming is not inferred
from an object directory name. The selected object episode and stratum are taken
from the authoritative cohort binding, not from provider metadata.

The request must contain every frozen target object exactly once and declare:

```json
"target_outcomes_used": false
```

The request's `prediction_run_spec_id` must equal the frozen promotion-lock value.
Individual provider manifests retain their own content-addressed `provider_run_id`;
those per-object run identities are recorded separately in the admission artifact.

## Admit and replay

```bash
prob4d-target-admit \
  promotion-lock.json \
  deform360-cohort-binding.json \
  target-provider-admission-config.json \
  --output target-provider-admission.json

prob4d-target-verify \
  target-provider-admission.json \
  promotion-lock.json \
  deform360-cohort-binding.json \
  target-provider-admission-config.json
```

Admission performs this order of operations:

1. validate the promotion lock and cohort binding;
2. require their calibration/target splits and BayesianPhysTwin source identity to
   agree exactly;
3. require the lock's frozen cohort-binding ID;
4. read each provider manifest into a private exact-byte snapshot;
5. reject source mutation between snapshot creation and completion of admission;
6. compare provider revision and model-set identity to the lock;
7. require the same provider contract across all target objects;
8. derive causal payload admission from each manifest's frame lineage; and
9. write one content-addressed no-clobber artifact.

The command intentionally does **not** call
`verify_prediction_provider_manifest`, because that operation opens payload
archives. Payload-byte verification belongs to the separately authorized target
execution after this metadata admission is sealed.

Replay reloads every manifest, recomputes the causal admission and content
identity, and requires exact equality with the retained artifact. Moving or
rewriting a provider manifest, changing its formatting bytes, changing a cutoff,
or changing any semantic identity invalidates replay.

## Acyclic provider-evaluation authorization

The provider-evaluation manifest is frozen **before** the promotion lock, because
its exact SHA-256 is an identity-bearing lock field. The target-provider admission
is created **after** that lock and therefore depends on the lock identity.
Consequently, the admission ID must not be inserted back into the frozen provider
evaluation manifest: doing so would create the impossible identity cycle

```text
provider manifest SHA -> promotion lock ID -> target admission ID -> provider manifest SHA
```

The provider evaluator instead receives the lock and admission as separate command
arguments. It parses the exact manifest bytes into a target-free structural
snapshot, validates all identities and registered groups/methods, and creates a
content-addressed authorization receipt before resolving or opening truth and
prediction artifacts:

```bash
prob4d evaluate provider provider-evaluation-v2.json \
  --promotion-lock promotion-lock.json \
  --target-provider-admission target-provider-admission.json \
  --bootstrap-resamples 10000 \
  --seed 20260805 \
  --output-dir outputs/provider \
  --require-decision-pass
```

For this target-authorized route, the evaluator requires:

- a decision-bearing schema-v2 provider-evaluation manifest;
- exact manifest-byte agreement with the promotion lock;
- exact target-group, provider-method, reference-method, minimum-group-count,
  bootstrap-resample, and bootstrap-seed agreement;
- exact target-admission agreement with the lock, cohort, provider revision,
  model set, run spec, and target groups;
- `allow_legacy_artifacts=false`; and
- no `target_provider_admission_id` inside manifest metadata.

The resulting provider report is schema version 4 and contains the top-level
`target_admission_authorization` receipt. That receipt binds the manifest SHA,
promotion-lock ID, target-admission ID, cohort-binding ID, target groups,
registered methods, reference, case count, decision minimum, bootstrap settings,
and the explicit statement
`target_outcomes_opened_during_authorization=false`.

The evaluator executes a private manifest generated from the exact snapshotted
bytes. Only path representation changes: relative target paths are rewritten to
absolute paths preserving the original manifest's directory semantics. The
original manifest bytes are checked again before publication. A malformed policy,
identity mismatch, circular metadata field, or changed source manifest fails
before a claim-bearing provider report is written.

## Mandatory result-stream binding

A cohort-bound held-out promotion cannot run merely because a valid admission
artifact exists. The two result streams bind the admission in different,
acyclic locations:

- the provider report carries the content-addressed
  `target_admission_authorization` receipt described above; and
- the raw and sealed BayesianPhysTwin query results carry
  `metadata.target_provider_admission_id`.

The held-out execution must supply the retained admission explicitly:

```bash
prob4d experiment heldout-provider run promotion-lock.json \
  --target-provider-admission target-provider-admission.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results query-results.raw.json \
  --output-dir outputs/promotion

prob4d experiment heldout-provider verify promotion-lock.json \
  --target-provider-admission target-provider-admission.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results outputs/promotion/query_results.sealed.json \
  --report outputs/promotion/promotion_report.json
```

Before query rows are sealed or provider gates are combined, `run` requires:

- a promotion lock with the same frozen cohort-binding identity;
- an admission bound to the exact promotion-lock identity, source revision,
  prediction run spec, provider revision, model set, and target groups;
- a schema-v4 provider report whose authorization receipt deterministically
  replays to that lock and admission; and
- the query stream's `metadata.target_provider_admission_id` to equal the same
  retained admission identity.

`verify` replays the same checks against the sealed query results. Missing,
malformed, circular, or mismatched identities fail closed. A symbolic-link
admission path is inadmissible.

Controlled and synthetic promotion locks without a frozen `cohort_binding` retain
the historical execution path and artifact semantics. Supplying an admission to
such a lock is rejected rather than silently converting it into a physical-cohort
protocol.

## Claim boundary

A passing admission and execution binding prove only that the exact target
manifest metadata, provider authorization, and both result streams agree with the
frozen lock and cohort under the declared information order. They do not establish
that payload bytes are correct, that the provider is competent, that uncertainty
is calibrated, that BayesianPhysTwin improves, that Causal4D improves, or that any
method is safe or state of the art.
