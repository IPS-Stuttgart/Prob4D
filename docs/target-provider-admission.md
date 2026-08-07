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
4. load and validate provider-manifest JSON only;
5. compare provider revision and model-set identity to the lock;
6. require the same provider contract across all target objects;
7. derive causal payload admission from each manifest's frame lineage; and
8. write one content-addressed no-clobber artifact.

The command intentionally does **not** call
`verify_prediction_provider_manifest`, because that operation opens payload
archives. Payload-byte verification belongs to the separately authorized target
execution after this metadata admission is sealed.

Replay reloads every manifest, recomputes the causal admission and content
identity, and requires exact equality with the retained artifact. Moving or
rewriting a provider manifest, changing its formatting bytes, changing a cutoff,
or changing any semantic identity invalidates replay.

## Promotion integration boundary

Before generating target provider metrics or guarded BayesianPhysTwin rows, retain
the admission ID in the target execution ledger and in the metadata of every
result bundle derived from the admitted manifests. A subsequent integration step
will make this identifier mandatory on the held-out `run` command; the current
artifact already provides the strict, replayable source of truth required for
that gate.

## Claim boundary

A passing admission proves only that the exact target manifest metadata agrees
with the frozen lock and cohort under the declared information order. It does not
establish that payload bytes are correct, that the provider is competent, that
uncertainty is calibrated, that BayesianPhysTwin improves, that Causal4D improves,
or that any method is safe or state of the art.
