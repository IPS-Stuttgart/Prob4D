# Pre-target selection locks and deployment ledgers

`prob4d.selection_lock` and `prob4d.deployment_ledger` separate two evidence
stages that must not share one content identity:

1. source-calibration candidate selection, sealed before target access; and
2. target deployment decisions, appended only after the selection lock exists.

The earlier `prob4d.selection_evidence` bundle remains a complete combined audit
artifact. The split contracts provide a stronger temporal proof for prospective
experiments without rewriting that version-2 artifact.

## SelectionLockV1

A selection lock contains:

- the exact experiment, repository, and source revision;
- every fully specified candidate;
- the complete calibration object/session by candidate matrix;
- the frozen objective, feasibility constraints, and tie-break rule;
- the independently replayed complete candidate order;
- the selected candidate; and
- finite immutable metadata such as split-registry and calibration identities.

It deliberately contains no deployment row and no target outcome. The lock ID and
replay digest can therefore be committed to the code and paper repositories before
opening target payloads.

```python
from prob4d.selection_lock import build_selection_lock, write_selection_lock

lock = build_selection_lock(
    experiment_id="prob4d-bpt-real-provider-v1",
    source_repository="IPS-Stuttgart/Prob4D",
    source_revision="<40-character lowercase Git SHA>",
    candidates=candidates,
    calibration_rows=calibration_rows,
    selection_rule=selection_rule,
    metadata={"split_registry_id": split_registry_id},
)
write_selection_lock(lock, "evidence/selection-lock.json")
```

Loading the JSON independently replays the complete candidate order and rejects
missing matrix rows, duplicate keys, changed candidate order, changed selection,
noncanonical provenance, altered claim boundaries, or content-ID mismatch.

## DeploymentLedgerV1

A deployment ledger is rooted at one exact `selection_lock_id`. The empty root can
be serialized before deployment begins. Each append returns a new immutable ledger
prefix while preserving the previous artifact and its ID:

```python
from prob4d.deployment_ledger import (
    append_deployment_decision,
    build_deployment_ledger,
    write_deployment_ledger,
)

ledger = build_deployment_ledger(lock, metadata={"target_split": "sealed-v1"})
write_deployment_ledger(ledger, "evidence/deployment-ledger-000.json")

ledger = append_deployment_decision(ledger, decision)
write_deployment_ledger(ledger, "evidence/deployment-ledger-001.json")
```

Every non-root ledger records the exact previous-prefix ID. Group IDs cannot be
appended twice, every decision must use the locked candidate, and the existing
`DeploymentDecisionV1` contract requires accepted decisions to deploy the exact
candidate artifact and rejected decisions to deploy the exact fallback artifact.

## Required prospective order

```text
retain complete source-calibration rows
-> create and independently replay SelectionLockV1
-> commit selection_lock_id before target access
-> create empty DeploymentLedgerV1
-> open each registered target unit under the frozen protocol
-> append its guard decision and exact deployed artifact
-> freeze target outcomes and run the separately registered analysis
```

Changing a target decision changes the deployment-ledger ID but cannot change the
selection-lock ID. Neither artifact alone establishes provider competence or
physical benefit; those remain object/session-level held-out analysis claims.
