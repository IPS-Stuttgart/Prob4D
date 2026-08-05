# Append-only material-identity hypotheses

`prob4d.material_identity_stream` is an experimental source-only contract for
carrying cross-window tracklet associations through a causal sequence without
rewriting window-local point IDs.

It addresses the next gap after pairwise cross-window association: a long
sequence may contain several overlapping prediction windows, and each new window
can have plausible links to more than one already admitted window. Treating those
links as immediate global identities would hide ambiguity, create transitive
conflicts, and make later append operations capable of changing earlier IDs.

## Safety boundary

The stream therefore retains **hypotheses**, not global identities:

- every endpoint remains `(window_id, track_id)`;
- every hypothesis binds the complete pairwise association `result_id`;
- the compatibility score remains a source-side ranking statistic, not a
  calibrated posterior probability;
- the pairwise mutual-best decision is recorded as
  `selected_by_pairwise_gate`, but several source windows may still nominate the
  same target track;
- no connected component, union-find label, or canonical material-point ID is
  emitted; and
- provider-v2 observation-factor IDs are never rewritten.

A downstream experiment can inspect these hypotheses, fit a separately frozen
multi-window admission rule, or marginalize identity uncertainty. It must not
interpret a pairwise-selected link as a calibrated global correspondence.

## Directed append contract

Create a root-only stream before admitting any cross-window result:

```python
from prob4d.material_identity_stream import create_material_identity_stream

stream = create_material_identity_stream(
    sequence_id="sequence-01",
    case_id="case-01",
    stream_id="camera0",
    source_repository="IPS-Stuttgart/Prob4D",
    source_revision="<40-character lowercase Git SHA>",
    root_window_id="window-000",
    metadata={"claim_bearing": False},
)
```

Root provenance is canonical and content-bearing: `source_repository` must use
exact `owner/name` form and `source_revision` must be a lowercase 40-character
Git SHA. Branch names, tags, abbreviated SHAs, and whitespace-padded aliases fail
closed so the stream cannot claim an ambiguous source revision.

Pairwise association results used for append must be directed as:

```text
left window  = already admitted source window
right window = one previously unseen target window
```

Append one or several source results for that same target and causal cutoff:

```python
from prob4d.material_identity_stream import append_material_identity_update

stream = append_material_identity_update(
    stream,
    [result_from_window_000, result_from_window_001],
    target_window_id="window-002",
)
```

The implementation sorts source summaries canonically and requires:

- exactly one new target window per update;
- every source window to have been admitted earlier;
- unique source windows within the update;
- one common exclusive causal cutoff;
- nondecreasing cutoffs across updates;
- contiguous update indices; and
- an exact previous-update hash chain.

Those rules make cycles and retroactive edge insertion impossible by
construction. Adding a new update preserves every previous update object and ID.

## Retained pairwise audit

For each source-to-target result, the stream retains:

- source and target track-domain sizes;
- possible, spatially admitted, spatially rejected, and fully evaluated pair
  counts;
- insufficient-overlap, zero-support, low-support, non-mutual, ambiguous, and
  threshold-rejection counts;
- exact unmatched source and target track IDs;
- every scored candidate with geometry, support, ranking score, shared frames,
  and pairwise-selection status; and
- the pairwise association result content identity.

The stream recomputes and verifies all count, domain, one-to-one, unmatched-set,
and nested content-address invariants during direct construction and loading.

## Portable persistence

```python
from prob4d.material_identity_stream import (
    load_material_identity_stream,
    write_material_identity_stream,
)

write_material_identity_stream(stream, "outputs/material-identities.json")
validated = load_material_identity_stream("outputs/material-identities.json")
```

The JSON loader rejects duplicate keys, unknown or missing fields, noncanonical
scalar types, malformed nested IDs, broken update chains, changed metadata, and
artifact-ID mismatches. Metadata is recursively immutable after validation.

## Required experiment before identity promotion

The stream is infrastructure for the registered downstream comparison, not a
new claim-bearing provider mode. A promotion experiment should freeze all
multi-window admission rules on development/calibration objects and compare:

1. framewise identities;
2. persistent within-window identities;
3. pairwise-selected cross-window hypotheses;
4. identity-uncertainty marginalization or a conservative multi-window rule; and
5. exact physical fallback.

Report association precision/retention where material labels exist, but make the
decisive endpoints object/session-level physical prediction, harmful accepted
updates, coverage and width, rejection rate, and exact fallback. A negative
result leaves local provider identities unchanged.

## Command-line validation

Validate a retained stream without importing the Python API:

```bash
prob4d identity validate-stream outputs/material-identities.json
```

The grouped interface and downstream mixture tools are documented in
[the material-identity command line](material-identity-cli.md).
