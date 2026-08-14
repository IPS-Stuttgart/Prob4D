# Joint multi-window material-identity posterior

`prob4d.joint_material_identity` conditions several source-calibrated
`MaterialIdentityMixtureV1` artifacts on one global consistency rule without
creating global point IDs.

A local mixture chooses exactly one candidate for one target-local track:

- its mandatory null candidate, which preserves the newest-window reference; or
- one linked predecessor identified by its original `(window_id, track_id)`.

Evaluating those mixtures independently can assign the same earlier material
track to two different tracks in one later window. Pairwise links can also become
inconsistent only after several windows are connected. The joint posterior
removes those assignments before any downstream physical likelihood is applied.

## Constraint

Every candidate selection forms a directed edge from an earlier source endpoint
to one target endpoint. A complete assignment is admitted only when every
connected component contains at most one endpoint from each prediction window.
Together with one candidate per target mixture and the declared window order,
this gives an acyclic window-local identity forest.

The rule permits one material point to continue across several windows while
preventing one inferred material component from splitting into two tracks in the
same window. Endpoints remain local, and no connected-component label is exported
as a provider-v2 point identity.

For local calibrated log weights `w_ij`, the source-side joint prior is

```text
q(a) proportional to exp(sum_i w_i,a_i) * 1[a is globally feasible].
```

Conditioning therefore changes local candidate marginals when independent local
choices are mutually incompatible. Additive constants in one local mixture
cancel because every joint assignment selects exactly one candidate from that
mixture.

## Bounded exact enumeration

Version 1 enumerates the exact Cartesian product only when its unconstrained size
is no larger than `maximum_joint_assignments`. It fails closed before enumeration
when the declared bound would be exceeded. There is no silent beam search,
approximate truncation, or assignment dropping.

The self-contained posterior embeds and revalidates every source mixture. Loading
replays:

- canonical mixture and target ordering;
- common calibration, association-rule, and implementation identities;
- the global window-prefix contract;
- every feasible and rejected assignment;
- normalized assignment probabilities;
- target-aligned candidate marginals;
- entropy and effective assignment count; and
- the content-derived posterior identity.

Changing a source mixture, selected candidate, probability, marginal, count, or
claim boundary therefore fails exact replay even when the outer JSON is otherwise
well formed.

## Python API

```python
from prob4d.joint_material_identity import (
    build_joint_material_identity_posterior,
    marginalize_joint_assignment_log_likelihoods,
)

posterior = build_joint_material_identity_posterior(
    local_mixtures,
    window_order=("window-000", "window-001", "window-002"),
    maximum_joint_assignments=100_000,
    metadata={"stage": "source-only"},
)

result = marginalize_joint_assignment_log_likelihoods(
    posterior,
    posterior.assignment_ids,
    downstream_log_likelihoods,
)
```

`assignment_components` returns the selected components as tuples of original
`LocalTrackEndpoint` values. `joint_candidate_marginals` projects either the
source joint prior or a downstream assignment posterior back to the exact local
candidate order.

Candidate IDs and assignment IDs must align exactly. A likelihood power of zero
returns the source-side joint prior without consulting impossible `-inf`
likelihood entries.

## Grouped command line

Create a configuration whose mixture paths are confined relative to the
configuration file:

```json
{
  "window_order": ["window-000", "window-001", "window-002"],
  "mixture_paths": [
    "mixtures/window-001-track-000.json",
    "mixtures/window-002-track-000.json"
  ],
  "maximum_joint_assignments": 100000,
  "metadata": {"stage": "source-only"}
}
```

Then run:

```bash
prob4d identity build-joint joint-config.json \
  --output joint-material-identity.json

prob4d identity validate-joint joint-material-identity.json
```

Marginalize a candidate-aligned downstream likelihood with:

```bash
prob4d identity marginalize-joint \
  joint-material-identity.json \
  joint-assignment-likelihoods.json
```

The likelihood file contains exactly `assignment_ids`, `log_likelihoods`, and
`likelihood_power`.

## Statistical and scientific boundary

The local log weights must be calibrated using complete source/calibration
objects or acquisition sessions. Exact global conditioning does not make those
weights calibrated, independent, or correct. It also does not establish
cross-window association precision, real-provider competence, physical-state
identifiability, BayesianPhysTwin benefit, Causal4D benefit, or deployment
safety.

Promotion requires an object/session-held-out comparison of framewise identity,
within-window persistence, pairwise hard links, independent local mixtures, the
joint posterior, an identity oracle used only for headroom, and exact physical
fallback. BayesianPhysTwin continues to own the downstream regret guard and
complete-belief fallback.
