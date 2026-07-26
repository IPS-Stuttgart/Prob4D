# Prob4D observation-provider contract

`prob4d provider manifest` exposes the narrow producer capabilities that a
Bayesian estimator may rely on without importing Prob4D implementation modules.
The manifest records the installed package version, an optional exact Git
revision, supported artifact schema versions, positive capabilities, and
explicit nonclaims.

```bash
prob4d provider manifest \
  --provider-revision 0123456789abcdef0123456789abcdef01234567 \
  --output outputs/prob4d-provider.json
```

## Stable Python import

Downstream development code should import the versioned facade:

```python
from prob4d.provider_v1 import (
    export_observation_belief,
    load_observation_belief_export,
    prob4d_provider_manifest,
    select_causal_source,
)
```

`prob4d.provider_v1` owns the dependency on the private causal-source and
metric-anchor implementations. It exposes the portable observation contract,
strict loader, joint-gauge utilities, and rich factor-bundle contract under one
stable import path. Backward-compatible additions may be made on the 0.2.x
package line. Removing or changing required semantics needs a new provider
module/API version. Frozen experiments continue to record an exact Git revision;
the versioned facade is for upgradeable development environments.

The version-1 contract declares that Prob4D can:

- select independently decoded windows whose complete source interval precedes
  an exclusive causal cutoff;
- produce a content-addressed `ObservationBeliefV1` artifact and append-invariant
  causal-source digest;
- keep local conditional point covariance separate from shared gauge factors;
- propagate the fixed metric-anchor uncertainty and selected relative-gauge
  constraints into one joint cross-window `Sim(3)` covariance;
- export one shared low-rank latent factor whose window blocks preserve that
  covariance, with trace-audited rank reduction;
- retain association probability separately from residual-independent prior
  reliability; and
- bind metric coordinates to an independently checksummed fixed `Sim(3)` anchor.

The production default is a causal sequential spanning tree. It preserves the
uncertainty of the selected causal constraints without pretending that redundant
dense alignment edges are independent. The legacy fixed-lag covariance path is
available only as an explicitly acknowledged reconstruction control because its
current boundary treatment fixes marginalized gauges at posterior means.

The manifest does **not** claim that exported covariance has passed prospective
target calibration, that redundant dense-edge fusion is validated, or that a
valid observation artifact by itself improves a Bayesian physical twin.

## Immutable prediction inputs

`PredictionWindow` defensively copies every supplied NumPy array after
validation and marks the stored copy read-only. Later mutation of the caller's
arrays cannot change a source window already admitted to a content-addressed
workflow. Frame IDs, point maps, validity, flow/deformation masks, and explicit
rays therefore remain stable for hashing, alignment, uncertainty construction,
and export. Methods that need temporary writable data return or construct their
own copies.

Use `prob4d-validate-observation` to reject schema drift, array drift, malformed
archives, and content-address mismatches. The richer `ObservationFactorBundle`
schema v3 remains available when the consumer estimates gauge nuisance variables
explicitly. Downstream projects must still validate serialized artifacts and
source lineage independently. The provider manifest describes compatibility;
exact Git revisions remain mandatory for frozen experiments.

The grouped `prob4d` command is the preferred discoverable interface. Existing
`prob4d-*` commands remain available so historical run manifests and frozen
experiments retain their exact command lines.
