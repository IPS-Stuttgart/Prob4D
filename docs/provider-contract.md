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

Use `prob4d-validate-observation` to reject schema drift, array drift, malformed
archives, and content-address mismatches. The richer `ObservationFactorBundle`
schema v3 remains available when the consumer estimates gauge nuisance variables
explicitly. Downstream projects must still validate serialized artifacts and
source lineage independently. The provider manifest describes compatibility;
exact Git revisions remain mandatory for frozen experiments.

The grouped `prob4d` command is the preferred discoverable interface. Existing
`prob4d-*` commands remain available so historical run manifests and frozen
experiments retain their exact command lines.
