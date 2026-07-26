# Prob4D observation-provider contract

`prob4d provider manifest` exposes the narrow producer capabilities that a
Bayesian estimator may rely on without importing Prob4D implementation modules.
The manifest records the installed package version, an optional exact Git
revision, supported artifact schema versions, positive capabilities, and
explicit limitations.

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
- retain association probability separately from residual-independent prior
  reliability; and
- bind metric coordinates to an independently checksummed fixed `Sim(3)` anchor.

The manifest also states what the compact artifact does **not** establish:

- `ObservationBeliefV1` carries per-window marginal `Sim(3)` gauge factors, not
  the full joint cross-window gauge covariance;
- exported covariance has not thereby passed prospective target calibration;
- a valid observation artifact does not itself demonstrate improvement of a
  Bayesian physical twin.

The richer `ObservationFactorBundle` schema v3 remains available when the
consumer estimates gauge nuisance variables explicitly. Downstream projects
must still validate the serialized artifact and source lineage independently.
The provider manifest describes compatibility; exact Git revisions remain
mandatory for frozen experiments.

The grouped `prob4d` command is the preferred discoverable interface. Existing
`prob4d-*` commands remain available so historical run manifests and frozen
experiments retain their exact command lines.
