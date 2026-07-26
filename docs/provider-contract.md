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

## Python import boundary

Downstream development code should import `prob4d.provider_v1` instead of
underscore-prefixed modules or experiment helpers. It exposes causal source
selection, the metric-anchor contract, observation export and strict loading,
the richer factor bundle, and the provider manifest. A breaking change requires
a new versioned module. Exact Git revisions remain mandatory for frozen
experiments.

## Artifact semantics

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
archives, and content-address mismatches. `PredictionWindow` inputs are copied
and frozen after validation so caller-side mutation cannot alter a sealed source
object. The richer `ObservationFactorBundle` schema v3 remains available when
the consumer estimates gauge nuisance variables explicitly.

The grouped `prob4d` command is the preferred discoverable interface. Existing
`prob4d-*` commands remain available so historical run manifests and frozen
experiments retain their exact command lines.
