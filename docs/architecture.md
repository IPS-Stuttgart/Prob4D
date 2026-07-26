# Prob4D architecture and repository boundary

Prob4D is an optional 4-D perception provider. It estimates uncertain `Sim(3)`
gauges for independently decoded MotionCrafter windows, calibrates conditional
point uncertainty, and exports observations without performing a physical-state
update.

The intended dependency direction is:

```text
MotionCrafter / alternative 4-D provider
    -> Prob4D provider API
    -> phys4d.observation_belief or ObservationFactorBundle
    -> Bayesian-PhysTwin guarded physical belief
    -> conditional Causal4D intervention analysis
```

Prob4D must not depend on Bayesian-PhysTwin or Causal4D at runtime. Those
repositories may validate the neutral artifacts independently, but should not
import Prob4D experiment helpers or underscore-prefixed modules.

## Stable provider API

Downstream development code should import `prob4d.provider_v1`. Version 1
exposes:

- causal source selection before any excluded payload is opened;
- the content-addressed metric-gauge anchor;
- the portable `phys4d.observation_belief` version-1 writer;
- the richer `ObservationFactorBundle` contract and serializer.

Root-package imports remain available for compatibility, but
`prob4d.provider_v1` is the versioned cross-repository boundary. A breaking
change requires a new provider module rather than silently changing version 1.
Frozen experiments should continue to record exact repository commits and input
artifact hashes even when they use a compatible provider API.

## Causal information boundary

Predictive exports use an exclusive `causal_frame_stop`. A selected window must
be independently decoded and every source frame used by that window must satisfy

```text
source_frame < causal_frame_stop.
```

Selection is performed from manifest metadata before payload loading. Alignment,
gauge estimation, overlap disagreement, uncertainty, and prior reliability are
then recomputed using only the admitted prefix. Appending future manifest entries
must not change the selected-source digest or the exported artifact.

## Covariance boundary

`ObservationBeliefV1` is a compact contract conditional on a fixed external
metric anchor. It stores conditional `3 x 3` point covariance and one coherent
rank-seven gauge factor per retained window. It does not represent the complete
joint cross-window gauge posterior; residual dependence is capped with
composite-likelihood weights.

`ObservationFactorBundle` is the lossless interface when a downstream estimator
needs explicit gauge nuisance blocks, an uncertain global anchor, or richer
factor-level provenance. Conditional covariance and gauge uncertainty must not be
added twice.

Association probability, source-side prior reliability, group nominal
probability, and composite weight are separate quantities. In particular,
overlap reliability is not reused as a nominal/outlier prior unless that prior
has been calibrated independently.

## Artifact ownership

Prob4D owns:

- MotionCrafter prediction manifests and decoded-window payloads;
- gauge and uncertainty calibration artifacts;
- portable observation beliefs and observation-factor bundles;
- provider-side benchmarks and run manifests.

Bayesian-PhysTwin owns physical priors, guarded updates, fallback behavior, and
accepted twin beliefs. Causal4D owns realized-intervention inference downstream
of an accepted, content-bound twin belief. Paper-facing tables and sealed result
manifests belong in the corresponding paper or reproduction repository rather
than the Python package.

## Experimental modules

The `prob4d-phystwin*` commands are integration experiments, not the stable
provider interface. They may evolve with a particular study. External consumers
should depend only on `prob4d.provider_v1` and the versioned artifact contracts.
