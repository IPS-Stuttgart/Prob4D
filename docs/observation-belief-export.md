# Causally sealed observation-belief export

Prob4D exports independently decoded prediction windows through the portable
`phys4d.observation_belief` version-1 container consumed by BayesianPhysTwin and
validated independently by Causal4D. Prob4D-specific statistical semantics are
versioned separately; the strict causal stream contract is version 2.

The export is distinct from reconstruction products. A row is admissible only
when the complete independently decoded source window lies before the exclusive
causal cutoff. The exporter opens only admitted payloads and recomputes
alignment, gauge estimation, overlap disagreement, uncertainty, and prior
reliability on that prefix.

## Metric gauge anchor

MotionCrafter points have an unresolved global `Sim(3)` gauge. Metric coordinates
and covariance require an independent prior for the first retained overlap
window:

```python
from prob4d.api.v2 import MetricGaugeAnchor, save_metric_gauge_anchor

anchor = MetricGaugeAnchor(
    window_id="window_0000",
    global_from_local=registration.transform,
    covariance=registration.covariance,
    coordinate_frame="phystwin-world",
    source_kind="prefix-only RGB-D registration",
    source_artifact_sha256=reference_prediction_payload_sha256,
    metadata={
        "calibration_split": "source-only",
        "calibration_artifact_sha256": calibration_artifact_sha256,
    },
)
save_metric_gauge_anchor("outputs/metric_gauge_anchor.json", anchor)
```

The anchor must bind the exact first admitted prediction payload and the external
registration/calibration artifact. Strict stream-v2 export rejects incomplete or
post-cutoff provenance. Nonzero anchor covariance is propagated into the shared
joint gauge factor.

## Claim-bearing command

```bash
prob4d observation export-calibrated \
  outputs/sequence/predictions.json \
  outputs/sequence/observation_belief.npz \
  --case-id sequence \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/metric_gauge_anchor.json \
  --gauge-covariance-calibration outputs/calibration/gauge.json \
  --point-uncertainty-calibration outputs/calibration/point.json \
  --pixel-stride 4 \
  --sampling-mode information_stratified \
  --max-gauge-rank 64 \
  --minimum-retained-gauge-trace 0.999 \
  --source-revision <full-prob4d-commit> \
  --summary-json outputs/sequence/observation_belief_summary.json

prob4d observation validate outputs/sequence/observation_belief.npz
```

Use `prob4d observation export-exploratory` only for explicitly labelled
uncalibrated, pointwise-fallback, alternate-root, or fixed-lag controls. The bare
`prob4d observation export` route prints guidance and runs no exporter.
Provider-v1 execution and `export-v1` were removed in Prob4D 0.5; pin Prob4D
0.4.1 to reproduce those runs.

`--causal-frame-stop` is exclusive. Unknown lineage schemas, path traversal,
inconsistent frame IDs, non-prefix selections, a mismatched anchor, incomplete
calibration provenance, or unverifiable runtime revision fail closed.

## Joint gauge posterior

The production default is a causal sequential spanning tree. Prob4D propagates
the metric-anchor covariance and every selected relative-alignment covariance
into one joint matrix, including cross-window covariance. Redundant edges are
reported but not fused under an unverified independence assumption.

A deterministic eigendecomposition produces a shared covariance root. Rank
truncation is accepted only when the retained observation-displacement covariance
trace passes the declared threshold. Fixed-lag mode carries an uncertainty-bearing
Schur-complement boundary prior but remains an explicitly labelled
reconstruction control because historical all-window cross-covariance is not
portable after marginalization.

## Artifact semantics

The archive binds metric 3-D means, conditional `3 x 3` covariance, absolute
frame/entity/view/window identities, association probability, prior reliability,
effective correlation groups, composite-likelihood weights, and shared low-rank
gauge factors.

A consumer that keeps gauge terms as explicit nuisance variables must use the
conditional covariance and must not add the shared factor covariance again.
Association probability describes decoded-pixel identity support, not physical
node association. Prior reliability is source-derived and independent of the
downstream physical innovation.

The descriptor, array roster, dtypes, shapes, and bytes are covered by the
artifact ID. Appending post-cutoff windows cannot change an already valid prefix
artifact.

## Cross-repository checks

Prob4D, BayesianPhysTwin, and Causal4D share an installed-wheel contract corpus.
BayesianPhysTwin independently revalidates the producer artifact and owns
physical-update guards. Causal4D consumes the resulting BayesianPhysTwin belief
and independently validates observation lineage before counterfactual use.

For release-facing interoperability evidence, run the
[three-repository installed-wheel release capsule](ecosystem-release-capsule.md).
