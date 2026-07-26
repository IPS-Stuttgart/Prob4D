# Causal Phys4D observation export

`prob4d-export-observation` converts independently decoded MotionCrafter overlap
windows into the shared `phys4d.observation_belief` schema consumed by
Bayesian-PhysTwin. The exporter is deliberately separate from reconstruction
fusion: it preserves window provenance, association probability, prior
reliability, correlation groups, and coherent gauge uncertainty.

## Causal prefix construction

The cutoff is exclusive. For `--causal-frame-stop-exclusive 134`, source frame
133 is admissible and source frame 134 is not.

The exporter reads `predictions.json` before opening prediction payloads. It
selects only overlap-window records whose complete half-open interval satisfies

```text
window.stop_frame <= causal_frame_stop_exclusive.
```

Excluded suffix payloads are never opened or hashed. Alignment, gauge
estimation, overlap-disagreement accumulation, uncertainty construction, and
row export are then rerun from the retained windows only. This prevents an
apparently pre-cutoff row from inheriting information from a window that also
read post-cutoff RGB.

The source artifact digest covers the causal manifest subset and every selected
window payload. It intentionally excludes later windows, absolute output paths,
and the full-sequence stop frame. Mutating an excluded suffix payload therefore
does not change the causal artifact ID, while mutating a selected payload does.

## Gauge posterior

The exporter selects one well-supported earlier overlap constraint for each new
window, forming a causal spanning tree. It propagates the complete joint gauge
covariance through this tree, including cross-window covariance. A deterministic
square root of that joint covariance is mapped through each point's `Sim(3)`
Jacobian and exported as one shared low-rank factor. Thus rows from different
windows retain coherent gauge covariance instead of receiving independent
marginal gauge factors.

`--max-gauge-rank` caps the exported factor rank for large sequences. The
retained covariance-trace fraction is stored in artifact metadata. When the
joint rank is below the cap, the representation is exact up to numerical
linearization.

## Coordinate modes

Every exported artifact declares one of two modes:

- `gauge_relative`: the first retained window defines an arbitrary reference
  gauge. Values are in `gauge_unit`, even though the version-1 schema retains
  its historical `*_m` array names. This mode does not authorize a metric
  physical-state update by itself. A downstream estimator must retain the
  unresolved global gauge as a nuisance variable or abstain.
- `metric_anchored`: the first window receives an independent metric `Sim(3)`
  prior from `--metric-anchor`. Metric claims are authorized only in this mode.

A metric-anchor NPZ contains exactly:

```text
mean:       7       # [log scale, rotation vector (3), translation (3)]
covariance: 7 x 7
```

The anchor file digest, mean, and covariance are included in the content
addressed metadata.

## Reliability and dependence

Association probability and prior reliability are separate arrays. Retained
valid pixels currently have association probability one. Prior reliability is
computed only from overlap disagreement and the declared observation covariance;
it never uses a Bayesian-PhysTwin physical innovation.

Rows sharing an absolute source frame form one effective correlation group.
The group composite weight caps the effective number of pixel identities, so
duplicating windows or rows cannot increase group power without bound.

The local `3 x 3` covariance excludes gauge uncertainty. Gauge uncertainty is
represented only by the shared low-rank factor, avoiding double counting.

## Commands

Gauge-relative export:

```bash
prob4d-export-observation \
  outputs/camera0/predictions.json \
  outputs/camera0/observation_belief.npz \
  --case-id double_stretch_sloth \
  --causal-frame-stop-exclusive 134 \
  --coordinate-mode gauge_relative \
  --pixel-stride 4 \
  --source-revision <prob4d-git-sha>
```

Metric-anchored export:

```bash
prob4d-export-observation \
  outputs/camera0/predictions.json \
  outputs/camera0/observation_belief_metric.npz \
  --case-id double_stretch_sloth \
  --causal-frame-stop-exclusive 134 \
  --coordinate-mode metric_anchored \
  --metric-anchor outputs/camera0/metric_anchor.npz \
  --source-revision <prob4d-git-sha>
```

Validate the schema, exact array set, and content address:

```bash
prob4d-validate-observation outputs/camera0/observation_belief.npz
```

The contract has the same golden artifact digest as Bayesian-PhysTwin. Causal4D
should bind only the selected Bayesian-PhysTwin belief and its actually consumed
observation artifact; Prob4D does not provide a direct Causal4D inference path.
