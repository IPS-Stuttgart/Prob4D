# DOT R11–R20 query-message real-source evaluation

This experiment evaluates prior-anchored Gaussian query messages on the public
DOT V29 rope data. It reuses the immutable routed-camera CUT3R provider artifact
from workflow run `33552798863`; CUT3R is not rerun.

## Question

Do two overlapping CUT3R windows contain useful information for a physical
material-point displacement query, and can query-space covariance intersection
retain that utility without the confidence inflation caused by an independence
assumption?

## Real query

For each complete sequence, the query is the 3-D displacement of every marker
that is valid in all compared provider runs from frame 3 to frame 4, divided by
the frame-3 rope span. The physical fallback is exact persistence, i.e. zero
displacement.

- Window A is aligned to marker truth on frames 1–3.
- Window B is aligned independently on frames 5–7.
- The continuous provider is aligned on frames 1–3.
- Frame 4 is never used for a held sequence's alignment or covariance fit.

The routed 2-D camera is `cam001`, `cam002`, or `cam005` according to the sealed
provider record. DOT's shared 3-D marker carrier remains the file labelled
`cam001`.

## Cross-validation and dependence

Every sequence is scored once. Biases, marginal error covariances, and the
window-A/window-B cross covariance are fitted from the other source sequences
with equal sequence weight. The joint cross covariance is shrunk and clipped in
canonical-correlation coordinates so the complete covariance remains positive
definite.

The sealed CUT3R artifact does not contain a covariance output. These moments are
therefore source-fitted error models, not native CUT3R uncertainty and not an
independent calibration claim.

## Methods

The registered comparison includes physical fallback; either window alone;
continuous CUT3R; dense empirical joint-correlated fusion; equal-weight and
log-determinant query-space covariance intersection; naive independent-message
addition; and a diagonal joint-covariance ablation.

The query-message implementation must reproduce each single-window posterior and
remain idempotent when one byte-identical message is repeated. A source result is
classified separately from these algebraic checks.

## Evaluation unit and outputs

The independent unit is a complete DOT sequence. Markers and coordinates are not
counted as independent experimental units. The retained artifact reports
sequence-level RMSE, Gaussian NLL, normalized NEES, nominal-90% coverage,
interval width, execute/fallback decisions, harmful execution, deployed decision
loss, message bytes, observed rank strata, and complete-sequence bootstrap
intervals.

## Custody boundary

Only the already-open `R11-20.zip` source archive is accessed. R21–R30 and
R31–R70 are neither enumerated nor opened. A favorable source result cannot
itself authorize confirmation; that would require a new reviewed protocol and
request frozen before any R21–R30 access.
