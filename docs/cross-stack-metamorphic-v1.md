# Cross-stack metamorphic invariants v1

This installed-wheel suite checks representation and coordinate invariants across
the Prob4D, BayesianPhysTwin, and Causal4D package boundary. It uses the
normative observation-contract vector published independently by all three
packages and runs from three freshly built wheels with the source trees hidden
from Python imports.

## Invariants

### Shared-root basis

For any orthogonal matrix \(Q\), a covariance root may be replaced by \(UQ\)
without changing the represented covariance:

\[
(UQ)(UQ)^\top = UU^\top.
\]

The suite rotates the shared low-rank root and requires the BayesianPhysTwin
gauge-aware batch to retain the same innovation and full marginal observation
covariance.

### Coordinate frame and units

For \(x' = Ax+t\), where \(A=sR\), the observation residual and covariance must
transform as

\[
r' = Ar,\qquad \Sigma' = A\Sigma A^\top.
\]

The suite applies a rigid frame change together with metres-to-millimetres
scaling. It transforms the observation belief, physical prediction, state and
query Jacobians, and response scale, then checks the complete conditional plus
shared covariance.

### Observation ordering

Permuting observation rows must only permute the innovation and corresponding
\(3\times3\) covariance blocks. It must not change the represented Gaussian or
silently treat the new order as new evidence.

### Exact fallback

A rejected BayesianPhysTwin candidate must deliver the unchanged baseline belief
to Causal4D. The guarded handoff must consume zero Prob4D observation evidence,
zero query covariance, and must reject attempts to alter any of those facts.

## Workflow

`.github/workflows/cross-stack-metamorphic.yml` checks out exact companion
revisions, builds one wheel from each repository, installs only those wheels in
a fresh environment, and runs the isolated suite with user and source-tree
imports disabled. The retained JUnit report, log, and wheel SHA-256 manifest make
the run auditable.

## Evidence boundary

Passing these tests establishes cross-package numerical and ownership
invariants. It does not establish real-provider competence, calibration,
BayesianPhysTwin physical benefit, Causal4D intervention benefit, deployment
safety, or state of the art.
