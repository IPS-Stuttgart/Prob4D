# Bounded finite-orbit certificates under uncertain geometry

Status: **secondary robustness evidence** on the 15 public real-trajectory recordings already used by the primary Tracking Cloth experiment. This is not a second independent confirmation and does not use a learned visual provider.

## Why this experiment matters

The primary result used exact motion-capture anchors to instantiate the controlled hidden SO(2) orbit. A practical system estimates those anchors. Simply inserting their point estimates into the orbit can be optimistic: an incorrectly oriented or displaced line can make the computed orbit too small and admit a query that is not identifiable for the unknown true geometry.

This experiment replaces that plug-in assumption with a deterministic outer certificate.

## Guarantee

For observed anchors \(\hat a,\hat b\) and probe \(\hat p\), suppose each true point lies inside a Euclidean ball of radius \(\epsilon\). Let \(\hat d=\|\hat b-\hat a\|\), \(\hat t\) be the observed axial coordinate, and \(\hat r\) the observed probe-to-line radius. When \(\hat d>2\epsilon\), normalized-vector perturbation and the triangle inequality give

\[
r \leq \hat r + 2\epsilon + |\hat t|\min\left(2,\frac{4\epsilon}{\hat d-2\epsilon}\right).
\]

If \(\hat d\leq2\epsilon\), the axis direction is not certified and the implementation fails closed with an infinite orbit-width bound. Consequently, any query accepted using this outer orbit is also accepted on the unknown true orbit, conditional on the declared point-error balls being valid.

`src/prob4d/bounded_axial_orbit.py` implements this bound without fitting to target outcomes. The tests include 2,048 randomized adversarial point perturbations, exact zero-error recovery, similarity equivariance, a plug-in underestimation counterexample, and the uninformative-anchor failure case.

## Public real-trajectory stress test

The error radius was swept from 0% to 10% of the true anchor separation. Sixteen deterministic boundary perturbations were evaluated for each of 1,803 cases from 15 recordings, giving 28,848 perturbation cases per error level. Decision thresholds were the 25th, 50th, and 75th percentiles of full radial-orbit width computed from 273 valid source cases in 24 shake/twist recordings. The recording file remained the bootstrap unit.

| Point error / anchor span | Plug-in containment | Certified containment | Certified width / truth | q50 plug-in false accept | q50 certified false accept | q50 certified false reject |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0% | 1.0000 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 0.5% | 0.5055 | 1.0000 | 1.064 | 0.0490 | 0.0000 | 0.1030 |
| 1.0% | 0.5057 | 1.0000 | 1.128 | 0.0890 | 0.0000 | 0.2851 |
| 2.0% | 0.5080 | 1.0000 | 1.261 | 0.1286 | 0.0000 | 0.4050 |
| 5.0% | 0.5259 | 1.0000 | 1.680 | 0.1507 | 0.0000 | 0.4493 |
| 10.0% | 0.5481 | 1.0000 | 2.470 | 0.1585 | 0.0000 | 0.4992 |

At every nonzero error level, the plug-in orbit contains the true orbit only about half the time. At 2% point error, it falsely accepts 12.86% of cases relative to the source-median width threshold. The certified outer orbit retains 100% containment and zero false acceptance across the complete sweep.

The guarantee is not free: at 2% error, the outer interval is 1.261 times as wide as the true interval on average and rejects 40.50% of cases that the oracle orbit would accept at the source-median threshold. At 10% error, mean width inflation reaches 2.470 and false rejection reaches 49.92%. This is the intended fail-closed tradeoff: transparent conservatism instead of hidden false acceptance.

The 95% recording-bootstrap interval for plug-in containment at 2% error is [0.5006, 0.5166]; for width inflation it is [1.2102, 1.3204]; and for q50 plug-in false acceptance it is [0.0612, 0.1990]. Certified containment and false acceptance remain exactly 1 and 0, respectively, because they follow from the valid error-ball construction rather than an empirical calibration claim.

## Unsupported second cohort

A separate header-only audit examined the 27 untouched Self-collision recordings without parsing trajectory values. No source/target marker-identity namespace shared three cloth markers. Those recordings were therefore not forced into an invalid correspondence-based replication and remain outcome-unopened.

This negative support finding prevents an apparently broader but scientifically invalid experiment. It does not count as evidence against finite-orbit gating.

## Interpretation and limits

The combined result addresses a central reviewer concern about the primary real-trajectory experiment: the orbit need not be known exactly. When its generator is uncertain but enclosed by declared point-error balls, an outer-orbit certificate preserves sound acceptance and fails closed when the axis is not identifiable.

The study remains controlled. Motion-capture trajectories provide the real geometry, while bounded perturbations model an imperfect upstream estimate. The method does not infer \(\epsilon\) from images, establish learned-provider competence, identify the full physical cloth state, or demonstrate BayesianPhysTwin task benefit. A learned provider must supply a defensible point-error bound before this certificate can be used operationally.

## Provenance

- Protocol: `6b372b78329a809aee508a68ffd51abeb945fc941c0db29ca996459ddf52f711`
- Result: `5afbb2ae23ab34f8cb1944ed39d9517c8d7233096de8ae2e07b822d0e7299552`
- Source seal: `2f85bfa0b31eb62af5eaff36c3cc9c0e1bb4c91f79d43e14dbda8e28a9959afc`
- Workflow run: `33564072262`
- Artifact: `9822421238`
- Artifact digest: `sha256:7d6ed4e938361f4cde0f4e3cbfcb12bba4036863934222e04730089925917868`
- Self-collision header audit: `8e062f5ac4a8521b51cf616986331614f1bc91b5ca1d4971ed7647a5c4bc2ed9`
