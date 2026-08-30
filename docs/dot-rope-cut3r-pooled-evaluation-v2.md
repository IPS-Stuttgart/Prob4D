# DOT rope CUT3R pooled evaluation v2

## Purpose

The sealed R01--R03 CUT3R provider bundle from workflow run `33329701704`
contains all nine registered prediction runs. The first marker-evaluation attempt
stopped before scoring because its helper required at least six valid markers in
every individual frame. That condition was stricter than the registered
multi-frame fitting problem and was not required by the numerical estimators.

This successor reuses the immutable provider bundle. It does not rerun CUT3R,
change a point prediction, tune a method from marker residuals, or open R04--R70.

## Evaluation-only repairs

### Exact image-to-point-map geometry

CUT3R resizes the longest image edge to 512 pixels and then takes a centered crop
whose dimensions are multiples of 16. For a source pixel center `(u, v)`, the
evaluator applies the same deterministic resize and crop before bilinear sampling
of the provider point map. It no longer assumes independent endpoint scaling in
both image axes.

### Pooled correspondence support

The overlap transform is fitted to the union of valid correspondences over
frames 3--5. The frozen support rule requires at least six common samples in
total and at least two nonempty frame clusters. Provider-to-marker metric fits
require six total samples over their registered frame group. A scoring-only
continuous prediction requires two total samples. Every per-frame count remains
in the evidence bundle.

The clustered bootstrap continues to resample whole frame clusters and markers
within those clusters. It therefore retains the original dependence treatment;
only the erroneous per-frame rejection is removed.

## Coordinate selection

A separately content-addressed marker-support audit may select one candidate from
the finite preregistered set of numeric columns and coordinate units. The scoring
request binds the audit run, audit ID, terminal decision, and selected candidate.
Ambiguous or support-negative audit decisions cannot authorize performance
scoring.

## Information custody

1. The normal-view RGB frames were opened only in the completed provider run.
2. The provider artifact was sealed before any marker record was opened.
3. The pooled evaluator downloads and verifies that exact artifact.
4. It opens only R01--R03 source marker files and computes the frozen metrics.
5. R04--R70 remain unopened.

The self-hosted job has read-only repository and Actions permissions and uploads
only result data, support diagnostics, and identities.

## Claim boundary

A completed run is source-development real-data evidence for CUT3R point-map
competence, overlap stitching, and the registered uncertainty closures on
R01--R03. It is not held-out transfer, independent calibration, BayesianPhysTwin
benefit, Causal4D benefit, deployment safety, or a state-of-the-art result.
